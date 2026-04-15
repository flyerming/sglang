"""Unit tests for memory_pool_host.py fixes related to:
  - RuntimeError: Destination indices must be a CUDA tensor

Three call sites are covered, all sharing the same root cause:
`transfer_kv_all_layer_mla_lf_pf` requires `dst_indices` to reside on
CUDA, but when `--hicache-io-backend kernel` is combined with
`--hicache-mem-layout page_first`, the indices tensor may still live on
CPU.  The fix is to move `dst_indices` to the CUDA device before calling
the kernel.  Note: the correct target device is `self.device_pool.device`
(the CUDA device), NOT `self.device` which is the host (CPU) device.

Affected locations
------------------
1. ``MLATokenToKVPoolHost.backup_from_device_all_layer`` (line ~1043-1057)
   – MLA KV-cache, page_first layout, non-JIT kernel path.
   Fix: ``host_indices.to(self.device_pool.device)``
2. ``MambaPoolHost._copy_tensor_all_layers_lf_pf``        (line ~1465-1476)
   – Mamba state, page_first layout, kernel path.
   Fix: ``dst_indices.to(device)`` where ``device`` is ``self.device_pool.device``
3. ``NSAIndexerPoolHost.backup_from_device_all_layer``    (line ~2017-2032)
   – NSA indexer, page_first layout, kernel path.
   Fix: ``host_page_indices.to(self.device_pool.device)``
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, NSATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    ALLOC_MEMORY_FUNCS,
    MLATokenToKVPoolHost,
    NSAIndexerPoolHost,
    alloc_with_pin_memory,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=12, suite="stage-b-test-1-gpu-small-amd")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_unsupported(test_case: unittest.TestCase) -> None:
    """Skip the test when the required GPU / driver stack is absent."""
    if not torch.cuda.is_available():
        test_case.skipTest("CUDA is required for host-pool transfer tests.")
    if is_npu() or is_xpu():
        test_case.skipTest("Host transfer tests only support CUDA/ROCm.")
    if not (is_cuda() or is_hip()):
        test_case.skipTest("CUDA/ROCm device not available.")


def _token_indices_for_pages(
    pages: torch.Tensor, page_size: int, device: str
) -> torch.Tensor:
    """Expand page indices to token indices."""
    parts = [
        torch.arange(
            int(p) * page_size,
            (int(p) + 1) * page_size,
            device=device,
            dtype=torch.int64,
        )
        for p in pages.tolist()
    ]
    return torch.cat(parts, dim=0)


def _assert_cuda_promoted_indices(
    test_case: unittest.TestCase,
    mocked_transfer,
    expected_indices: torch.Tensor,
) -> None:
    mocked_transfer.assert_called_once()
    dst_indices = mocked_transfer.call_args.kwargs["dst_indices"]
    test_case.assertTrue(dst_indices.is_cuda)
    test_case.assertTrue(torch.equal(dst_indices.cpu(), expected_indices.cpu()))


# ---------------------------------------------------------------------------
# Fix 1 – MLATokenToKVPoolHost.backup_from_device_all_layer (page_first)
# ---------------------------------------------------------------------------


class TestMLAHostPageFirstCPUIndices(CustomTestCase):
    """
    Regression test for the fix at memory_pool_host.py lines 1043-1057.

    Before the fix, calling ``backup_from_device_all_layer`` with
    ``layout="page_first"`` and CPU-resident *host_indices* raised:
        RuntimeError: Destination indices must be a CUDA tensor
    The root cause: ``host_indices.to(self.device)`` moved the tensor to
    ``self.device`` which is ``"cpu"`` (the host pool's own device), not CUDA.
    The fix uses ``self.device_pool.device`` to target the correct CUDA device.
    """

    def setUp(self):
        _skip_if_unsupported(self)

    def _make_pools(self, page_size: int, layer_num: int, device_size: int):
        kv_lora_rank = 128
        qk_rope_head_dim = 32

        device_pool = MLATokenToKVPool(
            size=device_size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=torch.bfloat16,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
        )

        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            host_pool = MLATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=page_size,
                layout="page_first",
                pin_memory=True,
                device="cpu",
                allocator_type="default",
            )
            host_pool.can_use_jit = False
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        return device_pool, host_pool

    def _run(self, page_size: int):
        layer_num = 2
        device_size = page_size * 4

        device_pool, host_pool = self._make_pools(page_size, layer_num, device_size)

        # Fill device KV buffer with deterministic data.
        for lid in range(layer_num):
            buf = device_pool.kv_buffer[lid]
            data = torch.arange(buf.numel(), device="cuda", dtype=buf.dtype).view_as(
                buf
            )
            buf.copy_(data + lid)

        device_pages = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int64)
        # CPU indices – this is exactly the scenario that triggered the bug.
        host_pages = torch.tensor([0, 1, 2], device="cpu", dtype=torch.int64)

        device_indices = _token_indices_for_pages(device_pages, page_size, "cuda")
        host_indices = _token_indices_for_pages(host_pages, page_size, "cpu")

        # Must not raise RuntimeError about CUDA tensor requirement.
        host_pool.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend="kernel"
        )

        # Correctness: host buffer should now mirror the device data.
        # page_first layout: kv_buffer shape = (size, layer_num, 1, kv_cache_dim)
        # device kv_buffer[lid] shape = (size+page_size, 1, kv_cache_dim)
        for lid in range(layer_num):
            for h_page, d_page in zip(host_pages.tolist(), device_pages.tolist()):
                h_start = h_page * page_size
                d_start = d_page * page_size
                # host: tokens at [h_start:h_start+page_size, lid]
                got = host_pool.kv_buffer[
                    h_start : h_start + page_size, lid
                ].cpu()
                # device: tokens at [d_start:d_start+page_size] for layer lid
                expected = device_pool.kv_buffer[lid][
                    d_start : d_start + page_size
                ].cpu()
                self.assertTrue(
                    torch.equal(got, expected),
                    msg=f"Mismatch at layer={lid} host_page={h_page}",
                )

    def test_page_first_kernel_cpu_indices_page_size_1(self):
        """page_size=1 (ROCm-compatible path)."""
        self._run(page_size=1)

    def test_page_first_kernel_cpu_indices_page_size_16(self):
        """page_size=16 (typical CUDA path)."""
        if is_hip():
            self.skipTest("page_size > 1 not tested on ROCm")
        self._run(page_size=16)

    def test_page_first_kernel_promotes_cpu_indices_before_kernel_call(self):
        page_size = 1
        layer_num = 2
        device_pool, host_pool = self._make_pools(page_size, layer_num, page_size * 4)

        device_pages = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
        host_pages = torch.tensor([2, 3], device="cpu", dtype=torch.int64)
        device_indices = _token_indices_for_pages(device_pages, page_size, "cuda")
        host_indices = _token_indices_for_pages(host_pages, page_size, "cpu")

        with patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_kv_all_layer_mla_lf_pf"
        ) as mocked_transfer:
            host_pool.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )

        _assert_cuda_promoted_indices(self, mocked_transfer, host_indices)

    def test_page_first_kernel_keeps_cuda_indices_when_already_on_device(self):
        page_size = 1
        layer_num = 2
        device_pool, host_pool = self._make_pools(page_size, layer_num, page_size * 4)

        device_pages = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
        host_pages = torch.tensor([2, 3], device="cuda", dtype=torch.int64)
        device_indices = _token_indices_for_pages(device_pages, page_size, "cuda")
        host_indices = _token_indices_for_pages(host_pages, page_size, "cuda")

        with patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_kv_all_layer_mla_lf_pf"
        ) as mocked_transfer:
            host_pool.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )

        mocked_transfer.assert_called_once()
        self.assertIs(mocked_transfer.call_args.kwargs["dst_indices"], host_indices)


# ---------------------------------------------------------------------------
# Fix 2 – MambaPoolHost._copy_tensor_all_layers_lf_pf (page_first)
# ---------------------------------------------------------------------------


class TestMambaHostPageFirstCPUIndices(CustomTestCase):
    """
    Regression test for the fix at memory_pool_host.py lines 1465-1476.

    ``MambaPoolHost._copy_tensor_all_layers_lf_pf`` is the exact helper that
    now promotes CPU ``dst_indices`` to CUDA before invoking the kernel.
    The fix correctly uses the ``device`` parameter (which callers pass as
    ``self.device_pool.device``) rather than ``self.device`` (which is CPU).
    These tests patch the kernel entrypoint directly so they stay focused and fast.
    """

    def setUp(self):
        _skip_if_unsupported(self)

    @staticmethod
    def _build_copy_inputs():
        num_layers = 2
        num_tokens = 4
        feat = 8
        src_layers = [
            torch.arange(num_tokens * feat, device="cuda", dtype=torch.float32).reshape(
                num_tokens, feat
            )
            * (layer_id + 1)
            for layer_id in range(num_layers)
        ]
        dst = torch.zeros(
            num_tokens, num_layers, feat, dtype=torch.float32, pin_memory=True
        )
        src_indices = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
        return src_layers, dst, src_indices, num_layers

    def test_static_helper_promotes_cpu_indices_before_kernel_call(self):
        from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost

        src_layers, dst, src_indices, num_layers = self._build_copy_inputs()
        dst_indices = torch.tensor([2, 3], device="cpu", dtype=torch.int64)

        with patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_kv_all_layer_mla_lf_pf"
        ) as mocked_transfer:
            MambaPoolHost._copy_tensor_all_layers_lf_pf(
                src_layers=src_layers,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                num_layers=num_layers,
                device="cuda",
                io_backend="kernel",
            )

        _assert_cuda_promoted_indices(self, mocked_transfer, dst_indices)

    def test_static_helper_keeps_cuda_indices_when_already_on_device(self):
        from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost

        src_layers, dst, src_indices, num_layers = self._build_copy_inputs()
        dst_indices = torch.tensor([2, 3], device="cuda", dtype=torch.int64)

        with patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_kv_all_layer_mla_lf_pf"
        ) as mocked_transfer:
            MambaPoolHost._copy_tensor_all_layers_lf_pf(
                src_layers=src_layers,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                num_layers=num_layers,
                device="cuda",
                io_backend="kernel",
            )

        mocked_transfer.assert_called_once()
        self.assertIs(mocked_transfer.call_args.kwargs["dst_indices"], dst_indices)


# ---------------------------------------------------------------------------
# Fix 3 – NSAIndexerPoolHost.backup_from_device_all_layer (page_first)
# ---------------------------------------------------------------------------


class TestNSAIndexerPageFirstCPUIndices(CustomTestCase):
    """
    Regression test for the fix at memory_pool_host.py lines 2017-2032.

    ``NSAIndexerPoolHost.backup_from_device_all_layer`` with
    ``layout="page_first"`` now moves *_dst_page_indices* to CUDA before
    passing it to ``transfer_kv_all_layer_mla_lf_pf``.
    The root cause was the same as Fix 1: ``host_page_indices.to(self.device)``
    used ``self.device`` (``"cpu"``).  The fix uses ``self.device_pool.device``.
    """

    def setUp(self):
        _skip_if_unsupported(self)

    def _make_pools(self, page_size: int, layer_num: int, device_size: int):
        kv_lora_rank = 128
        qk_rope_head_dim = 32
        # NSATokenToKVPool enforces page_size==1 on ROCm and page_size==64 on CUDA.
        assert page_size in (1, 64), f"NSA requires page_size 1 or 64, got {page_size}"

        device_pool = NSATokenToKVPool(
            size=device_size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=torch.bfloat16,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=kv_lora_rank + qk_rope_head_dim,
            index_head_dim=128,
        )

        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            mla_host = MLATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=page_size,
                layout="page_first",
                pin_memory=True,
                device="cpu",
                allocator_type="default",
                override_kv_cache_dim=device_pool.kv_cache_dim,
            )
            mla_host.can_use_jit = False
            indexer_host = NSAIndexerPoolHost(
                device_pool=device_pool,
                anchor_host=mla_host,
                layout="page_first",
                pin_memory=True,
                device="cpu",
                allocator_type="default",
            )
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        return device_pool, mla_host, indexer_host

    def _run(self, page_size: int):
        layer_num = 2
        device_size = page_size * 4

        device_pool, mla_host, indexer_host = self._make_pools(
            page_size, layer_num, device_size
        )

        # Fill device indexer buffer with deterministic data.
        for lid in range(layer_num):
            buf = device_pool.index_k_with_scale_buffer[lid]
            data = torch.arange(
                buf.numel(), device=buf.device, dtype=torch.uint8
            ).view_as(buf)
            buf.copy_((data + lid) % 256)
            kv_buf = device_pool.kv_buffer[lid]
            kv_data = torch.arange(
                kv_buf.numel(), device=kv_buf.device, dtype=kv_buf.dtype
            ).view_as(kv_buf)
            kv_buf.copy_(kv_data + lid)

        device_pages = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int64)
        # CPU host pages – the bug trigger for this code path.
        host_pages = torch.tensor([0, 1, 2], device="cpu", dtype=torch.int64)

        device_indices = _token_indices_for_pages(device_pages, page_size, "cuda")
        host_indices = _token_indices_for_pages(host_pages, page_size, "cpu")

        # Must not raise RuntimeError about CUDA tensor requirement.
        mla_host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend="kernel"
        )
        indexer_host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend="kernel"
        )

        # Correctness: indexer host buffer should mirror device data.
        # host pool layout=page_first:
        #   index_k_with_scale_buffer shape = (page_num, layer_num, 1, stride)
        #   access: buffer[h_page][lid][0]  -> shape (stride,)
        # device pool (layer_first list):
        #   index_k_with_scale_buffer[lid] shape = (num_pages, stride)
        #   access: buffer[lid][d_page]     -> shape (stride,)
        for lid in range(layer_num):
            for h_page, d_page in zip(host_pages.tolist(), device_pages.tolist()):
                got = indexer_host.index_k_with_scale_buffer[h_page][lid][0].cpu()
                expected = device_pool.index_k_with_scale_buffer[lid][d_page].cpu()
                self.assertTrue(
                    torch.equal(got, expected),
                    msg=f"Indexer mismatch at layer={lid} host_page={h_page}",
                )

                h_start = h_page * page_size
                d_start = d_page * page_size
                # page_first layout:
                # kv_buffer shape = (size, layer_num, 1, kv_cache_dim)
                got_kv = mla_host.kv_buffer[
                    h_start : h_start + page_size, lid
                ].cpu()
                expected_kv = device_pool.kv_buffer[lid][
                    d_start : d_start + page_size
                ].cpu()
                self.assertTrue(
                    torch.equal(got_kv, expected_kv),
                    msg=f"KV mismatch at layer={lid} host_page={h_page}",
                )

    def test_page_first_kernel_cpu_indices_page_size_1(self):
        """page_size=1 (ROCm-compatible path)."""
        if not is_hip():
            self.skipTest("page_size=1 NSA test only applies to ROCm")
        self._run(page_size=1)

    def test_page_first_kernel_cpu_indices_page_size_64(self):
        """page_size=64 (CUDA path)."""
        if is_hip():
            self.skipTest("page_size=64 NSA test only applies to CUDA")
        self._run(page_size=64)

    def test_page_first_kernel_promotes_page_indices_before_kernel_call(self):
        page_size = 1 if is_hip() else 64
        layer_num = 2
        device_pool, _, indexer_host = self._make_pools(
            page_size, layer_num, page_size * 6
        )

        device_pages = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
        host_pages = torch.tensor([2, 3], device="cpu", dtype=torch.int64)
        device_indices = _token_indices_for_pages(device_pages, page_size, "cuda")
        host_indices = _token_indices_for_pages(host_pages, page_size, "cpu")

        with patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_kv_all_layer_mla_lf_pf"
        ) as mocked_transfer:
            indexer_host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )

        _assert_cuda_promoted_indices(self, mocked_transfer, host_pages)

    def test_page_first_kernel_keeps_cuda_page_indices_when_already_on_device(self):
        page_size = 1 if is_hip() else 64
        layer_num = 2
        device_pool, _, indexer_host = self._make_pools(
            page_size, layer_num, page_size * 6
        )

        device_pages = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
        host_pages = torch.tensor([2, 3], device="cuda", dtype=torch.int64)
        device_indices = _token_indices_for_pages(device_pages, page_size, "cuda")
        host_indices = _token_indices_for_pages(host_pages, page_size, "cuda")

        with patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_kv_all_layer_mla_lf_pf"
        ) as mocked_transfer:
            indexer_host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )

        mocked_transfer.assert_called_once()
        dst_indices = mocked_transfer.call_args.kwargs["dst_indices"]
        self.assertTrue(dst_indices.is_cuda)
        self.assertTrue(torch.equal(dst_indices, host_pages))


if __name__ == "__main__":
    unittest.main()