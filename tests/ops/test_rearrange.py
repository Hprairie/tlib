"""Tests for rearrange operations in triton_lib.ops."""

import pytest
import triton
import triton.language as tl
import triton_lib as tlib
import torch
from tests.test_base import BaseTritonTest


class TestRearrangeOps(BaseTritonTest):
    """Test rearrange operations in ops module."""

    @pytest.mark.unit
    @pytest.mark.parametrize("H,W", [(8, 16), (16, 32), (32, 64)])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_rearrange_transpose(self, H, W, dtype_str, device):
        """Test basic transpose operation."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, H: tl.constexpr, W: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, H)[:, None] * W + tl.arange(0, W)[None, :])
            x = tlib.rearrange("h w -> w h", x)
            tl.store(o_ptr + tl.arange(0, W)[:, None] * H + tl.arange(0, H)[None, :], x)

        a = torch.rand((H, W), dtype=getattr(torch, dtype_str), device=device)
        out_ref = a.transpose(0, 1)
        out = torch.zeros((W, H), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, H=H, W=W)
        assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("B,H,W", [(2, 8, 16), (4, 16, 32)])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_rearrange_flatten(self, B, H, W, dtype_str, device):
        """Test flattening operation."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, B: tl.constexpr, H: tl.constexpr, W: tl.constexpr):
            x = tl.load(
                x_ptr
                + tl.arange(0, B)[:, None, None] * H * W
                + tl.arange(0, H)[None, :, None] * W
                + tl.arange(0, W)[None, None, :]
            )
            x = tlib.rearrange("b h w -> b (h w)", x)
            tl.store(o_ptr + tl.arange(0, B)[:, None] * (H * W) + tl.arange(0, H * W)[None, :], x)

        a = torch.rand((B, H, W), dtype=getattr(torch, dtype_str), device=device)
        out_ref = a.reshape(B, H * W)
        out = torch.zeros((B, H * W), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, B=B, H=H, W=W)
        assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("B,HW", [(2, 64), (4, 128)])
    @pytest.mark.parametrize("H,W", [(8, 8), (16, 8)])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_rearrange_unflatten(self, B, HW, H, W, dtype_str, device):
        """Test unflattening operation."""
        if H * W != HW:
            pytest.skip("H * W must equal HW")

        @triton.jit
        def _kernel(x_ptr, o_ptr, B: tl.constexpr, HW: tl.constexpr, H: tl.constexpr, W: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, B)[:, None] * HW + tl.arange(0, HW)[None, :])
            x = tlib.rearrange("b (h w) -> b h w", x, tlib.dict(h=H, w=W))
            tl.store(
                o_ptr
                + tl.arange(0, B)[:, None, None] * H * W
                + tl.arange(0, H)[None, :, None] * W
                + tl.arange(0, W)[None, None, :],
                x,
            )

        a = torch.rand((B, HW), dtype=getattr(torch, dtype_str), device=device)
        out_ref = a.reshape(B, H, W)
        out = torch.zeros((B, H, W), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, B=B, HW=HW, H=H, W=W)
        assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)

    # @pytest.mark.unit
    # @pytest.mark.parametrize("B,C,H,W", [(2, 3, 8, 8), (1, 4, 16, 16)])
    # @pytest.mark.parametrize("dtype_str", ["float32"])
    # def test_rearrange_permute(self, B, C, H, W, dtype_str, device):
    #     """Test permutation of dimensions."""

    #     @triton.jit
    #     def _kernel(x_ptr, o_ptr, B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr):
    #         x = tl.load(
    #             x_ptr
    #             + tl.arange(0, B)[:, None, None, None] * C * H * W
    #             + tl.arange(0, C)[None, :, None, None] * H * W
    #             + tl.arange(0, H)[None, None, :, None] * W
    #             + tl.arange(0, W)[None, None, None, :]
    #         )
    #         x = tlib.rearrange("b c h w -> b h w c", x)
    #         tl.store(
    #             o_ptr
    #             + tl.arange(0, B)[:, None, None, None] * H * W * C
    #             + tl.arange(0, H)[None, :, None, None] * W * C
    #             + tl.arange(0, W)[None, None, :, None] * C
    #             + tl.arange(0, C)[None, None, None, :],
    #             x,
    #         )

    #     a = torch.rand((B, C, H, W), dtype=getattr(torch, dtype_str), device=device)
    #     out_ref = a.permute(0, 2, 3, 1)
    #     out = torch.zeros((B, H, W, C), dtype=getattr(torch, dtype_str), device=device)
    #     _kernel[(1,)](a, out, B=B, C=C, H=H, W=W)
    #     assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)

    # @pytest.mark.unit
    # @pytest.mark.parametrize("N", [16, 32])
    # @pytest.mark.parametrize("dtype_str", ["float32"])
    # def test_rearrange_patch_embedding(self, N, dtype_str, device):
    #     """Test patch embedding-like operation."""

    #     @triton.jit
    #     def _kernel(x_ptr, o_ptr, N: tl.constexpr):
    #         # Load 4x4 patches from NxN image
    #         P = 4
    #         x = tl.load(
    #             x_ptr
    #             + tl.arange(0, N // P)[:, None, None, None] * N * P
    #             + tl.arange(0, N // P)[None, :, None, None] * P
    #             + tl.arange(0, P)[None, None, :, None] * N
    #             + tl.arange(0, P)[None, None, None, :]
    #         )
    #         x = tlib.rearrange("h w p1 p2 -> (h w) (p1 p2)", x)
    #         tl.store(o_ptr + tl.arange(0, (N // P) * (N // P))[:, None] * (P * P) + tl.arange(0, P * P)[None, :], x)

    #     P = 4
    #     if N % P != 0:
    #         pytest.skip(f"N ({N}) must be divisible by patch size ({P})")

    #     a = torch.rand((N // P, N // P, P, P), dtype=getattr(torch, dtype_str), device=device)
    #     out_ref = a.reshape((N // P) * (N // P), P * P)
    #     out = torch.zeros(((N // P) * (N // P), P * P), dtype=getattr(torch, dtype_str), device=device)
    #     _kernel[(1,)](a, out, N=N)
    #     assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)

    # @pytest.mark.unit
    # @pytest.mark.parametrize("LENGTH", [16, 32])
    # @pytest.mark.parametrize("dtype_str", ["float32"])
    # def test_rearrange_multiple_tensors(self, LENGTH, dtype_str, device):
    #     """Test rearrange with multiple input tensors."""

    #     @triton.jit
    #     def _kernel(x_ptr, y_ptr, o1_ptr, o2_ptr, LENGTH: tl.constexpr):
    #         x = tl.load(x_ptr + tl.arange(0, LENGTH)[:, None] * LENGTH + tl.arange(0, LENGTH)[None, :])
    #         y = tl.load(y_ptr + tl.arange(0, LENGTH)[:, None] * LENGTH + tl.arange(0, LENGTH)[None, :])
    #         o1, o2 = tlib.rearrange("h w, h w -> w h, w h", (x, y))
    #         tl.store(o1_ptr + tl.arange(0, LENGTH)[:, None] * LENGTH + tl.arange(0, LENGTH)[None, :], o1)
    #         tl.store(o2_ptr + tl.arange(0, LENGTH)[:, None] * LENGTH + tl.arange(0, LENGTH)[None, :], o2)

    #     a = torch.rand((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
    #     b = torch.rand((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
    #     out1_ref = a.transpose(0, 1)
    #     out2_ref = b.transpose(0, 1)
    #     out1 = torch.zeros((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
    #     out2 = torch.zeros((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
    #     _kernel[(1,)](a, b, out1, out2, LENGTH=LENGTH)
    #     assert torch.allclose(out1, out1_ref, atol=1e-3, rtol=1e-3)
    #     assert torch.allclose(out2, out2_ref, atol=1e-3, rtol=1e-3)

    # @pytest.mark.unit
    # def test_rearrange_error_handling(self, device):
    #     """Test error handling for invalid rearrange descriptions."""

    #     @triton.jit
    #     def _kernel_invalid_brackets():
    #         x = tl.zeros((4, 4), dtype=tl.float32)
    #         # This should raise an error due to brackets
    #         x = tlib.rearrange("h w -> [h w]", x)

    #     @triton.jit
    #     def _kernel_mismatched_tensors():
    #         x = tl.zeros((4, 4), dtype=tl.float32)
    #         y = tl.zeros((4, 4), dtype=tl.float32)
    #         # This should raise an error due to tensor count mismatch
    #         x = tlib.rearrange("h w -> w h", (x, y))

    #     # Note: These tests would need to be run in a way that captures compilation errors
    #     # For now, we just verify the functions can be defined
    #     assert _kernel_invalid_brackets is not None
    #     assert _kernel_mismatched_tensors is not None

    # @pytest.mark.performance
    # def test_rearrange_performance(self):
    #     """Test performance of rearrange operations."""
    #     # TODO: Implement performance benchmarks comparing against torch operations
    #     pass

    # @pytest.mark.unit
    # @pytest.mark.parametrize("B,H,W,C", [(1, 8, 8, 3)])
    # @pytest.mark.parametrize("dtype_str", ["float32"])
    # def test_rearrange_complex_transformation(self, B, H, W, C, dtype_str, device):
    #     """Test complex rearrangement with grouping and regrouping."""

    #     @triton.jit
    #     def _kernel(x_ptr, o_ptr, B: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr):
    #         x = tl.load(
    #             x_ptr
    #             + tl.arange(0, B)[:, None, None, None] * H * W * C
    #             + tl.arange(0, H)[None, :, None, None] * W * C
    #             + tl.arange(0, W)[None, None, :, None] * C
    #             + tl.arange(0, C)[None, None, None, :]
    #         )
    #         # Rearrange to group spatial dimensions
    #         x = tlib.rearrange("b h w c -> b c (h w)", x)
    #         tl.store(
    #             o_ptr
    #             + tl.arange(0, B)[:, None, None] * C * H * W
    #             + tl.arange(0, C)[None, :, None] * H * W
    #             + tl.arange(0, H * W)[None, None, :],
    #             x,
    #         )

    #     a = torch.rand((B, H, W, C), dtype=getattr(torch, dtype_str), device=device)
    #     out_ref = a.permute(0, 3, 1, 2).reshape(B, C, H * W)
    #     out = torch.zeros((B, C, H * W), dtype=getattr(torch, dtype_str), device=device)
    #     _kernel[(1,)](a, out, B=B, H=H, W=W, C=C)
    #     assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)
