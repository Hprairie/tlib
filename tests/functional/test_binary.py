"""Tests for binary operations in triton_lib.functional."""

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import tlib.functional as tlf
from tests.test_base import BaseTritonTest


class TestBinaryOps(BaseTritonTest):
    """Test binary operation implementations."""

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_kl_div_basic(self, LENGTH, dtype_str, device):
        """Test basic KL divergence functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            y = tl.load(y_ptr + tl.arange(0, LENGTH))
            result = tlf.kl_div(x, y)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.kl_div(a, b, reduction="mean")
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, b, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_kl_div_sum_reduction(self, LENGTH, dtype_str, device):
        """Test KL divergence with sum reduction."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            y = tl.load(y_ptr + tl.arange(0, LENGTH))
            result = tlf.kl_div(x, y, reduction="sum")
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.kl_div(a, b, reduction="sum")
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, b, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_kl_div_none_reduction(self, LENGTH, dtype_str, device):
        """Test KL divergence with none reduction."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            y = tl.load(y_ptr + tl.arange(0, LENGTH))
            result = tlf.kl_div(x, y, reduction="none")
            tl.store(o_ptr + tl.arange(0, LENGTH), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.kl_div(a, b, reduction="none")
        out = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, b, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_kl_div_log_target(self, LENGTH, dtype_str, device):
        """Test KL divergence with log_target=True."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            y = tl.load(y_ptr + tl.arange(0, LENGTH))
            result = tlf.kl_div(x, y, log_target=True)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.kl_div(a, b, reduction="mean", log_target=True)
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, b, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_mse_basic(self, LENGTH, dtype_str, device):
        """Test basic MSE functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            y = tl.load(y_ptr + tl.arange(0, LENGTH))
            result = tlf.mse(x, y)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.mse_loss(a, b, reduction="mean")
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, b, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_mse_sum_reduction(self, LENGTH, dtype_str, device):
        """Test MSE with sum reduction."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            y = tl.load(y_ptr + tl.arange(0, LENGTH))
            result = tlf.mse(x, y, reduction="sum")
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.mse_loss(a, b, reduction="sum")
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, b, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_mse_none_reduction(self, LENGTH, dtype_str, device):
        """Test MSE with none reduction."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            y = tl.load(y_ptr + tl.arange(0, LENGTH))
            result = tlf.mse(x, y, reduction="none")
            tl.store(o_ptr + tl.arange(0, LENGTH), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.mse_loss(a, b, reduction="none")
        out = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, b, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    # @pytest.mark.unit
    # @pytest.mark.functional
    # @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    # @pytest.mark.parametrize("dtype_str", ["float32"])
    # def test_cross_entropy_basic(self, LENGTH, dtype_str, device):
    #     """Test basic cross entropy functionality."""

    #     @triton.jit
    #     def _kernel(x_ptr, y_ptr, o_ptr, LENGTH: tl.constexpr):
    #         x = tl.load(x_ptr + tl.arange(0, LENGTH))
    #         y = tl.load(y_ptr + tl.arange(0, LENGTH))
    #         result = tlf.cross_entropy(x, y)
    #         tl.store(o_ptr, result)

    #     a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
    #     b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
    #     b = b / b.sum()  # Normalize to make it a valid probability distribution
    #     out_ref = F.cross_entropy(a.unsqueeze(0), b.unsqueeze(0))
    #     out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
    #     _kernel[(1,)](a, b, out, LENGTH=LENGTH)
    #     assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)
