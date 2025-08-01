"""Tests for unary operations in triton_lib.functional."""

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import tlib.functional as tlf
from tests.test_base import BaseTritonTest


class TestUnaryOps(BaseTritonTest):
    """Test unary operation implementations."""

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_cumsum_basic(self, LENGTH, dtype_str, device):
        """Test basic cumsum functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.cumsum(x)
            tl.store(o_ptr + tl.arange(0, LENGTH), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.cumsum(a, dim=0)
        out = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        if dtype_str == "float32":
            assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)
        else:
            assert torch.allclose(out, out_ref, atol=2e-3, rtol=2e-3)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_cumprod_basic(self, LENGTH, dtype_str, device):
        """Test basic cumprod functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.cumprod(x)
            tl.store(o_ptr + tl.arange(0, LENGTH), result)

        a = 0.9 + 0.1 * torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)  # Keep values close to 1
        out_ref = torch.cumprod(a, dim=0)
        out = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        if dtype_str == "float32":
            assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)
        else:
            assert torch.allclose(out, out_ref, atol=2e-3, rtol=2e-3)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_flip_basic(self, LENGTH, dtype_str, device):
        """Test basic flip functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.flip(x)
            tl.store(o_ptr + tl.arange(0, LENGTH), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.flip(a, dims=[0])
        out = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_sort_basic(self, LENGTH, dtype_str, device):
        """Test basic sort functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.sort(x)
            tl.store(o_ptr + tl.arange(0, LENGTH), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.sort(a, dim=0)[0]  # torch.sort returns (values, indices)
        out = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_sort_descending_basic(self, LENGTH, dtype_str, device):
        """Test basic sort descending functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.sort(x, descending=True)
            tl.store(o_ptr + tl.arange(0, LENGTH), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.sort(a, dim=0, descending=True)[0]  # torch.sort returns (values, indices)
        out = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    # @pytest.mark.unit
    # @pytest.mark.functional
    # @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    # @pytest.mark.parametrize("dtype_str", ["float32"])
    # def test_associative_scan_sum_basic(self, LENGTH, dtype_str, device):
    #     """Test basic associative scan with sum functionality."""

    #     @triton.jit
    #     def _sum_combine(x, y):
    #         return x + y

    #     @triton.jit
    #     def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
    #         x = tl.load(x_ptr + tl.arange(0, LENGTH))
    #         result = tlf.associative_scan(x, _sum_combine)
    #         tl.store(o_ptr + tl.arange(0, LENGTH), result)

    #     a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
    #     out_ref = torch.cumsum(a, dim=0)  # Associative scan with sum is equivalent to cumsum
    #     out = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
    #     _kernel[(1,)](a, out, LENGTH=LENGTH)
    #     assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    # @pytest.mark.unit
    # @pytest.mark.functional
    # @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    # @pytest.mark.parametrize("dtype_str", ["float32"])
    # def test_associative_scan_max_basic(self, LENGTH, dtype_str, device):
    #     """Test basic associative scan with max functionality."""

    #     @triton.jit
    #     def _max_combine(x, y):
    #         return tl.maximum(x, y)

    #     @triton.jit
    #     def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
    #         x = tl.load(x_ptr + tl.arange(0, LENGTH))
    #         result = tlf.associative_scan(x, _max_combine)
    #         tl.store(o_ptr + tl.arange(0, LENGTH), result)

    #     a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
    #     out_ref = torch.cummax(a, dim=0)[0]  # torch.cummax returns (values, indices)
    #     out = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
    #     _kernel[(1,)](a, out, LENGTH=LENGTH)
    #     assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_softmax_basic(self, LENGTH, dtype_str, device):
        """Test basic softmax functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.softmax(x)
            tl.store(o_ptr + tl.arange(0, LENGTH), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.softmax(a, dim=0)
        out = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_log_softmax_basic(self, LENGTH, dtype_str, device):
        """Test basic log_softmax functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.log_softmax(x)
            tl.store(o_ptr + tl.arange(0, LENGTH), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.log_softmax(a, dim=0)
        out = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)
