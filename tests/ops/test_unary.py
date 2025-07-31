"""Tests for unary operations in triton_lib.ops."""

import pytest
import triton
import triton.language as tl
import triton_lib as tlib
import torch
from tests.test_base import BaseTritonTest


class TestUnaryOps(BaseTritonTest):
    """Test unary operations in ops module."""

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_cumsum(self, LENGTH, dtype_str, device):
        """Test cumsum reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.cumsum("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :], x)

        a = torch.rand((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.cumsum(a, dim=1)
        out = torch.zeros((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-3, rtol=2e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_reduce_cumprod(self, LENGTH, dtype_str, device):
        """Test cumprod reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.cumprod("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :], x)

        a = torch.arange(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        a = a[:, None] + a[None, :]
        out_ref = torch.cumprod(a, dim=1)
        out = torch.zeros((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-3, rtol=2e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_flip(self, LENGTH, dtype_str, device):
        """Test flip reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.flip("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :], x)

        a = torch.arange(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        a = a[:, None] * LENGTH + a[None, :]
        out_ref = torch.flip(a, dims=(1,))
        out = torch.zeros((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-3, rtol=2e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_sort(self, LENGTH, dtype_str, device):
        """Test sort reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.sort("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :], x)

        a = torch.rand((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.sort(a, dim=1).values
        out = torch.zeros((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-3, rtol=2e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_softmax(self, LENGTH, dtype_str, device):
        """Test softmax reduction operations."""
        pass

        # @triton.jit
        # def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
        #     x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
        #     x = tlib.softmax("a [b]", x)
        #     tl.store(o_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :], x)

        # a = torch.rand((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        # out_ref = torch.softmax(a, dim=1)
        # out = torch.zeros((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        # _kernel[(1,)](a, out, LENGHT=LENGTH)
        # assert torch.allclose(out, out_ref, atol=2e-3, rtol=2e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_associative_scan(self, LENGTH, dtype_str, device):
        """Test associative scan reduction operations."""
        pass

        # @triton.jit
        # def _sum(x, y):
        #     return x + y

        # @triton.jit
        # def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
        #     x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
        #     x = tlib.associative_scan("a [b]", x, combine_fn=_sum)
        #     tl.store(o_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :], x)

        # a = torch.rand((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        # out_ref = torch.cumsum(a, dim=-1)
        # out = torch.zeros((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        # _kernel[(1,)](a, out, LENGHT=LENGTH)
        # assert torch.allclose(out, out_ref, atol=2e-3, rtol=2e-3)

    @pytest.mark.performance
    def test_unary_op_performance(self):
        """Test performance of unary operations."""
        # TODO: Implement performance tests
        pass
