"""Tests for binary operations in triton_lib.ops."""

import pytest
import torch
import triton
import triton.language as tl
import triton_lib as tlib
from tests.test_base import BaseTritonTest


class TestBinaryOps(BaseTritonTest):
    """Test binary operations in ops module."""

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_binary_op_add(self, LENGTH, dtype_str, device):
        """Test add binary operation functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlib.add("a, b -> a b", (x, x))
            tl.store(o_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :], x)

        a = torch.ones(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a[:, None] + a[None, :]
        out = torch.zeros((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_binary_op_broadcasting_1x1(self, LENGTH, dtype_str, device):
        """Test broadcasting in binary operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlib.add("a, a -> a", (x, x))
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a + a
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_binary_op_broadcasting_2x1(self, LENGTH, dtype_str, device):
        """Test broadcasting in binary operations."""

        @triton.jit
        def _kernel(a_ptr, b_ptr, o_ptr, LENGHT: tl.constexpr):
            a = tl.load(a_ptr + tl.arange(0, LENGHT)[:, None] + tl.arange(0, LENGHT)[None, :])
            b = tl.load(b_ptr + tl.arange(0, LENGHT))
            o = tlib.add("a b, b", (a, b))
            tl.store(o_ptr + tl.arange(0, LENGHT)[:, None] + tl.arange(0, LENGHT)[None, :], o)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        b = torch.ones(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out = torch.zeros((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, b, out, LENGHT=LENGTH)
        out_ref = a + b
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    def test_binary_op_broadcasting_1x2(self):
        """Test broadcasting in binary operations."""
        # TODO: Implement broadcasting tests
        pass

    @pytest.mark.unit
    def test_binary_op_broadcasting_3x2(self):
        """Test broadcasting in binary operations."""
        # TODO: Implement broadcasting tests
        pass
