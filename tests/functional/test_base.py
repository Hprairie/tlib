"""Tests for base operations in triton_lib.functional."""

import pytest
import torch
import triton
import triton.language as tl
import triton_lib.functional as tlf
from tests.test_base import BaseTritonTest


class TestBinaryOps(BaseTritonTest):
    """Test binary operation implementations."""

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_add_basic(self, LENGTH, dtype_str, device):
        """Test basic addition functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.add(x, x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a + a
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_sub_basic(self, LENGTH, dtype_str, device):
        """Test basic subtraction functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.subtract(x, x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a - a
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_mul_basic(self, LENGTH, dtype_str, device):
        """Test basic multiplication functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.multiply(x, x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a * a
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_true_div_basic(self, LENGTH, dtype_str, device):
        """Test basic division functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.true_divide(x, x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a / a
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_floor_div_basic(self, LENGTH, dtype_str, device):
        """Test basic floor division functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.floor_div(x, x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones(LENGTH, dtype=getattr(torch, dtype_str), device=device) + 1  # This is a little finicky
        out_ref = a // a
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_divide_basic(self, LENGTH, dtype_str, device):
        """Test basic divide functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.divide(x, x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a / a
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    def test_logical_and_basic(self, LENGTH, device):
        """Test basic logical and functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            y = tl.load(y_ptr + tl.arange(0, LENGHT))
            x = tlf.logical_and(x, y)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.randint(0, 2, (LENGTH,), dtype=torch.bool, device=device)
        b = torch.randint(0, 2, (LENGTH,), dtype=torch.bool, device=device)
        out_ref = a & b
        out = torch.zeros((LENGTH), dtype=torch.bool, device=device)
        _kernel[(1,)](a, b, out, LENGHT=LENGTH)
        assert torch.equal(out, out_ref)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    def test_logical_or_basic(self, LENGTH, device):
        """Test basic logical or functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            y = tl.load(y_ptr + tl.arange(0, LENGHT))
            x = tlf.logical_or(x, y)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.randint(0, 2, (LENGTH,), dtype=torch.bool, device=device)
        b = torch.randint(0, 2, (LENGTH,), dtype=torch.bool, device=device)
        out_ref = a | b
        out = torch.zeros((LENGTH), dtype=torch.bool, device=device)
        _kernel[(1,)](a, b, out, LENGHT=LENGTH)
        assert torch.equal(out, out_ref)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_less_basic(self, LENGTH, dtype_str, device):
        """Test basic less than functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            y = tl.load(y_ptr + tl.arange(0, LENGHT))
            result = tlf.less(x, y)
            tl.store(o_ptr + tl.arange(0, LENGHT), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a < b
        out = torch.zeros((LENGTH), dtype=torch.bool, device=device)
        _kernel[(1,)](a, b, out, LENGHT=LENGTH)
        assert torch.equal(out, out_ref)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_less_equal_basic(self, LENGTH, dtype_str, device):
        """Test basic less than or equal functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            y = tl.load(y_ptr + tl.arange(0, LENGHT))
            result = tlf.less_equal(x, y)
            tl.store(o_ptr + tl.arange(0, LENGHT), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a <= b
        out = torch.zeros((LENGTH), dtype=torch.bool, device=device)
        _kernel[(1,)](a, b, out, LENGHT=LENGTH)
        assert torch.equal(out, out_ref)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_greater_basic(self, LENGTH, dtype_str, device):
        """Test basic greater than functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            y = tl.load(y_ptr + tl.arange(0, LENGHT))
            result = tlf.greater(x, y)
            tl.store(o_ptr + tl.arange(0, LENGHT), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a > b
        out = torch.zeros((LENGTH), dtype=torch.bool, device=device)
        _kernel[(1,)](a, b, out, LENGHT=LENGTH)
        assert torch.equal(out, out_ref)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_greater_equal_basic(self, LENGTH, dtype_str, device):
        """Test basic greater than or equal functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            y = tl.load(y_ptr + tl.arange(0, LENGHT))
            result = tlf.greater_equal(x, y)
            tl.store(o_ptr + tl.arange(0, LENGHT), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a >= b
        out = torch.zeros((LENGTH), dtype=torch.bool, device=device)
        _kernel[(1,)](a, b, out, LENGHT=LENGTH)
        assert torch.equal(out, out_ref)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_equal_basic(self, LENGTH, dtype_str, device):
        """Test basic equality functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            y = tl.load(y_ptr + tl.arange(0, LENGHT))
            result = tlf.equal(x, y)
            tl.store(o_ptr + tl.arange(0, LENGHT), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a == b
        out = torch.zeros((LENGTH), dtype=torch.bool, device=device)
        _kernel[(1,)](a, b, out, LENGHT=LENGTH)
        assert torch.equal(out, out_ref)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_not_equal_basic(self, LENGTH, dtype_str, device):
        """Test basic not equal functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            y = tl.load(y_ptr + tl.arange(0, LENGHT))
            result = tlf.not_equal(x, y)
            tl.store(o_ptr + tl.arange(0, LENGHT), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = a != b
        out = torch.zeros((LENGTH), dtype=torch.bool, device=device)
        _kernel[(1,)](a, b, out, LENGHT=LENGTH)
        assert torch.equal(out, out_ref)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_maximum_basic(self, LENGTH, dtype_str, device):
        """Test basic maximum functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            y = tl.load(y_ptr + tl.arange(0, LENGHT))
            result = tlf.maximum(x, y)
            tl.store(o_ptr + tl.arange(0, LENGHT), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.maximum(a, b)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, b, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_minimum_basic(self, LENGTH, dtype_str, device):
        """Test basic minimum functionality."""

        @triton.jit
        def _kernel(x_ptr, y_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            y = tl.load(y_ptr + tl.arange(0, LENGHT))
            result = tlf.minimum(x, y)
            tl.store(o_ptr + tl.arange(0, LENGHT), result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        b = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.minimum(a, b)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, b, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)
