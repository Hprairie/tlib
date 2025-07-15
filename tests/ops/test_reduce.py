"""Tests for reduction operations in triton_lib.ops."""

import pytest
import torch
import triton
import triton.language as tl
import triton_lib as tlib
from tests.test_base import BaseTritonTest


class TestReduceOps(BaseTritonTest):
    """Test reduction operations."""

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_sum(self, LENGTH, dtype_str, device):
        """Test sum reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.sum("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.sum(a, dim=1)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_mean(self, LENGTH, dtype_str, device):
        """Test mean reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.mean("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.mean(a, dim=1)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_var(self, LENGTH, dtype_str, device):
        """Test var reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.var("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.var(a, dim=1)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_std(self, LENGTH, dtype_str, device):
        """Test std reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.std("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.std(a, dim=1)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_prod(self, LENGTH, dtype_str, device):
        """Test prod reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.prod("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.prod(a, dim=1)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_count_nonzero(self, LENGTH, dtype_str, device):
        """Test count nonzero reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.count_nonzero("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.count_nonzero(a, dim=1).to(dtype=getattr(torch, dtype_str))
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_any(self, LENGTH, dtype_str, device):
        """Test any reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.any("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.any(a, dim=1).to(dtype=getattr(torch, dtype_str))
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_all(self, LENGTH, dtype_str, device):
        """Test all reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.all("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.all(a, dim=1).to(dtype=getattr(torch, dtype_str))
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_max(self, LENGTH, dtype_str, device):
        """Test max reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.max("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.max(a, dim=1).values  # Fix this
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_min(self, LENGTH, dtype_str, device):
        """Test min reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.min("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.min(a, dim=1).values  # Fix this
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_argmax(self, LENGTH, dtype_str, device):
        """Test argmax reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.argmax("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.argmax(a, dim=1).to(dtype=getattr(torch, dtype_str))
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_reduce_argmin(self, LENGTH, dtype_str, device):
        """Test argmin reduction operations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
            x = tlib.argmax("a [b]", x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.argmax(a, dim=1).to(dtype=getattr(torch, dtype_str))
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    # @pytest.mark.unit
    # @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    # @pytest.mark.parametrize("dtype_str", ["float32"])
    # def test_reduce_logsumexp(self, LENGTH, dtype_str, device):
    #     """Test logsumexp reduction operations."""

    #     @triton.jit
    #     def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
    #         x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
    #         x = tlib.logsumexp("a [b]", x)
    #         tl.store(o_ptr + tl.arange(0, LENGHT), x)

    #     a = torch.ones((LENGTH, LENGTH), dtype=getattr(torch, dtype_str), device=device)
    #     out_ref = torch.logsumexp(a, dim=1)
    #     out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
    #     _kernel[(1,)](a, out, LENGHT=LENGTH)
    #     assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)
