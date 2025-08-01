"""Tests for activation functions in triton_lib.functional."""

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton_lib.functional as tlf
from tests.test_base import BaseTritonTest


class TestActivations(BaseTritonTest):
    """Test activation function implementations."""

    @pytest.mark.unit
    @pytest.mark.functional
    def test_relu_basic(self):
        """Test basic ReLU functionality."""
        pass

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_gelu_basic(self, LENGTH, dtype_str, device):
        """Test basic GELU functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.gelu(x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.gelu(a)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_gelu_tanh_basic(self, LENGTH, dtype_str, device):
        """Test basic GELU tanh functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.gelu_tanh(x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.gelu(a)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-3, rtol=2e-3)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_sigmoid_basic(self, LENGTH, dtype_str, device):
        """Test basic Sigmoid functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.sigmoid(x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.sigmoid(a)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_tanh_basic(self, LENGTH, dtype_str, device):
        """Test basic Tanh functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.tanh(x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.tanh(a)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_silu_basic(self, LENGTH, dtype_str, device):
        """Test basic silu functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.silu(x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.silu(a)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_silu_exp_basic(self, LENGTH, dtype_str, device):
        """Test basic silu exp2 functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGHT: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGHT))
            x = tlf.silu_exp2(x)
            tl.store(o_ptr + tl.arange(0, LENGHT), x)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = F.silu(a)
        out = torch.zeros((LENGTH), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGHT=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)
