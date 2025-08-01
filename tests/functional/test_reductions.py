"""Tests for reduction functions in triton_lib.functional."""

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import tlib.functional as tlf
from tests.test_base import BaseTritonTest


class TestReductions(BaseTritonTest):
    """Test reduction function implementations."""

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_sum_basic(self, LENGTH, dtype_str, device):
        """Test basic sum functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.sum(x)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.sum(a, dim=-1)
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        if dtype_str == "float32":
            assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)
        else:
            assert torch.allclose(out, out_ref, atol=2e-3, rtol=2e-3)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_mean_basic(self, LENGTH, dtype_str, device):
        """Test basic mean functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.mean(x)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.mean(a)
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        if dtype_str == "float32":
            assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)
        else:
            assert torch.allclose(out, out_ref, atol=2e-3, rtol=2e-3)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_var_basic(self, LENGTH, dtype_str, device):
        """Test basic var functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.var(x)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.var(a, dim=-1, unbiased=False)
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-4, rtol=2e-4)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_std_basic(self, LENGTH, dtype_str, device):
        """Test basic std functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.std(x)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.std(a, unbiased=False)
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-4, rtol=2e-4)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_prod_basic(self, LENGTH, dtype_str, device):
        """Test basic prod functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.prod(x)
            tl.store(o_ptr, result)

        a = 0.9 + 0.1 * torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)  # Keep values close to 1
        out_ref = torch.prod(a)
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-4, rtol=2e-4)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_count_nonzero_basic(self, LENGTH, dtype_str, device):
        """Test basic count_nonzero functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.count_nonzero(x)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        a[::2] = 0  # Set every other element to zero
        out_ref = torch.count_nonzero(a)
        out = torch.zeros((), dtype=torch.int32, device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out.float(), out_ref.float(), atol=0, rtol=0)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_any_basic(self, LENGTH, dtype_str, device):
        """Test basic any functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.any(x)
            tl.store(o_ptr, result)

        a = torch.zeros(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        a[LENGTH // 2] = 1.0  # Set one element to non-zero
        out_ref = torch.any(a)
        out = torch.zeros((), dtype=torch.bool, device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert out == out_ref

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_all_basic(self, LENGTH, dtype_str, device):
        """Test basic all functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.all(x)
            tl.store(o_ptr, result)

        a = torch.ones(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.all(a)
        out = torch.zeros((), dtype=torch.bool, device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert out == out_ref

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_min_basic(self, LENGTH, dtype_str, device):
        """Test basic min functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.min(x)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.min(a)
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32", "float16"])
    def test_max_basic(self, LENGTH, dtype_str, device):
        """Test basic max functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.max(x)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.max(a)
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5)

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_argmin_basic(self, LENGTH, dtype_str, device):
        """Test basic argmin functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.argmin(x)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.argmin(a)
        out = torch.zeros((), dtype=torch.int32, device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert out == out_ref

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_argmax_basic(self, LENGTH, dtype_str, device):
        """Test basic argmax functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.argmax(x)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.argmax(a)
        out = torch.zeros((), dtype=torch.int32, device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert out == out_ref

    @pytest.mark.unit
    @pytest.mark.functional
    @pytest.mark.parametrize("LENGTH", [16, 32, 64])
    @pytest.mark.parametrize("dtype_str", ["float32"])
    def test_logsumexp_basic(self, LENGTH, dtype_str, device):
        """Test basic logsumexp functionality."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
            x = tl.load(x_ptr + tl.arange(0, LENGTH))
            result = tlf.logsumexp(x)
            tl.store(o_ptr, result)

        a = torch.rand(LENGTH, dtype=getattr(torch, dtype_str), device=device)
        out_ref = torch.logsumexp(a, dim=0)
        out = torch.zeros((), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](a, out, LENGTH=LENGTH)
        assert torch.allclose(out, out_ref, atol=2e-4, rtol=2e-4)
