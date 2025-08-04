"""Tests for arange operations in triton_lib.ops."""

import pytest
import triton
import triton.language as tl
import tlib
import torch
import numpy as np
from tests.test_base import BaseTritonTest


class TestArangeOps(BaseTritonTest):
    """Test arange operations in ops module."""

    @pytest.mark.unit
    @pytest.mark.parametrize("length", [8, 16, 32])
    @pytest.mark.parametrize("dtype_str", ["int32", "int64"])
    def test_arange_1d(self, length, dtype_str, device):
        """Test basic 1D arange operation."""

        @triton.jit
        def _kernel(o_ptr, LENGTH: tl.constexpr):
            indices = tlib.arange("a", tlib.dict(a=LENGTH))
            tl.store(o_ptr + tl.arange(0, LENGTH), indices)

        out_ref = torch.arange(length, dtype=getattr(torch, dtype_str), device=device)
        out = torch.zeros(length, dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](out, LENGTH=length)
        assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("H,W", [(8, 16), (16, 32), (4, 8)])
    @pytest.mark.parametrize("dtype_str", ["int32"])
    def test_arange_2d(self, H, W, dtype_str, device):
        """Test 2D arange operation for indexing."""

        @triton.jit
        def _kernel(o_ptr, H: tl.constexpr, W: tl.constexpr):
            indices = tlib.arange("h w", tlib.dict(h=H, w=W))
            tl.store(o_ptr + tl.arange(0, H)[:, None] * W + tl.arange(0, W)[None, :], indices)

        # Create reference: h*W + w for each position
        h_indices, w_indices = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        out_ref = (h_indices * W + w_indices).to(getattr(torch, dtype_str))
        out = torch.zeros((H, W), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](out, H=H, W=W)
        assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("B,H,W", [(2, 4, 8), (1, 8, 8)])
    @pytest.mark.parametrize("dtype_str", ["int32"])
    def test_arange_3d(self, B, H, W, dtype_str, device):
        """Test 3D arange operation for multi-dimensional indexing."""

        @triton.jit
        def _kernel(o_ptr, B: tl.constexpr, H: tl.constexpr, W: tl.constexpr):
            indices = tlib.arange("b h w", tlib.dict(b=B, h=H, w=W))
            tl.store(
                o_ptr
                + tl.arange(0, B)[:, None, None] * H * W
                + tl.arange(0, H)[None, :, None] * W
                + tl.arange(0, W)[None, None, :],
                indices,
            )

        # Create reference: b*H*W + h*W + w for each position
        b_indices, h_indices, w_indices = torch.meshgrid(
            torch.arange(B, device=device),
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        out_ref = (b_indices * H * W + h_indices * W + w_indices).to(getattr(torch, dtype_str))
        out = torch.zeros((B, H, W), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](out, B=B, H=H, W=W)
        assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("N", [16, 32])
    @pytest.mark.parametrize("dtype_str", ["int32"])
    def test_arange_memory_indexing(self, N, dtype_str, device):
        """Test arange for memory pointer calculations."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, N: tl.constexpr):
            # Use arange to calculate memory offsets
            offsets = tlib.arange("n", tlib.dict(n=N))
            x = tl.load(x_ptr + offsets)
            # Store values at the same indices they were loaded from
            tl.store(o_ptr + offsets, x)

        x = torch.arange(N, dtype=torch.float32, device=device)
        out = torch.zeros(N, dtype=torch.float32, device=device)
        _kernel[(1,)](x, out, N=N)
        assert torch.allclose(out, x, atol=1e-3, rtol=1e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("H,W", [(8, 8), (16, 8)])
    @pytest.mark.parametrize("dtype_str", ["int32"])
    def test_arange_matrix_indexing(self, H, W, dtype_str, device):
        """Test arange for matrix element access patterns."""

        @triton.jit
        def _kernel(x_ptr, o_ptr, H: tl.constexpr, W: tl.constexpr):
            # Use arange to create row-major indexing
            row_indices = tlib.arange("h", tlib.dict(h=H))
            col_indices = tlib.arange("w", tlib.dict(w=W))

            # Load matrix elements using computed indices
            x = tl.load(x_ptr + row_indices[:, None] * W + col_indices[None, :])

            # Store elements using arange-generated offsets
            matrix_offsets = tlib.arange("h w", tlib.dict(h=H, w=W))
            tl.store(o_ptr + matrix_offsets, x)

        x = torch.arange(H * W, dtype=torch.float32, device=device).reshape(H, W)
        out = torch.zeros(H * W, dtype=torch.float32, device=device)
        _kernel[(1,)](x, out, H=H, W=W)
        assert torch.allclose(out, x.flatten(), atol=1e-3, rtol=1e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("outer,inner", [(4, 8), (8, 4)])
    @pytest.mark.parametrize("dtype_str", ["int32"])
    def test_arange_nested_dimensions(self, outer, inner, dtype_str, device):
        """Test arange with nested/hierarchical indexing patterns."""

        @triton.jit
        def _kernel(o_ptr, OUTER: tl.constexpr, INNER: tl.constexpr):
            # Create nested indices for outer and inner dimensions
            outer_indices = tlib.arange("o", tlib.dict(o=OUTER))
            inner_indices = tlib.arange("i", tlib.dict(i=INNER))
            combined_indices = tlib.arange("o i", tlib.dict(o=OUTER, i=INNER))

            # Use indices to access and store data
            tl.store(o_ptr + outer_indices[:, None] * INNER + inner_indices[None, :], combined_indices)

        # Reference: linear indices for flattened 2D array
        out_ref = torch.arange(outer * inner, dtype=getattr(torch, dtype_str), device=device).reshape(outer, inner)
        out = torch.zeros((outer, inner), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](out, OUTER=outer, INNER=inner)
        assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("length", [16, 32])
    @pytest.mark.parametrize("dtype_str", ["int32"])
    def test_arange_single_axis_description(self, length, dtype_str, device):
        """Test arange with single axis in description."""

        @triton.jit
        def _kernel(o_ptr, LENGTH: tl.constexpr):
            # Test single character axis name
            indices = tlib.arange("i", tlib.dict(i=LENGTH))
            tl.store(o_ptr + tl.arange(0, LENGTH), indices)

        out_ref = torch.arange(length, dtype=getattr(torch, dtype_str), device=device)
        out = torch.zeros(length, dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](out, LENGTH=length)
        assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("batch,seq", [(2, 16), (4, 8)])
    @pytest.mark.parametrize("dtype_str", ["int32"])
    def test_arange_sequence_indexing(self, batch, seq, dtype_str, device):
        """Test arange for sequence/batch indexing patterns."""

        @triton.jit
        def _kernel(o_ptr, BATCH: tl.constexpr, SEQ: tl.constexpr):
            # Create indices for batch and sequence dimensions
            batch_seq_indices = tlib.arange("b s", tlib.dict(b=BATCH, s=SEQ))

            # Store the computed indices
            tl.store(o_ptr + tl.arange(0, BATCH)[:, None] * SEQ + tl.arange(0, SEQ)[None, :], batch_seq_indices)

        # Reference: batch_idx * SEQ + seq_idx
        batch_indices, seq_indices = torch.meshgrid(
            torch.arange(batch, device=device), torch.arange(seq, device=device), indexing="ij"
        )
        out_ref = (batch_indices * seq + seq_indices).to(getattr(torch, dtype_str))
        out = torch.zeros((batch, seq), dtype=getattr(torch, dtype_str), device=device)
        _kernel[(1,)](out, BATCH=batch, SEQ=seq)
        assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)
