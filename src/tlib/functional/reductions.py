import builtins
import triton
import triton.language as tl
from triton.language import core

from typing import Iterable


# Reductions
@tl.constexpr_function
def _count_shape_dims(vals):
    return builtins.sum(vals) if isinstance(vals, Iterable) else vals


@triton.jit
def sum(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
    dtype: core.constexpr | None = None,
):
    if tl.constexpr(mask is not None):
        return tl.sum(tl.where(mask, input, 0.0), axis=axis, keep_dims=keep_dims, dtype=dtype)
    else:
        return tl.sum(input, axis=axis, keep_dims=keep_dims, dtype=dtype)


@triton.jit
def mean(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
    dtype: core.constexpr | None = None,
):
    if tl.constexpr(mask is not None):
        total = sum(input, axis=axis, mask=mask, keep_dims=keep_dims, dtype=dtype)
        return total / tl.sum(mask, keep_dims=keep_dims, dtype=dtype)
    else:
        total = tl.sum(input, axis=axis, keep_dims=keep_dims, dtype=dtype)
        if tl.constexpr(isinstance(total, tl.tuple_type)):
            total = total[0]
        if tl.constexpr(axis is None):
            return total / _count_shape_dims(input.shape)
        else:
            return total / _count_shape_dims(input.shape[axis])


@triton.jit
def var(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
    dtype: core.constexpr | None = None,
    return_mean: tl.constexpr = False,
):
    mean_val = mean(input, axis=axis, mask=mask, keep_dims=True, dtype=dtype)
    if tl.constexpr(mask is not None):
        norm = tl.where(mask, input - mean_val, 0)
        total = tl.sum(norm * norm, axis=axis, keep_dims=keep_dims, dtype=dtype)
        if return_mean:
            return total / tl.sum(mask, keep_dims=keep_dims, dtype=dtype), norm
        else:
            return total / tl.sum(mask, keep_dims=keep_dims, dtype=dtype)
    else:
        norm = input - mean_val
        total = tl.sum(norm * norm, axis=axis, keep_dims=keep_dims, dtype=dtype)
        if tl.constexpr(axis is None):
            out = total / _count_shape_dims(input.shape)
        else:
            out = total / _count_shape_dims(input.shape[axis])
        if return_mean:
            return out, norm
        else:
            return out


@triton.jit
def std(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
    dtype: core.constexpr | None = None,
    return_mean: tl.constexpr = False,
):
    if return_mean:
        _var, _mean = var(
            input, axis=axis, mask=mask, keep_dims=keep_dims, dtype=dtype, return_mean=return_mean
        )  # A little crude but oh well
        return tl.sqrt(_var), _mean
    else:
        return tl.sqrt(var(input, axis=axis, mask=mask, keep_dims=keep_dims, dtype=dtype))  # A little crude but oh well


@triton.jit
def _prod_reduce(x, y):
    return x * y


@triton.jit
def prod(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
):
    if tl.constexpr(mask is not None):
        return tl.reduce(tl.where(mask, input, 1), axis=axis, combine_fn=_prod_reduce, keep_dims=keep_dims)
    else:
        return tl.reduce(input, axis=axis, combine_fn=_prod_reduce, keep_dims=keep_dims)


@triton.jit
def count_nonzero(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
    dtype: core.constexpr | None = None,
):
    if tl.constexpr(mask is not None):
        return tl.sum(tl.where(mask, input, 0) != 0, axis=axis, keep_dims=keep_dims, dtype=dtype)
    else:
        return tl.sum(input != 0, axis=axis, keep_dims=keep_dims, dtype=dtype)


@triton.jit
def _any_reduce(x, y):
    return x | y


@triton.jit
def any(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
):
    if tl.constexpr(mask is not None):
        return tl.reduce(tl.where(mask, input != 0, False), axis=axis, combine_fn=_any_reduce, keep_dims=keep_dims)
    else:
        return tl.reduce(input != 0, axis=axis, combine_fn=_any_reduce, keep_dims=keep_dims)


@triton.jit
def _all_reduce(x, y):
    return x & y


@triton.jit
def all(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
):
    if tl.constexpr(mask is not None):
        return tl.reduce(tl.where(mask, input != 0, True), axis=axis, combine_fn=_all_reduce, keep_dims=keep_dims)
    else:
        return tl.reduce(input != 0, axis=axis, combine_fn=_all_reduce, keep_dims=keep_dims)


@triton.jit
def min(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
):
    if tl.constexpr(mask is not None):
        return tl.min(tl.where(mask, input, float("inf")), axis=axis, keep_dims=keep_dims)
    else:
        return tl.min(input, axis=axis, keep_dims=keep_dims)


@triton.jit
def max(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
):
    if tl.constexpr(mask is not None):
        return tl.max(tl.where(mask, input, float("-inf")), axis=axis, keep_dims=keep_dims)
    else:
        return tl.max(input, axis=axis, keep_dims=keep_dims)


@triton.jit
def argmin(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
):
    if tl.constexpr(mask is not None):
        if tl.constexpr(axis is None):
            return tl.argmin(tl.where(mask, input, float("inf")).ravel(), axis=-1, keep_dims=keep_dims)
        else:
            return tl.argmin(tl.where(mask, input, float("inf")), axis=axis, keep_dims=keep_dims)
    else:
        if tl.constexpr(axis is None):
            return tl.argmin(input.ravel(), axis=-1, keep_dims=keep_dims)
        else:
            return tl.argmin(input, axis=axis, keep_dims=keep_dims)


@triton.jit
def argmax(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
):
    if tl.constexpr(mask is not None):
        if tl.constexpr(axis is None):
            return tl.argmax(tl.where(mask, input, float("-inf")).ravel(), axis=0, keep_dims=keep_dims)
        else:
            return tl.argmax(tl.where(mask, input, float("-inf")), axis=axis, keep_dims=keep_dims)
    else:
        if tl.constexpr(axis is None):
            return tl.argmax(input.ravel(), axis=0, keep_dims=keep_dims)
        else:
            return tl.argmax(input, axis=axis, keep_dims=keep_dims)


@triton.jit
def logsumexp(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    keep_dims: tl.constexpr = False,
):
    """If a mask is used, then unknown behaviour/values in masked values (i.e., index marked as false)"""
    if tl.constexpr(mask is not None):
        _max = max(input, axis=axis, mask=mask, keep_dims=keep_dims)
        input = tl.exp(input - _max)
        input = tl.sum(tl.where(mask, input, 0), axis=axis, keep_dims=keep_dims)
        return _max + tl.log(input)
    else:
        _max = max(input, axis=axis, keep_dims=keep_dims)
        input = tl.exp(input - _max)
        input = tl.sum(input, axis=axis, keep_dims=keep_dims)
        return _max + tl.log(input)
