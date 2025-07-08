import builtins

import triton
import triton.language as tl
from triton.language import core

# Creation
to_tensor = core.to_tensor

# Rearrange
reshape = tl.reshape
transpose = tl.trans
broadcast_to = tl.broadcast_to
arange = tl.arange

# Cat/stack
concatenate = tl.cat
stack = tl.join


# Elementwise
@triton.jit
def add(x, y):
    return x + y


@triton.jit
def subtract(x, y):
    return x - y


@triton.jit
def multiply(x, y):
    return x * y


@triton.jit
def true_divide(x, y):
    return x / y


@triton.jit
def floor_div(x, y):
    return x // y


@triton.jit
def divide(x, y):
    return x / y


@triton.jit
def logical_and(x, y):
    return x & y


@triton.jit
def logical_or(x, y):
    return x | y


@triton.jit
def less(x, y):
    return x < y


@triton.jit
def less_equal(x, y):
    return x <= y


@triton.jit
def greater(x, y):
    return x > y


@triton.jit
def greater_equal(x, y):
    return x >= y


@triton.jit
def equal(x, y):
    return x == y


@triton.jit
def not_equal(x, y):
    return x != y


@triton.jit
def maximum(x, y):
    return max(x, y)


@triton.jit
def minimum(x, y):
    return min(x, y)


# Reductions
@tl.constexpr_function
def _count_shape_dims(vals):
    return builtins.sum(vals) if isinstance(vals, list) else vals


@triton.jit
def sum(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    if tl.constexpr(mask is not None):
        return tl.sum(tl.where(mask, input, 0.0), axis=axis, keep_dims=keep_dims, dtype=dtype)
    else:
        return tl.sum(input, axis=axis, keep_dims=keep_dims, dtype=dtype)


@triton.jit
def mean(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    if tl.constexpr(mask is not None):
        total = sum(input, axis=axis, mask=mask, keep_dims=keep_dims, dtype=dtype)
        return total / tl.sum(mask, keep_dims=keep_dims, dtype=dtype)
    else:
        total = tl.sum(input, axis=axis, keep_dims=keep_dims, dtype=dtype)
        return total / _count_shape_dims(input.shape[axis])


@triton.jit
def var(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    mean_val = mean(input, axis=axis, mask=mask, keep_dims=True, dtype=dtype)
    if tl.constexpr(mask is not None):
        norm = tl.where(mask, input - mean_val, 0)
        total = tl.sum(norm * norm, axis=axis, keep_dims=keep_dims, dtype=dtype)
        return total / tl.sum(mask, keep_dims=keep_dims, dtype=dtype)
    else:
        norm = input - mean_val
        total = tl.sum(norm * norm, axis=axis, keep_dims=keep_dims, dtype=dtype)
        return total / _count_shape_dims(input.shape[axis])


@triton.jit
def std(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    return tl.sqrt(var(input, axis=axis, mask=None, keep_dims=keep_dims, dtype=dtype))  # A little crude but oh well


@triton.jit
def _prod_reduce(x, y):
    return x * y


@triton.jit
def prod(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    if tl.constexpr(mask is not None):
        return tl.reduce(tl.where(mask, input.to(dtype), 1), axis=axis, combine_fn=_prod_reduce, keep_dims=keep_dims)
    else:
        return tl.reduce(input.to(dtype), axis=axis, combine_fn=_prod_reduce, keep_dims=keep_dims)


@triton.jit
def count_nonzero(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    if tl.constexpr(mask is not None):
        return tl.sum(tl.where(mask, input, 0) != 0, axis=axis, keep_dims=keep_dims, dtype=dtype)
    else:
        return tl.sum(input != 0, axis=axis, keep_dims=keep_dims, dtype=dtype)


@triton.jit
def _any_reduce(x, y):
    return x | y


@triton.jit
def any(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    if tl.constexpr(mask is not None):
        return tl.reduce(tl.where(mask, input != 0, False), axis=axis, combine_fn=_any_reduce, keep_dims=keep_dims).to(
            dtype=dtype
        )
    else:
        return tl.reduce(input != 0, axis=axis, combine_fn=_any_reduce, keep_dims=keep_dims).to(dtype=dtype)


@triton.jit
def _all_reduce(x, y):
    return x & y


@triton.jit
def all(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    if tl.constexpr(mask is not None):
        return tl.reduce(tl.where(mask, input != 0, True), axis=axis, combine_fn=_all_reduce, keep_dims=keep_dims).to(
            dtype=dtype
        )
    else:
        return tl.reduce(input != 0, axis=axis, combine_fn=_all_reduce, keep_dims=keep_dims).to(dtype=dtype)


@triton.jit
def min(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    if tl.constexpr(mask is not None):
        return tl.min(tl.where(mask, input.to(dtype=dtype), float("inf")), axis=axis, keep_dims=keep_dims)
    else:
        return tl.min(input.to(dtype=dtype), axis=axis, keep_dims=keep_dims).to(dtype=dtype)


@triton.jit
def max(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    if tl.constexpr(mask is not None):
        return tl.max(tl.where(mask, input.to(dtype=dtype), float("-inf")), axis=axis, keep_dims=keep_dims)
    else:
        return tl.max(input.to(dtype=dtype), axis=axis, keep_dims=keep_dims).to(dtype=dtype)


@triton.jit
def argmin(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    if tl.constexpr(mask is not None):
        return tl.argmin(tl.where(mask, input.to(dtype=dtype), float("inf")), axis=axis, keep_dims=keep_dims)
    else:
        return tl.argmin(input.to(dtype=dtype), axis=axis, keep_dims=keep_dims).to(dtype=dtype)


@triton.jit
def argmax(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    if tl.constexpr(mask is not None):
        return tl.argmax(tl.where(mask, input.to(dtype=dtype), float("-inf")), axis=axis, keep_dims=keep_dims)
    else:
        return tl.argmax(input.to(dtype=dtype), axis=axis, keep_dims=keep_dims).to(dtype=dtype)


@triton.jit
def logsumexp(input, axis=None, mask=None, keep_dims=False, dtype: core.constexpr | None = None):
    """If a mask is used, then unknown behaviour/values in masked values (i.e., index marked as false)"""
    if tl.constexpr(mask is not None):
        input = tl.log(input.to(dtype=dtype))
        input = tl.sum(tl.where(mask, input, 0), axis=axis, keep_dims=keep_dims)
        return tl.exp(input)
    else:
        input = tl.log(input.to(dtype=dtype))
        input = tl.sum(input, axis=axis, keep_dims=keep_dims).to(dtype=dtype)
        return tl.exp(input)
