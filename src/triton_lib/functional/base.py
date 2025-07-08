import triton
import triton.language as tl
from triton.language import core

# Creation
to_tensor = tl.tensor

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
@triton.jit
def sum(
    input,
    axis=None,
    mask=None,
    keep_dims=False,
    dtype: core.constexpr | None = None,
):
    if tl.constexpr(mask is not None):
        return tl.sum(
            tl.where(mask, input, 0.0), axis=axis, keep_dims=keep_dims, dtype=dtype
        )
    else:
        return tl.sum(input, axis=axis, keep_dims=keep_dims, dtype=dtype)


@triton.jit
def mean(
    input,
    axis=None,
    mask=None,
    keep_dims=False,
    dtype: core.constexpr | None = None,
):
    total = tl.sum(input, axis=axis, keep_dims=keep_dims, dtype=dtype)
    return total / input.shape[axis]


@triton.jit
def var(
    input,
    axis=None,
    mask=None,
    keep_dims=False,
    dtype: core.constexpr | None = None,
):
    mean_val = mean(input, axis=axis, keep_dims=True, dtype=dtype)
    norm = input - mean_val
    total = tl.sum(norm * norm, axis=axis, keep_dims=keep_dims, dtype=dtype)
    return total / input.shape[axis]


@triton.jit
def std(input, axis=None, keep_dims=False, dtype: core.constexpr | None = None):
    total = tl.sum(input, axis=axis, keep_dims=keep_dims, dtype=dtype)
    return input.to(total)
