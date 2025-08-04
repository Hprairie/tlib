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
expand_dims = tl.expand_dims


@triton.jit
def arange(
    n: tl.constexpr,
    stride: tl.constexpr = tl.constexpr(1),
) -> tl.tensor:
    return tl.arange(0, n) * stride


# Cat/stack
@triton.jit
def concatenate(tensors, axis: tl.constexpr | None = None) -> tl.tensor:
    raise NotImplementedError


@triton.jit
def stack(tensors, axis: tl.constexpr | None = None) -> tl.tensor:
    raise NotImplementedError


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
    return tl.floor(x / y)


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
