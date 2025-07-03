import triton
import triton.language as tl

# Elementwise
@triton.jit
def add(x, y):
    return x + y

@triton.jit
def sub(x, y):
    return x - y

@triton.jit
def mul(x, y):
    return x * y

@triton.jit
def div(x, y):
    return x / y

@triton.jit
def floor_div(x, y):
    return x // y

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