import triton
import triton.language as tl


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def silu_exp2(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def sigmoid(x):
    return tl.sigmoid(x)


@triton.jit
def gelu(x):
    M_SQRT1_2 = 0.70710678118654752440
    ALPHA = M_SQRT1_2
    return 0.5 * x * (1.0 + tl.erf(x * ALPHA))


@triton.jit
def gelu_tanh(x):
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return 0.5 * x * (1.0 + tanh(inner))
