import triton
import triton.language as tl
from triton.language import core


@triton.jit
def log_softmax(input, axis=None, mask: tl.tensor | None = None, dtype: core.constexpr | None = None) -> tl.tensor: ...


@triton.jit
def kl_div(
    input1, input2, axis=None, mask: tl.tensor | None = None, dtype: core.constexpr | None = None
) -> tl.tensor: ...


@triton.jit
def cross_entropy(
    input1, input2, axis=None, mask: tl.tensor | None = None, dtype: core.constexpr | None = None
) -> tl.tensor: ...


@triton.jit
def mse(input1, input2, axis=None, mask: tl.tensor | None = None, dtype: core.constexpr | None = None) -> tl.tensor: ...
