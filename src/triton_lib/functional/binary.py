import triton
import triton.language as tl
from triton.language import core

import triton_lib.functional as tlf


@triton.jit
def kl_div(input1, input2, mask: tl.tensor | None = None) -> tl.tensor: ...


@triton.jit
def cross_entropy(input1, input2, mask: tl.tensor | None = None) -> tl.tensor: ...


@triton.jit
def mse(input1, input2, axis: tl.constexpr | None = None, mask: tl.tensor | None = None) -> tl.tensor:
    return tlf.mean((input1 - input2) ** 2, axis, mask=mask)
