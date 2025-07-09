import triton
import triton.language as tl
from triton.language import core


@triton.jit
def cumsum(
    input,
    axis=None,
    mask: tl.tensor | None = None,
    reverse: tl.constexpr | None = None,
    dtype: core.constexpr | None = None,
) -> tl.tensor:
    raise NotImplementedError


@triton.jit
def cumprod(
    input,
    axis=None,
    mask: tl.tensor | None = None,
    reverse: tl.constexpr | None = None,
    dtype: core.constexpr | None = None,
) -> tl.tensor:
    raise NotImplementedError
