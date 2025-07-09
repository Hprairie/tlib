import triton
import triton.language as tl

import triton_lib.functional as tlf


@triton.jit
def cumsum(
    input,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    reverse: tl.constexpr | None = None,
) -> tl.tensor:
    if tl.constexpr(mask is not None):
        return tl.cumsum(tl.where(mask, input, 0), axis=axis, reverse=reverse)
    else:
        return tl.cumsum(input, axis=axis, reverse=reverse)


@triton.jit
def cumprod(
    input,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    reverse: tl.constexpr | None = None,
) -> tl.tensor:
    if tl.constexpr(mask is not None):
        return tl.cumprod(tl.where(mask, input, 1), axis=axis, reverse=reverse)
    else:
        return tl.cumprod(input, axis=axis, reverse=reverse)


@triton.jit
def flip(
    input,
    axis: tl.constexpr | None = None,
) -> tl.tensor:
    return tl.flip(input, dim=axis)  # Why does triton have dim and not axis for this, so dumb


@triton.jit
def softmax(
    input,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
) -> tl.tensor:
    if tl.constexpr(mask is not None):
        # Can this be a single tl.where? Seem's like potentially unsafe and could create nan
        _norm = input - tlf.max(input, axis, mask=mask)
        _exp = tl.exp(_norm)
        _denom = tlf.sum(_exp, axis, mask=mask)
        return _exp / _denom
    else:
        _norm = input - tlf.max(input, axis)
        _exp = tl.exp(_norm)
        _denom = tlf.sum(_exp, axis)
        return _exp / _denom
