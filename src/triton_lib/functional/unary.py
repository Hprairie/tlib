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
    input: tl.tensor,
    axis: tl.constexpr | None = None,
) -> tl.tensor:
    return tl.flip(input, dim=axis)  # Why does triton have dim and not axis for this, so dumb


@triton.jit
def softmax(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    mask: tl.tensor | None = None,
    eps: tl.constexpr = 1e-10,
) -> tl.tensor:
    raise NotImplemented
    if tl.constexpr(mask is not None):
        # Can this be a single tl.where? Seem's like potentially unsafe and could create nan
        _norm = input - tlf.max(input, axis, mask=mask)
        _exp = tl.exp(_norm)
        _denom = tlf.sum(_exp, axis, mask=mask)
        return _exp / (_denom)
    else:
        _norm = input - tlf.max(input, axis=axis, keep_dims=True)
        _exp = tl.exp(_norm)
        _denom = tlf.sum(_exp, axis=axis, keep_dims=True)
        return _exp / (_denom + eps)


@triton.jit
def log_softmax(input, axis=None, mask: tl.tensor | None = None) -> tl.tensor: ...


@triton.jit
def sort(
    input: tl.tensor,
    axis: tl.constexpr | None = None,
    descending: tl.constexpr = False,
) -> tl.tensor:
    return tl.sort(input, dim=axis, descending=descending)  # Why does triton have dim and not axis for this, so dumb


@triton.jit
def associative_scan(
    input: tl.tensor,
    combine_fn,
    axis: tl.constexpr | None = None,
    reverse: tl.constexpr = False,
) -> tl.tensor:
    tl.static_print(combine_fn)
    return tl.associative_scan(input, axis=axis, combine_fn=combine_fn, reverse=reverse)
