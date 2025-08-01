import triton
import triton.language as tl
from triton.language import core

import tlib.functional as tlf


@triton.jit
def _reduce(vals, reduction: tl.constexpr, mask: tl.tensor | None = None):
    if tl.constexpr(reduction == "mean"):
        return tlf.mean(vals.ravel(), axis=0, mask=mask)
    elif tl.constexpr(reduction == "sum"):
        return tlf.sum(vals.ravel(), axis=0, mask=mask)
    elif tl.constexpr(reduction == "none"):
        return vals
    else:
        tl.static_assert(False, f"Unknown reduction type: {reduction.value}")


@triton.jit
def kl_div(
    input,
    target,
    mask: tl.tensor | None = None,
    reduction: tl.constexpr = tl.constexpr("mean"),
    log_target: tl.constexpr = False,
) -> tl.tensor:
    if log_target:
        _loss_pointwise = target.exp() * (target - input)
    else:
        _loss_pointwise = target * (target.log() - input)
    return _reduce(_loss_pointwise, reduction=reduction, mask=mask)


@triton.jit
def cross_entropy(input1, input2, mask: tl.tensor | None = None) -> tl.tensor: ...


@triton.jit
def mse(
    input,
    target,
    mask: tl.tensor | None = None,
    reduction: tl.constexpr = tl.constexpr("mean"),
) -> tl.tensor:
    _temp_loss_pointwise = input - target
    _loss_pointwise = _temp_loss_pointwise * _temp_loss_pointwise
    return _reduce(_loss_pointwise, reduction=reduction, mask=mask)
