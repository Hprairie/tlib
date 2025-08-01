import triton
import triton.language as tl
import tlib.functional as tlf


@triton.jit
def layer_norm(
    input: tl.tensor,
    normalized_shape: tl.constexpr = -1,
    mask: tl.tensor | None = None,
    weight: tl.tensor | None = None,
    bias: tl.tensor | None = None,
    eps: tl.constexpr = 1e-05,
    return_rstd_mean: tl.constexpr = False,
) -> tl.tensor | tuple[tl.tensor, tl.tensor, tl.tensor]:
    var, mean = tlf.var(input, axis=normalized_shape, mask=mask, return_mean=True)
    rstd = tl.rsqrt(var + eps)
    if tl.constexpr(weight is not None and bias is not None):
        out = mean * rstd * weight + bias
    elif tl.constexpr(weight is not None):
        out = mean * rstd * weight
    elif tl.constexpr(bias is not None):
        raise ValueError(f"Bias is passed however weight is not...")
    else:
        out = mean * rstd
    # Prepare output
    if return_rstd_mean:
        return out, rstd, mean
    else:
        return out
