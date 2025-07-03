import triton
import triton.language as tl
import triton.language.core as tlc

@tl.constexpr_function
def get_shape(x: tl.tensor | None) -> tlc.tuple:
    if x is not None:
        return x.shape
    else:
        return None