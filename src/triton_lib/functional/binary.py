import triton
import triton.language as tl
from triton.language import core


@triton.jit
def kl_div(input1, input2, mask: tl.tensor | None = None) -> tl.tensor: ...


@triton.jit
def cross_entropy(input1, input2, mask: tl.tensor | None = None) -> tl.tensor: ...


@triton.jit
def mse(input1, input2, mask: tl.tensor | None = None) -> tl.tensor: ...
