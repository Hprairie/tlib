import torch
import triton
import triton.language as tl
from typing import Union, List

import triton_lib as tlib


@triton.jit
def add(x):
    return x + x

@triton.jit
def sub(x):
    return x - x

@tl.constexpr_function
def function(query: str):
    if query == "+":
        return add
    else:
        return sub

@triton.jit
def rearrange(
    query: tl.constexpr, x: tl.tensor, y: tl.tensor | None = None, z: tl.tensor | None = None) -> Union[tl.tensor, List[tl.tensor]]:
    if y is not None:
        return function(query)(x) * y
    return function(query)(x)

@triton.jit
def my_kernel(
    x_ptr,
    o_ptr,
    LENGHT: tl.constexpr,
):
    x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
    # x = rearrange("-", x)
    x = tlib.rearrange("a b -> b a", x)
    tl.store(o_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :], x)

def launch(x):
    o = torch.zeros_like(x)
    my_kernel[(1, 1, 1)](x, o, x.shape[0])
    return o

x = torch.arange(8).to("cuda")
x = x[:, None] * x[None, :]
o = launch(x)
print(o)



# @tl.constexpr_function
# def make_constexpr_function(string):
