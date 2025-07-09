import torch
import triton
import triton.language as tl
import triton_lib as tlib
import triton_lib.functional as tlf


@triton.jit
def take_3(x, y, z):
    return x, y, z


@triton.jit
def take_many(tensors):
    if tl.constexpr(isinstance(tensors, tl.tensor)):
        tl.static_print("hello")
        return tensors
    else:
        x, y, z = take_3(*tensors)  # Triton allows unpacking!!!!
        return x, y, z


@triton.jit
def my_kernel(
    x_ptr,
    o_ptr,
    LENGHT: tl.constexpr,
):
    x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
    x = tlib.unary("a [b]", x, "cumsum")
    # x = tlib.sum("a [b]", x)
    # x = tlib.rearrange("a, c -> (a + c)", (x, x))
    tl.store(o_ptr + tl.arange(0, LENGHT * 2), x)


def launch(x):
    o = torch.zeros_like(x)
    print(x)
    my_kernel[(1, 1, 1)](x, o, x.shape[0])
    return o


x = torch.arange(8).to("cuda")
x = x[:, None] * 8 + x[None, :]
o = launch(x)
print(o)


# @tl.constexpr_function
# def make_constexpr_function(string):
