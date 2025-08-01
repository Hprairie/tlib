import torch
import triton
import triton.language as tl
import triton_lib as tlib
import triton_lib.functional as tlf


@triton.jit
def add(x, y):
    return x + y


@triton.jit
def my_kernel(
    x_ptr,
    o_ptr,
    LENGHT: tl.constexpr,
):
    x = tl.load(x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :])
    # x = tl.load(x_ptr + tl.arange(0, LENGHT))
    # x = tlib.add("a, b -> a b", (x, x))
    # x = tlib.sort("a [b]", x, descending=True)
    y = tlib.rearrange("a b -> b a", x)
    o = tl.dot(x, y)
    # x = tlib.dot("a b, c b -> a c", (x, x))
    # x = tlib.associative_scan("a [b]", x, add)
    # x = tlib.rearrange("a, c -> (a + c)", (x, x))
    # tl.store(o_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :], x)
    tl.store(o_ptr + tl.arange(0, LENGHT), x)


def launch(x):
    o = torch.zeros_like(x)
    print(x)
    my_kernel[(1, 1, 1)](x, o, x.shape[0])
    return o


x = torch.arange(8).to("cuda", dtype=torch.float32)
x = x[:, None] * 8 + x[None, :]
o = launch(x)
# print(torch.softmax(x, dim=-1))
print(o)


# @tl.constexpr_function
# def make_constexpr_function(string):
