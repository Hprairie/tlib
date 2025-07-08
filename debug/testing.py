import torch
import triton
import triton.language as tl
import triton_lib as tlib


@triton.jit
def my_kernel(
    x_ptr,
    o_ptr,
    LENGHT: tl.constexpr,
):
    x = tl.load(
        x_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :]
    )
    # x = tlib.rearrange("a b -> b a", x)
    # tl.store(
    #     o_ptr + tl.arange(0, LENGHT)[:, None] * LENGHT + tl.arange(0, LENGHT)[None, :],
    #     x,
    # )
    # x = tlib.reduce("[a] b", x, "sum", mask=tl.arange(0, LENGHT) < (LENGHT // 2))
    x = tlib.reduce("[a] b", x, "sum")
    # x = tlib.var("[a] b", x)
    tl.store(o_ptr + tl.arange(0, LENGHT), x)


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
