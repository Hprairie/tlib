<h1 align="center" style="fontsize:50em"><b>Triton Lib</b></h1>

Triton Lib (`tlib`) is an python library providing universal einstien notation functionality to triton kernels, along with a functional expansion on triton-lang's frontend. Tlib is written purely in python and is compatibable within functions decorated with `@triton.jit`. The design of this library is two-fold:

- **Provide an expanded set of base functional operations (`tlf`) for triton, which are numericaly stable and allow masking**: `tlf.{mean|var|std|mse|kl_div|...}`

- **Provide an [einx](https://github.com/fferflo/einx) style ops syntax to all base and expanded triton ops, which are dynamically generated at compile time to incur no overhead within kernels.**

Tlib is built off of the [einx](https://github.com/fferflo/einx) syntax and compiler, with a few major changes to enable compatability with triton. This means that all functions are dynamically generated and then compiled with python's `exec()` during triton's compile-time, creating no bottlenecks during kernel runtime.

**Getting Started**

- Installation (COMING SOON)
- Tutorial (COMING SOON)
- [Einx Notation](https://einx.readthedocs.io)
- API Reference (COMING SOON)


# Installation

Tlib is built using new features from triton `3.4.0`, which is only compatible with `torch >= 2.8.0`. Tlib can be installed using the following `pip` command

```bash
pip install triton-lib
```
or built from source using the following command
```bash
git clone https://github.com/Hprairie/tlib.git
cd tlib
pip install -e .
```

# What doese `tlib` look like in kernels?

Tlib provideds ops for almost all base `triton` frontend ops and more added on by `tlf` (tlib.funtional).

```python
import triton
import triton.language as tl
import tlib
import tlib.functional as tlf

@triton.jit
def my_kernel(x_ptr, y_ptr, o_ptr, LENGTH: tl.constexpr):
    # arange indexing ops
    x = tl.load(x_ptr + tlib.arange("a b", tlib.dict(a=LENGTH, b=LENGTH)))
    y = tl.load(x_ptr + tlib.arange("a b c", tlib.dict(a=LENGTH, b=LENGTH, c=LENGTH)))

    # Rearrange ops
    o = tlib.rearrange("a b -> b a", x) # This is equivalent to tl.trans(x, (1, 0))
    o = tlib.rearrange("a b -> (a b)", x) # This is equivalent to tl.reshape(x, (LENGTH * LENGTH,))
    o = tlib.rearrange("a b c -> c (a b)", y) # This is equivalent to tl.reshape(x, (LENGTH, LENGTH * LENGTH)) followed by tl.trans(x, (0, 1))

    # Unary Ops
    o = tlib.cumsum("a [b]", x) # This is equivalent to tl.cumsum(x, axis=1)
    o = tlib.cumprod("a [b] c", y) # This is equivalent to tl.cumprod(x, axis=1)
    o = tlib.flip("[a] b", x) # This is equivalent to tl.flip(x, axis=0)
    o = tlib.sort("a b [c]", y) # This is equivalent to tl.sort(x, axis=2)
    o = tlib.softmax("a [b]", x) # This is equivalent to tl.softmax(x, axis=1)

    # Binary Ops
    o = tlib.add("a b, a b c", (x, y)) # This is equivalent to x[:, :, None] + y
    o = tlib.add("a c, a b c", (x, y)) # This is equivalent to x[:, None, :] + y
    o = tlib.add("b c, a b c", (x, y)) # This is equivalent to x[None, :, :] + y
    o = tlib.subtract("a b, a b c", (x, y)) # This is equivalent to x[:, :, None] - y
    o = tlib.multiply("a b, a b c", (x, y)) # This is equivalent to x[:, :, None] * y
    o = tlib.divide("a b, a b c", (x, y)) # This is equivalent to x[:, :, None] / y

    # Reduction Ops
    out = tlib.sum("a [b]", x) # This is equivalent to tl.sum(x, axis=1)
    out = tlib.mean("a [b]", x) # This is equivalent to tlf.mean(x, axis=1)
    out = tlib.var("a [b]", x) # This is equivalent to tlf.var(x, axis=1)
    out = tlib.count_nonzero("a [b]", x) # This is equivalent to tlf.count_nonzero(x, axis=1)
    out = tlib.max("a [b]", x) # This is equivalent to tl.max(x, axis=1)
    out = tlib.min("a [b]", x) # This is equivalent to tl.min(x, axis=1)
    out = tlib.argmax("a [b]", x) # This is equivalent to tl.argmax(x, axis=1)
    out = tlib.argmin("a [b]", x) # This is equivalent to tl.argmin(x, axis=1)
```

# Why create/use Tlib

I will discuss, both `ops` and `functional` libraries added in tlib. Adding einstein notation `ops` to triton seemed like a no brainer. The readability of einstein notation in other high level frameworks such as torch, tensorfloew, jax, etc., makes it an incredibly appealing tool. Porting this functionality to triton, where we can evalue each expression at compile time convert it directly to `tl` syntax, makes it have features of high level abstractions without the performace reduction created by them.

Furthermore, on my quest to improve readability, I strongly desired to expand on the functionality of `tl` base language. I really desired to have the same functionality as torch but in triton. The best way to do this was to implement standard `triton.jit` functions for new functional values.

# Limitations

As you might have noticed from the examples, there are some API differences between `tlib` and `einx`/`einops`. First, when passing multiple tensors, say to `tlib.rearrange`, we need to wrap them in a tuple object. 

```python
o = rearrange("a b c, d e f -> a c b, d f e", x, y)
o = tlib.rearrange("a b c, d e f -> a c b, d f e", (x, y))
```

Additionally, dictionaries aren't supported in triton, thus I have created a wrapper: `tlib.dict`, which functions like a dictionary, but is a `tl.constexpr`.


# Misc

This section will eventually be moved, but outlined are the current roadmap for functionality and the limitations of triton lib

### ToDo

- [ ] Implement `dot` einstein notation ops
- [ ] Build a PyPi package
- [ ] Create Documentation
- [ ] Fix associative scan operation in `tlib`

# References

This package is partly built on [einx](https://github.com/fferflo/einx), whose copyright has been added into the project and added upon.