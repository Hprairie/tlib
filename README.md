<h1 align="center" style="fontsize:50em"><b>Triton Lib</b></h1>

> WARNING: TLIB is currently under construction with some new features in Triton. Current implementations of `tlib` require compiling Triton from source.

Triton Lib (`tlib`) is an expansion on triton-lang's frontend. Tlib is written purely in python and is compatibable within functions decorated with `@triton.jit`, building off of [einx's](https://github.com/fferflo/einx) syntax and compiler with a few major changes to enable compatability with triton. This means that all functions are dynamically generated and then compiled with python's `exec()` during triton's compile-time, creating no bottlenecks during kernel runtime.

Tlib expands upon the triton frontend by providing APIs for common functions. For example triton doesn't come with built in `mean` and `var` functions. Tlib adds these functions and more! The overall goal of tlib is to make kernel code even more readible. Even in torch and other frameworks, tools like einops and einx have been used to drastically improve the readibility of functions. Writing GPU code should be the same!

# Installation

> WARNING: Currently requires Triton from source and torch 2.7.1, waiting for Triton 3.4.0 wheels and then will create PyPi package.

```bash
pip install -e .
```

When the latest triton and torch wheels come out in August I will create a pypi package for Tlib.

# Ops Examples

Just like every other einstein notation library, `tlib` comes with fundemental ops such as `rearrange` and `reduce`. 

### Rearrange

```python
import triton
import triton.language as tl
import triton_lib as tlib

@triton.jit
def my_kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, LENGTH)[:, None] * LENGTH + tl.arange(0, LENGTH)[None, :])
    # We can use rearrange to perform inbuilt triton shape manipulation ops
    out = tlib.rearrange("a b -> b a", x) # This is equivalent to tl.trans
    # Or equivalently in pure triton
    out = tl.trans(x, (1, 0))
```

### Reduce

Tlib has built in functionality for tensor reduction. Following syntax from einx, tlib allows bracket notation to indicate which axis to reduce along.

```python
import triton
import triton.language as tl
import triton_lib as tlib

@triton.jit
def my_kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, LENGTH)[:, None] * LENGTH + tl.arange(0, LENGTH)[None, :])
    # We can use tlib.reduce and specify an op
    out = tlib.reduce("a [b]", x, "sum") # This is equivalent to tl.sum
    # Or we can use built in functions
    out = tlib.sum("a [b]", x)
    # Or equivalently in pure triton
    out = tl.sum(x, axis=1)
```

### Unary VMAP

Tlib supports unary operations performed on a single tensor. This include most scans.

```python
import triton
import triton.language as tl
import triton_lib as tlib

@triton.jit
def my_kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, LENGTH)[:, None] * LENGTH + tl.arange(0, LENGTH)[None, :])
    # We can use tlib.binary and specify an op
    out = tlib.unary("a [b]", x, "cumsum") # This is equivalent to tl.cumsum on axis=1
    # Or we can use built in functions
    out = tlib.cumsum("a [b]", x)
    # Or equivalently in pure triton
    out = tl.cumsum(x, axis=1)
```

### Binary VMAP

Tlib supports unary operations performed between several tensors tensor. This makes broadcasting simpler. 

```python
import triton
import triton.language as tl
import triton_lib as tlib

@triton.jit
def my_kernel(x_ptr, y_ptr, o_ptr, LENGTH: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, LENGTH)[:, None] * LENGTH + tl.arange(0, LENGTH)[None, :])
    y = tl.load(y_ptr + tl.arange(0, LENGTH))
    # We can use tlib.binary and specify an op
    out = tlib.binary("a b, a", (x, y), "add")
    # Or we can use built in functions
    out = tlib.add("a b, a", (x, y))
    # Or equivalently in pure triton
    out = x + y[:, None]
```


### Dot VMAP

We can also use einstin notation to specify dot products. Coming soon!

# Why create/use Tlib

I will discuss, both `ops` and `functional` libraries added in tlib. Adding einstein notation `ops` to triton seemed like a no brainer. The readability of einstein notation in other high level frameworks such as torch, tensorfloew, jax, etc., makes it an incredibly appealing tool. Porting this functionality to triton, where we can evalue each expression at compile time convert it directly to `tl` syntax, makes it have features of high level abstractions without the performace reduction created by them.

Furthermore, on my quest to improve readability, I strongly desired to expand on the functionality of `tl` base language. I really desired to have the same functionality as torch but in triton. The best way to do this was to implement standard `triton.jit` functions for new functional values.

# Misc

This section will eventually be moved, but outlined are the current roadmap for functionality and the limitations of triton lib

### Roadmap

- [x] Implement `rearrange`, `reduce`, `unary`, and `binary` einstein notation ops
- [x] Add testing suite to `tlib`
- [x] Improve the number of reductions and built in operations (i.e., `var`, `mean`, etc.)
- [ ] Implement `dot` einstein notation ops
- [ ] Implement more useful binary and unary ops such as `softmax`, `kl_div`, `cross_entrop` into einstein notation ops
- [ ] Build a PyPi package
- [ ] Create Documentation
- [ ] Add testing to rearrange ops in `tlib`
- [x] Add testing to unary ops in `tlib`
- [x] Add testing to reduce ops in `tlib`
- [ ] Fix associative scan operation in `tlib`
### Limitations of Triton Lib