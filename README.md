<h1 align="center" style="fontsize:50em"><b>Triton Lib</b></h1>

> WARNING: TLIB is currently under construction with some new features in Triton. Current implementations of `tlib` require compiling Triton from source.

Triton Lib (`tlib`) is an expansion on triton-lang's frontend. Tlib is written purely in python and is compatibable within functions decorated with `@triton.jit`, building off of [einx's](https://github.com/fferflo/einx) syntax and compiler with a few major changes to enable compatability with triton. This means that all functions are dynamically generated and then compiled with python's `exec()` during triton's compile-time, creating no bottlenecks during kernel runtime.

# Installation

> WARNING: Currently requires Triton from source and torch 2.7.1

```bash
pip install -e .
```

# Examples

Just like every other einstein notation library, `tlib` comes with fundemental ops such as `rearrange` and `reduce` (WIP). 

### Rearrange

```python
import triton
import triton.language as tl
import triton_lib as tlib

@triton.jit
def my_kernel(x_ptr, o_ptr, LENGTH: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, LENGTH)[:, None] * LENGTH + tl.arange(0, LENGTH)[None, :])
    x = tlib.rearrange("a b -> b a", x) # This is equivalent to transpose in triton
```

# Misc

This section will eventually be moved, but outlined are the current roadmap for functionality and the limitations of triton lib

### Roadmap

- [ ] Implement `rearrange`, `reduce`, `element-by-element`, and `dot` einstein notation ops
- [ ] Add testing suite to `tlib`
- [ ] Improve the number of reductions and built in operations (i.e., `var`, `mean`, etc.)

### Limitations of Triton Lib