"""Base test class and utilities for triton-lib tests."""

import torch
import numpy as np
import triton
import inspect
import textwrap
import triton.language as tl
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from triton._internal_testing import is_interpreter

import tlib

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


def patch_kernel(template, to_replace):
    if is_interpreter():
        local_namespace = {}
        src = textwrap.dedent(inspect.getsource(template.fn))
        for k, v in to_replace.items():
            src = src.replace(k, v)
        exec(src, globals(), local_namespace)
        return local_namespace[template.fn.__name__]
    else:
        kernel = triton.JITFunction(template.fn)
        for key, value in to_replace.items():
            kernel._unsafe_update_src(kernel.src.replace(key, value))
        return kernel


class BaseTritonTest:
    """Base class for triton-lib tests with common utilities."""

    def setup_method(self):
        """Setup method called before each test."""
        torch.manual_seed(42)
        np.random.seed(42)

    def assert_output_correct(
        self,
        triton_func,
        triton_load,
        triton_store,
        torch_func,
        IN_SHAPE,
        OUT_SHAPE,
        device: torch.device,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        @triton.jit
        def kernel(x_ptr, o_ptr, IN_SHAPE: tl.constexpr, OUT_SHAPE: tl.constexpr):
            x = FUNCTION_LOAD
            x = FUNCTION_TO_REPLACE
            FUNCTION_STORE

        kernel = patch_kernel(
            kernel, {"FUNCTION_TO_REPLACE": triton_func, "FUNCTION_LOAD": triton_load, "FUNCTION_STORE": triton_store}
        )
        x = torch.randn(IN_SHAPE, dtype=torch.float32, device=device)
        triton_output = torch.zeros(OUT_SHAPE, dtype=torch.float32, device=device)
        torch_output = torch_func(x.clone())
        kernel[(1, 1, 1)](x, triton_output, IN_SHAPE=IN_SHAPE, OUT_SHAPE=OUT_SHAPE)
        assert triton_output.shape == torch_output.shape
        assert triton_output.dtype == torch_output.dtype
        assert torch.allclose(triton_output, torch_output, rtol=rtol, atol=atol)


class PerformanceTest:
    """Base class for performance testing."""

    def benchmark_function(
        self,
        func: Callable,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        num_warmup: int = 10,
        num_runs: int = 100,
        **kwargs,
    ) -> Dict[str, float]:
        """Benchmark a function and return timing statistics."""
        import time

        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)

        # Warmup
        for _ in range(num_warmup):
            _ = func(*inputs, **kwargs)

        # Sync GPU if using CUDA
        if inputs[0].is_cuda:
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = func(*inputs, **kwargs)
            if inputs[0].is_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "median": float(np.median(times)),
        }


# Decorator functions only available if pytest is installed
if PYTEST_AVAILABLE:

    def parametrize_dtypes(*dtypes):
        """Decorator to parametrize tests over multiple dtypes."""
        return pytest.mark.parametrize("dtype", dtypes)

    def parametrize_shapes(*shapes):
        """Decorator to parametrize tests over multiple shapes."""
        return pytest.mark.parametrize("shape", shapes)

    def parametrize_devices(*devices):
        """Decorator to parametrize tests over multiple devices."""
        return pytest.mark.parametrize("device", devices)

else:
    # Dummy decorators when pytest is not available
    def parametrize_dtypes(*dtypes):
        def decorator(func):
            return func

        return decorator

    def parametrize_shapes(*shapes):
        def decorator(func):
            return func

        return decorator

    def parametrize_devices(*devices):
        def decorator(func):
            return func

        return decorator
