"""Test configuration and fixtures for triton-lib tests."""

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

import torch
import numpy as np
from typing import Generator, Any


if PYTEST_AVAILABLE:

    @pytest.fixture(scope="session")
    def device() -> str:
        """Get the device to run tests on."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture(scope="session")
    def torch_device(device: str) -> torch.device:
        """Get torch device object."""
        return torch.device(device)

    @pytest.fixture
    def random_seed() -> int:
        """Set a random seed for reproducible tests."""
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        return seed

    @pytest.fixture
    def small_tensor(torch_device: torch.device) -> torch.Tensor:
        """Create a small test tensor."""
        return torch.randn(4, 4, device=torch_device, dtype=torch.float32)

    @pytest.fixture
    def medium_tensor(torch_device: torch.device) -> torch.Tensor:
        """Create a medium test tensor."""
        return torch.randn(32, 32, device=torch_device, dtype=torch.float32)

    @pytest.fixture
    def large_tensor(torch_device: torch.device) -> torch.Tensor:
        """Create a large test tensor."""
        return torch.randn(128, 128, device=torch_device, dtype=torch.float32)


class TensorFactory:
    """Factory class for creating test tensors with various properties."""

    def __init__(self, device: torch.device):
        self.device = device

    def create_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        fill_value: float | None = None,
    ) -> torch.Tensor:
        """Create a tensor with specified properties."""
        if fill_value is not None:
            tensor = torch.full(shape, fill_value, device=self.device, dtype=dtype)
        else:
            if dtype.is_floating_point:
                tensor = torch.randn(shape, device=self.device, dtype=dtype)
            else:
                tensor = torch.randint(0, 100, shape, device=self.device, dtype=dtype)

        if requires_grad and dtype.is_floating_point:
            tensor.requires_grad_(True)

        return tensor


if PYTEST_AVAILABLE:

    @pytest.fixture
    def tensor_factory(torch_device: torch.device) -> TensorFactory:
        """Factory for creating test tensors."""
        return TensorFactory(torch_device)


def assert_tensors_close(
    actual: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8, msg: str = ""
) -> None:
    """Assert that two tensors are close with custom tolerances."""
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}. {msg}"
    assert actual.dtype == expected.dtype, f"Dtype mismatch: {actual.dtype} vs {expected.dtype}. {msg}"

    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        max_diff = torch.max(torch.abs(actual - expected))
        raise AssertionError(
            f"Tensors are not close. Max difference: {max_diff:.6e}, " f"rtol: {rtol}, atol: {atol}. {msg}"
        )


# Configure custom pytest markers only if pytest is available
if PYTEST_AVAILABLE:

    def pytest_configure(config):
        """Configure custom pytest markers."""
        config.addinivalue_line("markers", "unit: Unit tests")
        config.addinivalue_line("markers", "integration: Integration tests")
        config.addinivalue_line("markers", "functional: Functional tests")
        config.addinivalue_line("markers", "performance: Performance tests")
        config.addinivalue_line("markers", "slow: Slow tests")
        config.addinivalue_line("markers", "cuda: Tests requiring CUDA")
        config.addinivalue_line("markers", "triton: Tests requiring Triton")
