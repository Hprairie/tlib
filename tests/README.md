# Test Infrastructure for triton-lib

This directory contains the test infrastructure for the triton-lib project, focusing on testing the `functional` and `ops` modules.

## Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                   # Pytest configuration and fixtures  
├── test_base.py                  # Base test classes and utilities
├── functional/                   # Tests for triton_lib.functional
│   ├── __init__.py
│   ├── test_activations.py      # Activation function tests
│   ├── test_unary.py            # Unary operation tests
│   ├── test_binary.py           # Binary operation tests
│   └── test_reductions.py       # Reduction operation tests
└── ops/                         # Tests for triton_lib.ops
    ├── __init__.py
    ├── test_unary.py            # Unary ops tests
    ├── test_binary.py           # Binary ops tests  
    ├── test_dot.py              # Dot product tests
    ├── test_reduce.py           # Reduction tests
    ├── test_rearrange.py        # Rearrange operation tests
    ├── test_solve.py            # Solve operation tests
    └── test_vmap_with_axis.py   # Vmap operation tests
```

## Installation

Install the test dependencies:

```bash
# Install development dependencies (includes test dependencies)
pip install -e ".[dev]"

# Or install just test dependencies
pip install -e ".[test]"
```

## Running Tests

### Using pytest directly

```bash
# Run all tests
pytest

# Run tests for a specific module
pytest tests/functional/
pytest tests/ops/

# Run tests with specific markers
pytest -m unit          # Unit tests only
pytest -m functional    # Functional tests only
pytest -m cuda          # CUDA-requiring tests only
pytest -m performance   # Performance tests only

# Run specific test files
pytest tests/functional/test_activations.py
pytest tests/ops/test_dot.py

# Run with coverage
pytest --cov=triton_lib --cov-report=html
```

### Using the Makefile

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-functional
make test-ops
make test-cuda
make test-performance

# Run with coverage
make coverage

# Run tests in parallel
make test-parallel

# Run with verbose output
make test-verbose
```

## Test Categories

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.functional` - Functional correctness tests
- `@pytest.mark.performance` - Performance benchmarking tests
- `@pytest.mark.cuda` - Tests requiring CUDA
- `@pytest.mark.triton` - Tests requiring Triton
- `@pytest.mark.slow` - Slow-running tests

## Fixtures

Common fixtures are available in `conftest.py`:

- `device` - Device string ("cuda" or "cpu")
- `torch_device` - PyTorch device object
- `random_seed` - Sets reproducible random seed
- `small_tensor`, `medium_tensor`, `large_tensor` - Pre-created test tensors
- `tensor_factory` - Factory for creating custom test tensors

## Writing Tests

### Basic Test Structure

```python
import pytest
import torch
from tests.test_base import BaseTritonTest

class TestMyFunction(BaseTritonTest):
    @pytest.mark.unit
    def test_basic_functionality(self, tensor_factory):
        # Create test data
        x = tensor_factory.create_tensor((32, 32), torch.float32)
        
        # Test your function
        result = my_triton_function(x)
        expected = torch_reference_function(x)
        
        # Assert correctness
        self.assert_output_correct(my_triton_function, torch_reference_function, x)
```

### Testing Against PyTorch Reference

Use the base test class methods:

```python
# Test output correctness
self.assert_output_correct(triton_func, torch_func, inputs)

# Test gradient correctness  
self.assert_gradients_correct(triton_func, torch_func, inputs)
```

### Performance Testing

```python
from tests.test_base import PerformanceTest

class TestPerformance(PerformanceTest):
    @pytest.mark.performance
    def test_function_speed(self, large_tensor):
        stats = self.benchmark_function(my_function, large_tensor)
        assert stats["mean"] < 0.01  # Less than 10ms
```

### Parametrized Tests

```python
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(32, 32), (64, 128)])
def test_multiple_configs(self, tensor_factory, dtype, shape):
    x = tensor_factory.create_tensor(shape, dtype)
    # Test logic here
```

## Configuration

Test configuration is in:
- `pytest.ini` - Main pytest configuration
- `pyproject.toml` - Test dependencies and tool configuration
- `.github/workflows/test.yml` - CI/CD configuration

## CI/CD

Tests run automatically on:
- Push to main/develop branches
- Pull requests to main
- CUDA tests run on self-hosted runners with GPU access

## TODO

Each test file contains placeholder tests with TODO comments. To implement a test:

1. Remove the TODO comment
2. Import the relevant triton_lib functions
3. Implement the test logic
4. Add appropriate markers (@pytest.mark.unit, etc.)

Example:
```python
# Before
def test_relu_basic(self):
    # TODO: Implement ReLU test
    pass

# After  
def test_relu_basic(self, tensor_factory):
    x = tensor_factory.create_tensor((32, 32), requires_grad=True)
    self.assert_output_correct(tlf.relu, torch.relu, x)
    self.assert_gradients_correct(tlf.relu, torch.relu, x)
```
