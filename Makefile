.PHONY: test test-unit test-functional test-ops test-cuda test-performance install-dev clean coverage

# Install development dependencies
install-dev:
	pip install -e ".[dev]"

# Run all tests
test:
	pytest

# Run only unit tests
test-unit:
	pytest -m unit

# Run functional module tests
test-functional:
	pytest tests/functional/

# Run ops module tests  
test-ops:
	pytest tests/ops/

# Run CUDA-specific tests
test-cuda:
	pytest -m cuda

# Run performance tests
test-performance:
	pytest -m performance

# Run tests with coverage (requires pytest-cov)
coverage:
	@if python -c "import pytest_cov" 2>/dev/null; then \
		pytest --cov=triton_lib --cov-report=html --cov-report=term; \
	else \
		echo "pytest-cov not installed. Install with: pip install pytest-cov"; \
		pytest; \
	fi

# Run tests in parallel
test-parallel:
	pytest -n auto

# Clean up test artifacts
clean:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Run specific test file
test-file:
	@echo "Usage: make test-file FILE=tests/functional/test_activations.py"
	@if [ -n "$(FILE)" ]; then pytest $(FILE); fi

# Run tests with verbose output
test-verbose:
	pytest -v

# Run tests and stop on first failure
test-fail-fast:
	pytest -x
