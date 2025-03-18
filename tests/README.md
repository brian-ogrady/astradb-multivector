# AstraMultiVector Tests

This directory contains comprehensive unit tests for the `astra_multivector` library, covering all major components and functionality.

## Directory Structure

- `table/`: Tests for the table-related functionality (AstraMultiVectorTable, AsyncAstraMultiVectorTable, VectorColumnOptions)
- `late_interaction/`: Tests for late interaction models and pipeline components
  - `models/`: Tests for specific model implementations (ColBERT, ColPali)
  - `test_pipeline.py`: Tests for LateInteractionPipeline
  - `test_utils.py`: Tests for utility functions (pooling, parameter expansion)
- `run_tests.py`: Script to run all the tests

## Running Tests

You can run all tests using:

```bash
python tests/run_tests.py
```

Or run individual test files:

```bash
# Multi-vector table tests
python -m unittest tests/table/test_vector_column_options.py
python -m unittest tests/table/test_astra_multi_vector_table.py
python -m unittest tests/table/test_async_astra_multi_vector_table.py

# Late interaction tests
python -m unittest tests/late_interaction/models/test_base.py
python -m unittest tests/late_interaction/models/test_colbert.py
python -m unittest tests/late_interaction/models/test_colpali.py
python -m unittest tests/late_interaction/test_pipeline.py
python -m unittest tests/late_interaction/test_utils.py
```

## Test Coverage

The test suite provides comprehensive coverage of:
- Synchronous and asynchronous implementations
- Error cases and edge conditions
- Device handling and resource management
- Concurrency controls and parallel processing
- Token-level embedding and pooling strategies

To run tests with coverage reports:

```bash
# Install coverage package if needed
pip install coverage

# Run tests with coverage
python -m coverage run --source=astra_multivector tests/run_tests.py

# Generate coverage report
python -m coverage report -m

# For HTML report (optional)
python -m coverage html
# Then open htmlcov/index.html in a browser
```

## Test Requirements

The tests require:
- unittest (from Python standard library)
- mock (for mocking database operations)
- asyncio (for testing asynchronous code)
- PyTorch (for testing model implementations)

## Environment Variables

Some tests may use these environment variables if available:
- `ASTRA_TEST_TOKEN`: Token for optional integration testing with AstraDB
- `ASTRA_TEST_API_ENDPOINT`: API endpoint for optional integration testing
- `USE_CUDA`: Set to "true" to test CUDA device selection (if available)

## Contributing

When adding new features to the library, please follow these guidelines for tests:

1. Create unit tests that cover both normal operation and error conditions
2. For async code, use `IsolatedAsyncioTestCase` and proper async mocking
3. Mock external dependencies (AstraDB, torch models) appropriately
4. Ensure at least 90% code coverage for new components
5. Include tests for performance-critical code paths

Run `python -m coverage report -m` to verify coverage before submitting changes.