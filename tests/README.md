# AstraMultiVector Tests

This directory contains unit tests for the `astra_multivector` library.

## Directory Structure

- `table/`: Tests for the table-related functionality (AstraMultiVectorTable, AsyncAstraMultiVectorTable, VectorColumnOptions)
- `run_tests.py`: Script to run all the tests

## Running Tests

You can run all tests using:

```bash
python tests/run_tests.py
```

Or run individual test files:

```bash
python -m unittest tests/table/test_vector_column_options.py
python -m unittest tests/table/test_astra_multi_vector_table.py
python -m unittest tests/table/test_async_astra_multi_vector_table.py
```

## Test Requirements

The tests require:
- unittest (from Python standard library)
- mock (for mocking database operations)