#!/usr/bin/env bash
# Setup a virtual environment for testing
rm -rf .venv/
uv venv
uv pip install -e ".[all,dev]"
uv pip install coverage  # Install coverage package
source .venv/bin/activate

# Run tests with coverage
echo "Running tests with coverage..."
coverage erase  # Clear any previous coverage data

# Run your tests with coverage
coverage run -m unittest tests/late_interaction/test_pipeline.py

# Add other test modules to the coverage
coverage run --append -m unittest tests/test_vector_column_options.py
coverage run --append -m unittest tests/test_astra_multi_vector_table.py
coverage run --append -m unittest tests/test_async_astra_multi_vector_table.py
coverage run --append -m unittest tests/late_interaction/test_utils.py
coverage run --append -m unittest tests/late_interaction/models/test_base.py
coverage run --append -m unittest tests/late_interaction/models/test_colpali.py
coverage run --append -m unittest tests/test_init.py

# Generate coverage reports
echo "Generating coverage report..."
coverage report -m
coverage html

# Displaying the location of the HTML report
echo "HTML coverage report generated in htmlcov/ directory"
echo "Open htmlcov/index.html in your browser to view it"

# Cleanup
deactivate
rm -rf .venv/