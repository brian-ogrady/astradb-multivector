#!/usr/bin/env python3
"""
Test runner script for the astra-multivector package.

Discovers and runs all test files in the 'tests' directory.
"""
import unittest
import sys
import os

# Configure Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    """
    Main entry point for test execution.
    
    Discovers all test files matching 'test_*.py' pattern in the tests directory,
    runs them with verbosity level 2, and sets the exit code based on test results.
    """
    test_suite = unittest.defaultTestLoader.discover('tests', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    sys.exit(not result.wasSuccessful())