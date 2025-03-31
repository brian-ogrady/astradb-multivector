"""
Tests for the astra_multivector package initialization.

This module contains unit tests that verify the correct initialization
of the astra_multivector package, including:
- Exported classes and interfaces
- Proper handling of optional dependencies
- Version information
"""

import unittest
import importlib
import sys
from unittest.mock import patch


class TestInitModule(unittest.TestCase):
    """
    Test suite for verifying the initialization behavior of the astra_multivector package.
    
    These tests ensure that the package correctly exports all expected classes,
    properly handles optional dependencies, and provides version information.
    """
    
    def test_package_exports(self):
        """
        Verify that the package exports all required classes and attributes.
        
        This test confirms that:
        1. All expected classes are accessible as direct attributes
        2. The __all__ list contains all expected exports
        3. A valid version string is defined
        """
        import astra_multivector
        
        self.assertTrue(hasattr(astra_multivector, 'AstraMultiVectorTable'))
        self.assertTrue(hasattr(astra_multivector, 'AsyncAstraMultiVectorTable'))
        self.assertTrue(hasattr(astra_multivector, 'VectorColumnOptions'))
        
        expected_exports = [
            'AstraMultiVectorTable', 
            'AsyncAstraMultiVectorTable',
            'VectorColumnOptions'
        ]
        self.assertListEqual(sorted(astra_multivector.__all__), sorted(expected_exports))
        
        self.assertTrue(hasattr(astra_multivector, '__version__'))
        self.assertIsInstance(astra_multivector.__version__, str)
    
    def test_optional_dependencies_available(self):
        """
        Verify the HAS_LATE_INTERACTION flag when optional dependencies are available.
        
        This test mocks the presence of the torch package and confirms that
        the package correctly sets HAS_LATE_INTERACTION to True when the
        dependency is available.
        """
        import astra_multivector
        
        mock_torch = importlib.util.module_from_spec(importlib.util.find_spec('unittest'))
        
        with patch.dict(sys.modules, {'torch': mock_torch}):
            importlib.reload(astra_multivector)
            
            self.assertTrue(astra_multivector.HAS_LATE_INTERACTION)
    
    def test_optional_dependencies_not_available(self):
        """
        Verify the HAS_LATE_INTERACTION flag when optional dependencies are missing.
        
        This test simulates the absence of the torch package and confirms that
        the package correctly sets HAS_LATE_INTERACTION to False when the
        dependency is not available, ensuring the package can function without
        optional features.
        """
        original_import = __import__
        
        def custom_import(name, *args, **kwargs):
            if name == 'torch':
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)
        
        try:
            with patch.dict(sys.modules, {'torch': None}):
                with patch('builtins.__import__', custom_import):
                    import astra_multivector
                    importlib.reload(astra_multivector)
                    
                    self.assertFalse(astra_multivector.HAS_LATE_INTERACTION)
        finally:
            pass


if __name__ == '__main__':
    unittest.main()