"""
AstraDB Multi-Vector package for efficient vector operations.

This package provides utilities for working with multi-vector tables in AstraDB.
"""

from .astra_multi_vector_table import AstraMultiVectorTable
from .async_astra_multi_vector_table import AsyncAstraMultiVectorTable
from .vector_column_options import VectorColumnOptions

__all__ = [
    'AstraMultiVectorTable',
    'AsyncAstraMultiVectorTable',
    'VectorColumnOptions',
]

__version__ = "0.1.0"

# Check if optional dependencies are available, but don't import them
try:
    import torch
    HAS_LATE_INTERACTION = True
except ImportError:
    HAS_LATE_INTERACTION = False