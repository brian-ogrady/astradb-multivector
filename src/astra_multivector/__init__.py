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

try:
    import torch 
    
    from .late_interaction import (
        ColBERTModel,
        ColPaliModel,
        LateInteractionPipeline,
        expand_parameter,
        pool_doc_embeddings,
        pool_query_embeddings,
    )
    
    __all__.extend([
        'ColBERTModel',
        'ColPaliModel',
        'LateInteractionPipeline',
        'expand_parameter',
        'pool_doc_embeddings',
        'pool_query_embeddings',
    ])
    
    HAS_LATE_INTERACTION = True
    
except ImportError:
    HAS_LATE_INTERACTION = False
