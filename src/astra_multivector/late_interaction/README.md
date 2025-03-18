# Late Interaction

A module for implementing and using Late Interaction retrieval models with AstraDB.

## Overview

Late Interaction is an approach to information retrieval that defers matching between query and document tokens until retrieval time, providing higher accuracy than traditional dense retrieval methods at the cost of increased computation.

This library provides:

- Ready-to-use implementations of popular late interaction models (ColBERT, ColPali)
- Integration with AstraDB for scalable token-level indexing
- Efficient indexing and search pipelines with concurrency control
- Optimizations like token pooling to balance performance and accuracy

## Key Components

### LateInteractionPipeline

The main entry point for using late interaction models with AstraDB:

- Creates and manages document and token-level tables
- Handles document indexing and storage with flexible document schema
- Implements efficient two-stage retrieval with auto-scaling parameters:
  1. ANN search to find candidate tokens
  2. MaxSim scoring for final ranking
- Optimizes performance through token pooling and concurrency controls

### Supported Models

- **ColBERT**: Text-to-text late interaction model
  - Efficient token-level matching for high-precision search
  - Compatible with various HuggingFace ColBERT checkpoints

- **ColPali**: Multimodal late interaction model
  - Supports image-to-text and text-to-image search
  - Uses token-level cross-attention for fine-grained matching

## Usage Example

```python
import asyncio
import uuid
from astrapy import AsyncDatabase
from late_interaction import LateInteractionPipeline, ColBERTModel

async def main():
    # Initialize database connection
    db = async_db = DataAPIClient(
        token="your-token",
    ).get_async_database(
        api_endpoint="your-api-endpoint",
    )
    
    # Create a ColBERT model
    model = ColBERTModel(
        model_name="answerdotai/answerai-colbert-small-v1",
        device="cuda"  # or "cpu"
    )
    
    # Create pipeline
    pipeline = LateInteractionPipeline(
        db=db,
        model=model,
        base_table_name="my_colbert_index",
    )
    
    # Initialize tables
    await pipeline.initialize()
    
    # Index documents
    doc_row = {
        "content": "This is a sample document for testing late interaction retrieval.",
        "doc_id": uuid.uuid4()
    }
    doc_id = await pipeline.index_document(doc_row)
    
    # Search for similar documents
    results = await pipeline.search(
        query="sample retrieval",
        k=5,  # Number of results to return
        n_ann_tokens=200,  # Optional: number of tokens to retrieve per query token
        n_maxsim_candidates=20  # Optional: number of candidates for final scoring
    )
    
    # Print results
    for doc_id, score, content in results:
        print(f"Document: {doc_id}, Score: {score:.4f}")
        print(f"Content: {content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Performance Optimizations

The library includes several optimizations to balance retrieval quality with computational efficiency:

- **Query Pooling**: Clusters similar query tokens to reduce token count
- **Document Pooling**: Hierarchically pools document tokens to reduce index size
- **Adaptive Parameter Scaling**: Automatically scales search parameters based on result count
- **Concurrency Control**: Manages parallel operations for optimal throughput

## Requirements

- Python 3.12+
- PyTorch
- AstraDB Python SDK
- ColBERT (for text retrieval)
- ColPali (for multimodal retrieval)
