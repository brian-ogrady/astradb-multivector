# AstraMultiVector

A Python library for creating and using multi-vector tables in DataStax Astra DB, supporting both client-side and server-side embedding generation with support for both synchronous and asynchronous operations.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Async Usage](#async-usage)
- [Multiple Vector Columns](#multiple-vector-columns)
- [Schema Design](#schema-design)
  - [Multi-Vector Architecture](#multi-vector-architecture)
  - [Late Interaction Architecture](#late-interaction-architecture)
- [Gutenberg Example](#gutenberg-example)
- [Late Interaction](#late-interaction)
- [API Reference](#api-reference)
  - [VectorColumnOptions](#vectorcolumnoptions)
  - [AstraMultiVectorTable](#astramultivectortable)
  - [AsyncAstraMultiVectorTable](#asyncastramultivectortable)
- [Contributing](#contributing)
- [License](#license)

## Overview

AstraMultiVector provides classes to:
- Create database tables with multiple vector columns
- Associate each vector column with either:
  - Client-side embeddings using sentence-transformers
  - Server-side embeddings using Astra's Vectorize feature
- Search across any vector column using similarity search
- Support both synchronous and asynchronous operations

This allows for storing and retrieving text data with multiple embedding representations, which is useful for:
- Multilingual document search
- Comparing different embedding models
- Specialized embeddings for different query types

## Installation

```bash
# Install from PyPI
pip install astra-multivector

# Or install from source
git clone https://github.com/datastax/astra-multivector.git
cd astra-multivector
pip install -e .
```

### Requirements

- Python 3.8 or higher
- Dependencies:
  - astrapy >= 2.0.0
  - pydantic>=2.10.6
  - python-dotenv>=1.0.1
  - sentence-transformers>=3.4.1
  - rerankers[api,transformers]>=0.8.0
  - tqdm>=4.67.1

Optional dependencies for late interaction models:
  - colbert-ai >= 0.2.0
  - colpali-engine>=0.3.1,<0.4.0
  - torch >= 2.0.0
  - transformers>=4.38.2
  - scikit-learn>=1.3.0
  - numpy>=1.24.0

## Quick Start

```python
from astrapy.database import Database
from astra_multivector import AstraMultiVectorTable, VectorColumnOptions
from sentence_transformers import SentenceTransformer

# Create database connection
db = Database(
    token="your-token",
    api_endpoint="your-api-endpoint"
)

# Create embedding models and vector options
english_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
english_options = VectorColumnOptions.from_sentence_transformer(english_model)

# Create the table
table = AstraMultiVectorTable(
    db=db,
    table_name="my_vectors",
    vector_column_options=[english_options]
)

# Insert data
table.insert_chunk("This is a sample text to embed and store.")

# Search
results = table.search_by_text("sample text", limit=5)
for result in results:
    print(result["content"])
```

## Async Usage

```python
import asyncio
from astrapy.database import AsyncDatabase
from astrapy import DataAPIClient
from astra_multivector import AsyncAstraMultiVectorTable, VectorColumnOptions

async def main():
    # Create async database connection
    async_db = DataAPIClient(
        token="your-token",
    ).get_async_database(
        api_endpoint="your-api-endpoint",
    )
    
    # Create the table with the same vector options
    async_table = AsyncAstraMultiVectorTable(
        db=async_db,
        table_name="my_vectors",
        vector_column_options=[english_options],
        default_concurrency_limit=10
    )
    
    # Batch insert with concurrency control
    await async_table.bulk_insert_chunks(
        text_chunks=["Text 1", "Text 2", "Text 3"],
        max_concurrency=5
    )
    
    # Batch search
    queries = ["first query", "second query", "third query"]
    all_results = await async_table.batch_search_by_text(queries)

# Run the async code
asyncio.run(main())
```

## Multiple Vector Columns

You can create tables with multiple vector columns, each using a different model or vectorization approach:

```python
from astrapy.constants import VectorMetric
from astrapy.info import TableVectorIndexOptions, VectorServiceOptions

# Client-side embedding with a Spanish model
spanish_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-es")
spanish_options = VectorColumnOptions.from_sentence_transformer(
    model=spanish_model,
    table_vector_index_options=TableVectorIndexOptions(
        metric=VectorMetric.COSINE,
    )
)

# Server-side embedding with OpenAI
openai_options = VectorColumnOptions.from_vectorize(
    column_name="openai_embeddings",
    dimension=1536,
    vector_service_options=VectorServiceOptions(
        provider='openai',
        model_name='text-embedding-3-small',
        authentication={
            "providerKey": "OPENAI_API_KEY",
        },
    ),
    table_vector_index_options=TableVectorIndexOptions(
        metric=VectorMetric.COSINE,
    )
)

# Create multi-vector table
table = AstraMultiVectorTable(
    db=db,
    table_name="multilingual_vectors",
    vector_column_options=[spanish_options, openai_options]
)
```

## Schema Design

### Multi-Vector Architecture

The multi-vector architecture stores multiple vector representations of the same content in separate columns of a single table:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ Table: my_vectors                                                              │
├────────────┬─────────────────────┬──────────────────┬──────────────────────────┤
│ chunk_id   │ content             │ english_embeddings│ multi-lingual embeddings│
├────────────┼─────────────────────┼──────────────────┼──────────────────────────┤
│ UUID-1     │ "Hello world"       │ [0.1, 0.2, ...]  │ [0.3, 0.4, ...]         │
│ UUID-2     │ "Vector search"     │ [0.5, 0.6, ...]  │ [0.7, 0.8, ...]         │
└────────────┴─────────────────────┴──────────────────┴──────────────────────────┘
     │                │                    │                   │
     │                │                    │                   │
     │                │                    ▼                   ▼
     │                │             ┌─────────────┐    ┌───────────────┐
     │                │             │ Vector Index│    │ Vector Index  │
     │                │             │ (english)   │    │(multi-lingual)│
     │                │             └─────────────┘    └───────────────┘
     │                │
     │                ▼
     │         Used directly for
     │         Vectorize columns
     │
     ▼
Partition Key
```

This design allows for:
- Multiple embedding representations of the same content
- Choice of embedding model at query time
- Combination of results from different embeddings

### Late Interaction Architecture

The late interaction architecture splits documents into token-level embeddings across multiple tables:

```
┌────────────────────────────────────────┐     ┌────────────────────────────────────────┐
│ Table: my_colbert_docs                 │     │ Table: my_colbert_tokens               │
├──────────┬───────────────────────────┐ │     ├──────────┬──────────┬─────────────────┐│
│ doc_id   │ content                   │ │     │ doc_id   │ token_id │ token_embedding ││
├──────────┼───────────────────────────┤ │     ├──────────┼──────────┼─────────────────┤│
│ UUID-1   │ "Example document content"│ │     │ UUID-1   │ tok-1.1  │ [0.1, 0.2, ...] ││
│ UUID-2   │ "Another document example"│ │     │ UUID-1   │ tok-1.2  │ [0.3, 0.4, ...] ││
└──────────┴───────────────────────────┘ │     │ UUID-1   │ tok-1.3  │ [0.5, 0.6, ...] ││
                                         │     │ UUID-2   │ tok-2.1  │ [0.7, 0.8, ...] ││
                                         │     │ UUID-2   │ tok-2.2  │ [0.9, 1.0, ...] ││
                                         │     └──────────┴──────────┴─────────────────┘│
                                                         │            │
                                                         │            ▼
                                                         │     ┌─────────────┐
                                                         │     │ Vector Index│
                                                         │     └─────────────┘
                                                         │
                                                         ▼
                                                 Referenced during
                                                 token->document lookup
```

This architecture allows for:
- Token-level similarity matching between queries and documents
- Higher precision retrieval with late interaction models like ColBERT
- Multimodal matching between text and images with models like ColPali

## Gutenberg Example

The repository includes a complete example for ingesting and searching books from Project Gutenberg using multiple vector models. This example demonstrates:

1. Setting up multiple embedding models:
   - Language-specific models (English, Spanish)
   - OpenAI embeddings via Vectorize

2. Processing books in parallel with async operations:
   - Concurrent book downloads
   - Batch processing with configurable concurrency

3. Performing searches across different vector columns:
   - Language-specific searches
   - Parallel batch searching

To run the example:

```python
# See examples/gutenberg_example.py
import asyncio
import os
from dotenv import load_dotenv
from astra_multivector import VectorColumnOptions, AsyncAstraMultiVectorTable
from astra_multivector.ingest import download_and_ingest_multiple_books

# Load environment variables
load_dotenv()

# Run the example
asyncio.run(main())
```

## Late Interaction

The library includes an optional sub-module for late interaction retrieval, which defers matching between query and document tokens until retrieval time, providing higher accuracy than traditional dense retrieval methods.

```python
import uuid
from astrapy.database import AsyncDatabase
from astra_multivector.late_interaction import LateInteractionPipeline, ColBERTModel

# Initialize database and model
db = AsyncDatabase(token="your-token", api_endpoint="your-api-endpoint")
model = ColBERTModel(model_name="answerdotai/answerai-colbert-small-v1")

# Create pipeline with optimization options
pipeline = LateInteractionPipeline(
    db=db,
    model=model,
    base_table_name="my_colbert_index",
    doc_pool_factor=2,  # Compress document tokens by this factor
    query_pool_distance=0.03,  # Pool similar query tokens
    default_concurrency_limit=10,  # Control parallel operations
)

# Initialize tables
await pipeline.initialize()

# Index documents with dictionary format
doc_row = {
    "content": "This is a sample document for testing late interaction retrieval.",
    "doc_id": uuid.uuid4()  # Optional: auto-generated if not provided
}
doc_id = await pipeline.index_document(doc_row)

# Batch indexing with concurrency control
docs = [
    {"content": "Document one for batch indexing"},
    {"content": "Document two for batch indexing"},
    {"content": "Document three for batch indexing"}
]
doc_ids = await pipeline.bulk_index_documents(
    document_rows=docs,
    concurrency=5,
    batch_size=2
)

# Search with auto-scaled parameters
results = await pipeline.search(
    query="sample retrieval", 
    k=5,  # Number of results to return
    # Optional parameters, auto-calculated if not provided:
    n_ann_tokens=200,         # Tokens to retrieve per query token
    n_maxsim_candidates=20    # Document candidates for scoring
)

# Process search results
for doc_id, score, content in results:
    print(f"Document: {doc_id}, Score: {score:.4f}")
    print(f"Content: {content}")
```

### Multimodal Search with ColPali

For multimodal search supporting images and text:

```python
from PIL import Image
from astra_multivector.late_interaction import LateInteractionPipeline, ColPaliModel

# Initialize model
model = ColPaliModel(model_name="vidore/colpali-v0.1")

# Create pipeline
pipeline = LateInteractionPipeline(
    db=db,
    model=model,
    base_table_name="my_colpali_index"
)

# Index an image
image = Image.open("example.jpg")
doc_id = await pipeline.index_document({
    "content": image,  # Directly pass PIL Image
    "doc_id": uuid.uuid4()
})

# Search for images using text query
results = await pipeline.search(
    query="a cat sitting on a chair", 
    k=5
)

# Search with image query requires preprocessing the image first
query_image = Image.open("query.jpg")
query_embeddings = await model.encode_query(query_image)
results = await pipeline.search_with_embeddings(
    query_embeddings,
    k=5
)
```

Supported models include:
- **ColBERT**: Text-to-text token-level matching for high-precision search
- **ColPali**: Multimodal token-level matching supporting image-to-text and text-to-image search

For more details, see the [Late Interaction README](src/astra_multivector/late_interaction/README.md).

## API Reference

### VectorColumnOptions

Configures vector columns with embedding options:

- `from_sentence_transformer()`: For client-side embeddings with sentence-transformers
- `from_vectorize()`: For server-side embeddings with Astra's Vectorize

### AstraMultiVectorTable

Synchronous table operations:

- `insert_chunk()`: Insert a single text chunk with embeddings
- `bulk_insert_chunks()`: Insert multiple chunks in batches
- `search_by_text()`: Search for similar text in a vector column
- `batch_search_by_text()`: Perform multiple searches

### AsyncAstraMultiVectorTable

Asynchronous table operations:

- `insert_chunk()`: Insert a single text chunk asynchronously
- `bulk_insert_chunks()`: Insert multiple chunks with concurrency control
- `search_by_text()`: Perform async search for similar text
- `batch_search_by_text()`: Execute multiple searches in parallel
- `parallel_process_chunks()`: Process items in parallel with custom function

## Contributing

Contributions to AstraMultiVector are welcome! Here's how you can contribute:

### Development Setup

1. Fork the repository and clone your fork
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Testing

All contributions should include tests:

```bash
# Run all tests
python tests/run_tests.py

# Check test coverage
python -m coverage run --source=astra_multivector tests/run_tests.py
python -m coverage report -m
```

Aim for at least 90% test coverage for new code.

### Submitting Changes

1. Create a new branch for your feature
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Run the test suite to ensure everything passes
5. Submit a pull request with a clear description of the changes

### Code Style

This project follows:
- PEP 8 for code style
- Google style docstrings
- Type annotations for all functions

## License

Apache License 2.0
