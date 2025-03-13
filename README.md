# AstraMultiVector

A Python library for creating and using multi-vector tables in DataStax Astra DB, supporting both client-side and server-side embedding generation with support for both synchronous and asynchronous operations.

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

## License

Apache License 2.0