# Project Gutenberg Example

This example demonstrates how to use AstraMultiVector to download, process, and search books from Project Gutenberg with multiple embedding models for multilingual search.

## Overview

The Gutenberg example showcases:

1. **Multilingual Search**: Using language-specific embedding models alongside a general-purpose multilingual model
2. **Concurrent Processing**: Efficiently downloading and ingesting books in parallel
3. **Async API**: Full asynchronous operation with concurrency control
4. **Batch Operations**: Processing books in manageable chunks with progress tracking
5. **Multiple Vectors**: Storing different vector representations for the same content

## Components

The example consists of two main files:

1. **`gutenberg_example.py`**: Main example script demonstrating the complete workflow
2. **`ingest.py`**: Utility functions for downloading and ingesting books

## Setup

Before running the example:

1. Create a `.env` file in the project root with your AstraDB credentials:

```
ASTRA_DB_APPLICATION_TOKEN="your-astra-token"
ASTRA_DB_API_ENDPOINT="your-astra-api-endpoint"
OPENAI_API_KEY="your-openai-api-key"  # Required for OpenAI embeddings
```

2. Install the required dependencies:

```bash
pip install astra-multivector langchain tqdm sentence-transformers python-dotenv requests
```

## Running the Example

To run the complete example:

```bash
python gutenberg_example.py
```

The script will:

1. Download and ingest "Moby Dick" (English) and "Don Quixote" (Spanish)
2. Process each book with language-specific and multilingual embedding models
3. Perform various searches in both books, demonstrating different querying techniques

## Key Features

### Multiple Vector Models

The example uses three different embedding models:

```python
# English-specific model
english_vector_options = VectorColumnOptions.from_sentence_transformer(
    model=SentenceTransformer("BAAI/bge-small-en-v1.5")
)

# Spanish-specific model
spanish_vector_options = VectorColumnOptions.from_sentence_transformer(
    model=SentenceTransformer("jinaai/jina-embeddings-v2-base-es")
)

# Multilingual model (server-side with OpenAI)
multilingual_vector_options = VectorColumnOptions.from_vectorize(
    column_name="openai_embeddings",
    dimension=1536,
    vector_service_options=VectorServiceOptions(
        provider='openai',
        model_name='text-embedding-3-small'
    )
)
```

### Configurable Book Processing

Each book can be configured with specific processing options:

```python
books_config = [
    {
        "book_name": "moby_dick", 
        "book_id": 2701,
        "vector_column_options": [english_vector_options, multilingual_vector_options],
        "text_splitter": RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "! ", "? "],
            chunk_size=500,
        ),
    },
    # Additional books...
]
```

### Concurrent Ingestion

The example demonstrates concurrent processing of multiple books:

```python
await download_and_ingest_multiple_books(
    db=async_db,
    books_config=books_config,
    max_concurrency=10,
    batch_size=50,
    use_async=True
)
```

### Search Capabilities

The example shows several ways to search:

1. **Simple Search**: Query a specific vector column in a book
2. **Comparison Search**: Compare results between different embedding models
3. **Batch Search**: Execute multiple queries in parallel

```python
# Execute multiple searches in parallel
all_results = await table.batch_search_by_text(
    queries=["Describe Captain Ahab", "How does the story begin?"],
    vector_column=language_options.column_name,
    limit=max_results
)
```

## Extending the Example

You can extend this example in various ways:

1. **Add More Books**: Modify the `books_config` list to include additional books
2. **Use Different Models**: Swap out the embedding models for others
3. **Customize Chunking**: Adjust the text splitters for different languages or chunking strategies
4. **Add Filters**: Implement metadata and filtering in your searches
5. **Performance Tuning**: Adjust batch sizes and concurrency parameters

## Performance Considerations

- **Concurrency**: The `max_concurrency` parameter controls how many operations can run in parallel
- **Batch Size**: The `batch_size` parameter controls how many chunks are processed in a single batch
- **Chunking**: Smaller chunks lead to more precise retrieval but require more storage

## Troubleshooting

- **Rate Limiting**: If you encounter rate limiting with OpenAI, reduce the concurrency or add retries
- **Memory Issues**: If you run into memory problems, reduce batch sizes
- **Timeout Errors**: For large books, you may need to increase timeout settings in your AstraDB client