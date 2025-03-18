# AstraMultiVector Examples

This directory contains example applications demonstrating various ways to use the AstraMultiVector library for creating and searching multi-vector tables in AstraDB.

## Available Examples

### 1. Async Multi-Vector with Reranking (`async_multivector_example.py`)

A comprehensive demonstration of the `AsyncAstraMultiVectorTable` with advanced features:

- Setting up multiple embedding models (client-side and server-side)
- Concurrent document processing with controlled parallelism
- Multi-vector search across different embedding columns
- Reranking search results for improved relevance
- Custom parallel processing workflows

```bash
# Set up environment variables first
export ASTRA_DB_APPLICATION_TOKEN="your-token"
export ASTRA_DB_API_ENDPOINT="your-endpoint"
export OPENAI_API_KEY="your-key"  # For vectorize columns

# Run the example
python async_multivector_example.py
```

### 2. Gutenberg Books Search (in `gutenberg/` folder)

Example of ingesting and searching books from Project Gutenberg with multilingual support:

- Downloads books from Project Gutenberg
- Processes books with language-specific text splitters
- Creates multiple vector columns for English, Spanish, and multilingual search
- Demonstrates parallel batch processing for efficient ingestion
- Compares search results between different embedding models

```bash
# Run the example
python gutenberg/gutenberg_example.py
```

### 3. Late Interaction Models (in `late_interaction/` folder)

Examples demonstrating token-level late interaction models with AstraDB:

- ColBERT Example: Text-to-text late interaction retrieval
- ColPali Example: Multimodal text-to-image retrieval

See the [late_interaction README](late_interaction/README.md) for more details.

## Setup Requirements

Before running these examples, make sure you have:

1. An AstraDB instance with an API token and endpoint
2. Python 3.8 or higher
3. Required dependencies installed:
   ```bash
   pip install astra-multivector[all]
   ```
4. Environment variables set up (or use a `.env` file):
   ```bash
   ASTRA_DB_APPLICATION_TOKEN="your-token"
   ASTRA_DB_API_ENDPOINT="your-endpoint"
   OPENAI_API_KEY="your-key"  # For OpenAI vectorize examples
   ```

## Key Concepts Demonstrated

These examples showcase several key capabilities of the AstraMultiVector library:

- **Multiple Vector Representations**: Using different embedding models for the same content
- **Client-side and Server-side Embeddings**: Using both local embedding models and AstraDB's Vectorize feature
- **Asynchronous Operations**: Efficient parallel processing with asyncio
- **Reranking**: Improving search relevance with a separate reranking step
- **Batch Processing**: Processing documents and queries in efficient batches
- **Multilingual Support**: Working with content in different languages