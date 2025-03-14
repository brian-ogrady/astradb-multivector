# Late Interaction Examples

This directory contains example applications demonstrating how to use the Late Interaction module with AstraDB for advanced retrieval use cases.

## Prerequisites

Before running these examples, make sure you have:

1. An AstraDB database with vector search capability
2. Required environment variables:
   - `ASTRA_TOKEN`: Your AstraDB API token
   - `ASTRA_API_ENDPOINT`: Your AstraDB API endpoint URL

## Examples

### 1. ColBERT Example (`colbert_example.py`)

This example demonstrates text-to-text late interaction retrieval using the ColBERT model:

- Sets up a ColBERT model and LateInteractionPipeline
- Indexes sample text documents with metadata
- Performs different types of searches, including:
  - Basic similarity search
  - Metadata-filtered search
  - Customized parameter search

To run this example:

```bash
# Install required dependencies
pip install colbert-ai dotenv astrapy

# Set up environment variables (or create a .env file)
export ASTRA_TOKEN="your-astra-token"
export ASTRA_API_ENDPOINT="your-api-endpoint"

# Run the example
python colbert_example.py
```

### 2. ColPali Example (`colpali_example.py`)

This example demonstrates multimodal late interaction (text-to-image) using the ColPali model:

- Sets up a ColPali model for multimodal processing
- Creates and indexes sample images with metadata tags
- Performs text queries to find matching images using token-level matching
- Shows different search configurations and filtering options

To run this example:

```bash
# Install required dependencies
pip install colpali-engine pillow dotenv astrapy

# Set up environment variables (or create a .env file)
export ASTRA_TOKEN="your-astra-token"
export ASTRA_API_ENDPOINT="your-api-endpoint"

# Run the example
python colpali_example.py
```

## Configuration Options

Both examples demonstrate several configuration options:

### Pipeline Configuration

- `doc_pool_factor`: Controls document token pooling (higher values = fewer tokens)
- `query_pool_distance`: Controls query token pooling threshold
- `default_concurrency_limit`: Limits concurrent operations

### Search Configuration

- `k`: Number of results to return
- `n_ann_tokens`: Number of tokens to retrieve per query token
- `n_maxsim_candidates`: Number of document candidates for MaxSim scoring
- `filter_condition`: Optional filter conditions for metadata

## Extending the Examples

These examples provide a starting point for your own applications. You can extend them by:

1. Using your own documents and images
2. Customizing the metadata structure
3. Trying different models (other ColBERT or ColPali variants)
4. Adjusting pooling parameters to optimize for your specific use case
5. Implementing more complex search patterns