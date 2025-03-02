# Vector Search with Rerankers

This repository demonstrates how to implement a two-stage retrieval system using vector search followed by reranking.

## Why Rerankers Matter for Enterprise Search

Rerankers provide significant improvements to vector search by:
- Analyzing both the query and document together at search time
- Using cross-attention to match specific parts of queries to specific parts of documents
- Improving relevance for complex, multi-faceted queries
- Prioritizing results based on actual relevance rather than just semantic similarity

This two-stage approach combines the efficiency of vector search with the precision of rerankers.

## Repository Structure

- `vector_column_options.py`: Configurations for vector columns in the database
- `gutenberg_text_vector_table.py`: Table creation and vector search functionality
- `ingestion.py`: Book downloading and processing with concurrent ingestion

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vector-search-rerankers.git
cd vector-search-rerankers

# Create and activate virtual environment (using uv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your AstraDB credentials:
```
ASTRA_DB_API_ENDPOINT=your-api-endpoint
ASTRA_DB_APPLICATION_TOKEN=your-token
```

## Usage

### Setting Up Vector Columns

```python
from sentence_transformers import SentenceTransformer
from vector_column_options import VectorColumnOptions

# Initialize embedding model
model = SentenceTransformer("intfloat/e5-large-v2")

# Create vector column options for client-side embedding
vector_options = VectorColumnOptions.from_sentence_transformer(model)

# Alternatively, for server-side vectorization:
from astrapy.info import VectorServiceOptions
vector_service_options=ectorColumnOptions.from_vectorize(
        column_name="openai_embeddings",
        dimension=1536,
        vector_service_options=vector_options,
        table_vector_index_options=index_options,
)

```

### Creating Tables and Ingesting Data

```python
from astrapy import database
from gutenberg_text_vector_table import GutenbergTextVectorTable
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize database connection
db = database.Database(
    token="your-token",
    api_endpoint="your-endpoint"
)

# Create table with vector search capabilities
table = GutenbergTextVectorTable(
    database=db,
    table_name="book_vectors",
    vector_options,
)

# Download and ingest a book
from ingest import download_and_ingest_book

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". "],
    chunk_size=500,
    chunk_overlap=50
)

download_and_ingest_book(
    database=db,
    book_name="moby_dick",
    book_id=2701,
    text_splitter=text_splitter,
    vector_options,
)
```

### Concurrent Processing of Multiple Books

```python
import asyncio
from ingestion import download_and_ingest_multiple_books

# Configure books for processing
books_config = [
    {
        "book_name": "moby_dick", 
        "book_id": 2701,
        "vector_column_options": [vector_options],
        "text_splitter": text_splitter
    },
    {
        "book_name": "don_quixote", 
        "book_id": 996,
        "vector_column_options": [vector_options],
        "text_splitter": text_splitter
    }
]

# Run concurrent ingestion
asyncio.run(download_and_ingest_multiple_books(db, books_config))
```

## Example Queries

When implementing the search functionality, follow this pattern for best results:
1. Perform vector search to retrieve top-k candidates
2. Apply a reranker to these candidates for more precise ranking
3. Return the reranked results to the user

This approach ensures both efficiency and relevance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.