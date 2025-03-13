import asyncio
import os
from typing import Any, Dict, List

from astrapy.database import AsyncDatabase
from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.info import TableVectorIndexOptions, VectorServiceOptions
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from astra_multivector import VectorColumnOptions, AsyncAstraMultiVectorTable
from astra_multivector.ingest import download_and_ingest_multiple_books


# Define vector column options for different languages/models
english_vector_options = VectorColumnOptions.from_sentence_transformer(
    model=SentenceTransformer("BAAI/bge-small-en-v1.5"),
    table_vector_index_options=TableVectorIndexOptions(
        metric=VectorMetric.COSINE,
    )
)

spanish_vector_options = VectorColumnOptions.from_sentence_transformer(
    model=SentenceTransformer("jinaai/jina-embeddings-v2-base-es"),
    table_vector_index_options=TableVectorIndexOptions(
        metric=VectorMetric.COSINE,
    )
)

multilingual_vector_options = VectorColumnOptions.from_vectorize(
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
        source_model="openai-v3-small",
    )
)

# Configure books for processing
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
    {
        "book_name": "don_quixote", 
        "book_id": 2000,
        "vector_column_options": [spanish_vector_options, multilingual_vector_options],
        "text_splitter": RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "¡", "¿", "! ", "? "],
            chunk_size=500,
        ),
    }
]


async def ingest_books_async(books_config: List[Dict[str, Any]], max_concurrency: int = 10) -> None:
    """Ingest multiple books using async operations."""
    load_dotenv()
    
    # Create async database connection
    async_db: AsyncDatabase = DataAPIClient(
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    ).get_async_database(
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    )
    
    # Process all books concurrently
    await download_and_ingest_multiple_books(
        db=async_db,
        books_config=books_config,
        max_concurrency=max_concurrency,
        batch_size=50,
        use_async=True
    )


async def search_example(book_name: str, query: str, max_results: int = 5) -> None:
    """Example of how to search a book using the async API."""
    load_dotenv()
    
    # Create async database connection
    async_db: AsyncDatabase = DataAPIClient(
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    ).get_async_database(
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    )
    
    # Determine which vector options to use based on the book
    if book_name == "moby_dick":
        language_options = english_vector_options
    elif book_name == "don_quixote":
        language_options = spanish_vector_options
    else:
        raise ValueError(f"Unknown book: {book_name}")
    
    # Create the table object
    table = AsyncAstraMultiVectorTable(
        db=async_db,
        table_name=book_name,
        vector_column_options=[language_options, multilingual_vector_options],
    )
    
    # Search in both vector columns
    print(f"\nSearching for '{query}' in {book_name} using language-specific model:")
    language_results = await table.search_by_text(
        query_text=query,
        vector_column=language_options.column_name,
        limit=max_results
    )
    
    for i, result in enumerate(language_results):
        print(f"{i+1}. {result['content'][:150]}...")
    
    print(f"\nSearching for '{query}' in {book_name} using OpenAI embeddings:")
    openai_results = await table.search_by_text(
        query_text=query,
        vector_column="openai_embeddings",
        limit=max_results
    )
    
    for i, result in enumerate(openai_results):
        print(f"{i+1}. {result['content'][:150]}...")


async def batch_search_example(book_name: str, queries: List[str], max_results: int = 3) -> None:
    """Example of how to perform multiple searches concurrently."""
    load_dotenv()
    
    # Create async database connection
    async_db: AsyncDatabase = DataAPIClient(
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    ).get_async_database(
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    )
    
    # Determine which vector options to use based on the book
    if book_name == "moby_dick":
        language_options = english_vector_options
    elif book_name == "don_quixote":
        language_options = spanish_vector_options
    else:
        raise ValueError(f"Unknown book: {book_name}")
    
    # Create the table object
    table = AsyncAstraMultiVectorTable(
        db=async_db,
        table_name=book_name,
        vector_column_options=[language_options, multilingual_vector_options],
    )
    
    # Execute multiple searches in parallel
    print(f"\nPerforming {len(queries)} searches in {book_name} in parallel:")
    all_results = await table.batch_search_by_text(
        queries=queries,
        vector_column=language_options.column_name,
        limit=max_results
    )
    
    # Display results for each query
    for i, (query, results) in enumerate(zip(queries, all_results)):
        print(f"\nResults for query: '{query}'")
        for j, result in enumerate(results):
            print(f"{j+1}. {result['content'][:100]}...")


async def main():
    """Main async function to run the examples."""
    # Choose which operations to run
    should_ingest = True
    should_search = True
    
    if should_ingest:
        print("Ingesting books...")
        await ingest_books_async(books_config)
    
    if should_search:
        # English book search example
        await search_example(
            book_name="moby_dick",
            query="What is the white whale?",
            max_results=3
        )
        
        # Spanish book search example
        await search_example(
            book_name="don_quixote",
            query="¿Quién es Sancho Panza?",
            max_results=3
        )
        
        # Batch search examples
        await batch_search_example(
            book_name="moby_dick",
            queries=[
                "Describe Captain Ahab",
                "How does the story begin?",
                "What is the significance of the sea?"
            ],
            max_results=2
        )
        
        await batch_search_example(
            book_name="don_quixote",
            queries=[
                "Describe los molinos de viento",
                "¿Cómo describe a Dulcinea?",
                "El caballero de la triste figura"
            ],
            max_results=2
        )


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())