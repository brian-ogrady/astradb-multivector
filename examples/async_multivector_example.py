#!/usr/bin/env python3
"""
Async Multi-Vector Table Example with Reranking

This example demonstrates how to use the AsyncAstraMultiVectorTable class with advanced
search functionality like multi-vector search and reranking.

Features shown:
- Setting up multiple vector columns (client-side and server-side embeddings)
- Efficient concurrent document insertion
- Multi-vector searching across columns
- Reranking with external reranker model
- Advanced parallelization techniques
"""

import asyncio
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.info import TableVectorIndexOptions, VectorServiceOptions
from sentence_transformers import SentenceTransformer
from reranker import Reranker

from astra_multivector import AsyncAstraMultiVectorTable, VectorColumnOptions


# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    "Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns.",
    "Transformers have become the dominant architecture for natural language processing tasks, replacing recurrent neural networks.",
    "The Astrapy Python SDK provides an easy way to interact with AstraDB for vector search applications.",
    "Vector databases store and retrieve high-dimensional vectors that represent embeddings of data points.",
    "Sentence transformers convert text into dense vector representations that capture semantic meaning.",
    "Cosine similarity is a measure of similarity between two non-zero vectors that measures the cosine of the angle between them.",
    "Late interaction models like ColBERT compare token-level embeddings rather than pooled document embeddings.",
    "Databases that support similarity search are essential for modern AI applications like semantic search and recommendation systems.",
    "Machine learning algorithms learn patterns from data without being explicitly programmed for specific tasks.",
    "Natural language processing (NLP) is a field of AI focused on enabling computers to understand human language."
]


async def setup_vector_table() -> AsyncAstraMultiVectorTable:
    """Set up the multi-vector table with multiple embedding models."""
    # Load environment variables
    load_dotenv()
    
    # Get AstraDB credentials
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    
    if not token or not api_endpoint:
        raise ValueError("Please set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT environment variables")
    
    # Create async database connection
    db_client = DataAPIClient(token=token)
    db = db_client.get_async_database(api_endpoint=api_endpoint)
    
    # Set up vector column options for different models
    
    # 1. Primary embedding model (client-side)
    embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    primary_options = VectorColumnOptions.from_sentence_transformer(
        model=embedding_model,
        table_vector_index_options=TableVectorIndexOptions(
            metric=VectorMetric.COSINE
        )
    )
    
    # 2. Secondary embedding model with different characteristics (client-side)
    # Using a model that captures different semantic aspects
    secondary_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    secondary_options = VectorColumnOptions.from_sentence_transformer(
        model=secondary_model,
        column_name="secondary_embeddings",
        table_vector_index_options=TableVectorIndexOptions(
            metric=VectorMetric.COSINE
        )
    )
    
    # 3. OpenAI embeddings via Vectorize (server-side)
    # Note: Requires OPENAI_API_KEY environment variable or proper key setup in Astra
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
            metric=VectorMetric.COSINE
        )
    )
    
    # Create the table with all embedding options
    vector_table = AsyncAstraMultiVectorTable(
        db=db,
        table_name="async_multivector_example",
        vector_column_options=[primary_options, secondary_options, openai_options],
        default_concurrency_limit=5
    )
    
    return vector_table


async def insert_sample_documents(table: AsyncAstraMultiVectorTable) -> None:
    """Insert sample documents with parallel processing."""
    print(f"Inserting {len(SAMPLE_DOCUMENTS)} documents...")
    
    # Insert documents in parallel with controlled concurrency
    await table.bulk_insert_chunks(
        text_chunks=SAMPLE_DOCUMENTS,
        max_parallel_embeddings=3
    )
    
    print("Documents inserted successfully")


async def simple_vector_search(table: AsyncAstraMultiVectorTable) -> None:
    """Demonstrate simple vector search on individual columns."""
    print("\n=== Simple Vector Search ===")
    
    # Search using the primary embedding model
    query = "How do neural networks work?"
    print(f"Searching for: '{query}' using primary embeddings")
    results = await table.search_by_text(
        query_text=query,
        vector_column="embeddings",  # Default from sentence_transformer
        limit=2
    )
    
    print("Results from primary embeddings:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['content']}")
    
    # Search using the OpenAI embeddings
    print(f"\nSearching for: '{query}' using OpenAI embeddings")
    results = await table.search_by_text(
        query_text=query,
        vector_column="openai_embeddings",
        limit=2
    )
    
    print("Results from OpenAI embeddings:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['content']}")


async def multi_vector_search(table: AsyncAstraMultiVectorTable) -> None:
    """Demonstrate multi-vector search across all embedding columns."""
    print("\n=== Multi-Vector Search ===")
    
    query = "databases for AI applications"
    print(f"Searching for: '{query}' across all vector columns")
    
    results = await table.multi_vector_similarity_search(
        query_text=query,
        candidates_per_column=3
    )
    
    print(f"Found {len(results)} unique documents across all columns:")
    for i, result in enumerate(results):
        # Get source columns information
        sources = ", ".join([f"{sc['column']} (rank {sc['rank']})" 
                           for sc in result['source_columns']])
        print(f"{i+1}. {result['content']}")
        print(f"   Found in: {sources}")


async def search_with_reranking(table: AsyncAstraMultiVectorTable) -> None:
    """Demonstrate search with reranking for improved results."""
    print("\n=== Search with Reranking ===")
    
    # Initialize reranker model
    reranker = Reranker.from_pretrained("BAAI/bge-reranker-base")
    
    query = "vector similarity for machine learning"
    print(f"Searching and reranking for: '{query}'")
    
    # Search and rerank in one step
    reranked_results = await table.search_and_rerank(
        query_text=query,
        reranker=reranker,
        candidates_per_column=5,  # Get 5 candidates per column
        rerank_limit=3            # Return top 3 after reranking
    )
    
    print("Top reranked results:")
    for i, result in enumerate(reranked_results.results):
        print(f"{i+1}. {result.doc} (score: {result.score:.3f})")


async def batch_search_examples(table: AsyncAstraMultiVectorTable) -> None:
    """Demonstrate batch searching with and without reranking."""
    print("\n=== Batch Search Examples ===")
    
    queries = [
        "vector databases",
        "neural networks",
        "natural language processing"
    ]
    
    print("Performing batch search without reranking...")
    results = await table.batch_search_by_text(
        queries=queries,
        vector_columns=["embeddings", "openai_embeddings"],  # Use two columns
        candidates_per_column=2,
        max_concurrency=3  # Process 3 queries at a time
    )
    
    for i, (query, query_results) in enumerate(zip(queries, results)):
        print(f"\nResults for '{query}':")
        for j, doc in enumerate(query_results):
            print(f"{j+1}. {doc['content']}")
    
    # Batch search with reranking
    print("\nPerforming batch search with reranking...")
    reranker = Reranker.from_pretrained("BAAI/bge-reranker-base")
    
    reranked_results = await table.batch_search_by_text(
        queries=queries,
        vector_columns=["embeddings", "openai_embeddings"],
        candidates_per_column=5,
        rerank=True,
        reranker=reranker,
        rerank_limit=2  # Keep top 2 reranked results per query
    )
    
    for i, (query, ranked) in enumerate(zip(queries, reranked_results)):
        print(f"\nReranked results for '{query}':")
        for j, result in enumerate(ranked.results):
            print(f"{j+1}. {result.doc} (score: {result.score:.3f})")


async def custom_parallel_processing(table: AsyncAstraMultiVectorTable) -> None:
    """Demonstrate custom parallel processing with the utility method."""
    print("\n=== Custom Parallel Processing ===")
    
    # Define a custom async processing function
    async def process_with_metadata(doc_id: int) -> Dict[str, Any]:
        # Get the content
        content = SAMPLE_DOCUMENTS[doc_id]
        
        # Simulate some processing
        await asyncio.sleep(0.2)  # Simulate work
        
        # Return enhanced result
        word_count = len(content.split())
        return {
            "id": doc_id,
            "content": content,
            "word_count": word_count,
            "processed": True
        }
    
    # Process multiple items in parallel
    print("Processing documents with custom function...")
    processed_items = await table.parallel_process_chunks(
        items=list(range(5)),  # Process first 5 documents
        process_fn=process_with_metadata,
        max_concurrency=3
    )
    
    # Display the processed items
    for item in processed_items:
        print(f"Document {item['id']}: {item['word_count']} words")
        print(f"  Preview: {item['content'][:50]}...")


async def main():
    try:
        # Set up the vector table
        table = await setup_vector_table()
        
        # Insert sample documents
        await insert_sample_documents(table)
        
        # Run the search examples
        await simple_vector_search(table)
        await multi_vector_search(table)
        await search_with_reranking(table)
        await batch_search_examples(table)
        await custom_parallel_processing(table)
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())