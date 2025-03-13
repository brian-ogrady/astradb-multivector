import asyncio
import concurrent.futures
import requests
from typing import List, Optional, Union, Any

from astrapy.database import Database, AsyncDatabase
from langchain.text_splitter import TextSplitter
from tqdm import tqdm

from astra_multivector import AstraMultiVectorTable, AsyncAstraMultiVectorTable, VectorColumnOptions


def download_gutenberg_book(book_id: int) -> Optional[str]:
    """Download a book from Project Gutenberg using its ID."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to download book {book_id}. Status: {response.status_code}")
        
        
async def download_gutenberg_book_async(book_id: int) -> Optional[str]:
    """Download a book from Project Gutenberg using its ID asynchronously.
    
    This function runs the synchronous download in a thread pool to avoid
    blocking the event loop.
    """
    return await asyncio.to_thread(download_gutenberg_book, book_id)


def download_and_ingest_book(
    db: Database,
    book_name: str,
    book_id: int,
    text_splitter: TextSplitter,
    vector_column_options: List[VectorColumnOptions],
    max_workers: int = 4,
) -> None:
    """Download and ingest a book from Project Gutenberg into a vector database.
    
    Args:
        database: The AstraDB database instance
        book_name: Name to use for the database table
        book_id: The Project Gutenberg ID of the book
        text_splitter: The text splitter to use for chunking the book
        *vector_column_options: Vector column options for embeddings
        max_workers: Maximum number of concurrent workers for chunk insertion
    """
    table = AstraMultiVectorTable(
        db=db,
        table_name=book_name,
        vector_column_options=vector_column_options,
    )
    
    text = download_gutenberg_book(book_id)
    
    chunks = text_splitter.split_text(text)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(table.insert_chunk, chunks),
            total=len(chunks),
            desc=f"Ingesting {book_name}"
        ))
    
    print(f"Successfully ingested book {book_name} into table")


async def download_and_ingest_book_async(
    db: AsyncDatabase,
    book_name: str,
    book_id: int,
    text_splitter: TextSplitter,
    vector_column_options: List[VectorColumnOptions],
    max_concurrency: int = 10,
    batch_size: int = 100
) -> None:
    """Download and ingest a book from Project Gutenberg into a vector database using async operations.
    
    Args:
        db: The AsyncAstraDB database instance
        book_name: Name to use for the database table
        book_id: The Project Gutenberg ID of the book
        text_splitter: The text splitter to use for chunking the book
        vector_column_options: Vector column options for embeddings
        max_concurrency: Maximum number of concurrent operations
        batch_size: Size of batches for bulk operations
    """
    # Create async table
    table = AsyncAstraMultiVectorTable(
        db=db,
        table_name=book_name,
        vector_column_options=vector_column_options,
        default_concurrency_limit=max_concurrency
    )
    
    # Download the book
    text = await download_gutenberg_book_async(book_id)
    
    # Split into chunks
    chunks = text_splitter.split_text(text)
    
    # Process in batches with progress reporting
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"Ingesting {book_name} - batch {batch_num}/{total_batches} ({len(batch)} chunks)")
        await table.bulk_insert_chunks(
            text_chunks=batch,
            max_concurrency=max_concurrency,
            batch_size=batch_size
        )
    
    print(f"Successfully ingested book {book_name} into table")


async def download_and_ingest_multiple_books(
    db: Union[Database, AsyncDatabase],
    books_config: List[dict],
    default_text_splitter: Optional[TextSplitter] = None,
    max_workers: int = 4,
    max_concurrency: int = 10,
    batch_size: int = 100,
    use_async: bool = True
) -> None:
    """Concurrently download and ingest multiple books from Project Gutenberg.
    
    Args:
        db: The AstraDB database instance (sync or async)
        books_config: List of dictionaries containing:
            - book_name: Name for the table
            - book_id: Project Gutenberg ID
            - vector_column_options: List of VectorColumnOptions for this specific book
            - text_splitter: (Optional) Language-specific TextSplitter for this book
        default_text_splitter: Default splitter to use if not specified in book config
        max_workers: Maximum number of concurrent workers per book (for sync operations)
        max_concurrency: Maximum number of concurrent operations (for async operations)
        batch_size: Size of batches for bulk operations
        use_async: Whether to use async operations (requires AsyncDatabase)
    
    Example:
        ```python
        books_config = [
            {
                "book_name": "moby_dick", 
                "book_id": 2701,
                "vector_column_options": [english_vector_options, multilingual_vector_options],
                "text_splitter": RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ". ", "! ", "? "],
                    chunk_size=500
                )
            },
            {
                "book_name": "don_quixote", 
                "book_id": 2000,
                "vector_column_options": [spanish_vector_options, multilingual_vector_options],
                "text_splitter": RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ". ", "¡", "¿", "! ", "? "],
                    chunk_size=500
                )
            }
        ]
        
        # For sync operations:
        from astrapy import Database
        db = Database(token="token", api_endpoint="endpoint")
        await download_and_ingest_multiple_books(db, books_config, use_async=False)
        
        # For async operations:
        from astrapy import AsyncDatabase
        async_db = AsyncDatabase(token="token", api_endpoint="endpoint")
        await download_and_ingest_multiple_books(async_db, books_config, use_async=True)
        ```
    """
    tasks = []
    
    if use_async and not isinstance(db, AsyncDatabase):
        raise TypeError("use_async=True requires an AsyncDatabase instance")
    
    for book_config in books_config:
        book_name = book_config["book_name"]
        book_id = book_config["book_id"]
        vector_options = book_config.get("vector_column_options", [])
        
        text_splitter = book_config.get("text_splitter", default_text_splitter)
        if text_splitter is None:
            raise ValueError(f"No text splitter provided for book {book_name} and no default splitter specified")
        
        if use_async:
            # Use fully async pipeline
            task = asyncio.create_task(
                download_and_ingest_book_async(
                    db,
                    book_name,
                    book_id,
                    text_splitter,
                    vector_options,
                    max_concurrency=max_concurrency,
                    batch_size=batch_size
                )
            )
        else:
            # Use thread-based async execution with synchronous operations
            task = asyncio.create_task(
                asyncio.to_thread(
                    download_and_ingest_book,
                    db,
                    book_name,
                    book_id,
                    text_splitter,
                    vector_options,
                    max_workers=max_workers
                )
            )
        
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    print(f"Successfully ingested {len(books_config)} books")
