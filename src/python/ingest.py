import asyncio
import concurrent.futures
import requests
from typing import List, Optional

from astrapy import database
from langchain.text_splitter import TextSplitter
from tqdm import tqdm

from .astra_multi_vector_table import AstraMultiVectorTable
from .vector_column_options import VectorColumnOptions


def download_gutenberg_book(book_id: int) -> Optional[str]:
    """Download a book from Project Gutenberg using its ID."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to download book {book_id}. Status: {response.status_code}")


def download_and_ingest_book(
    database: database,
    book_name: str,
    book_id: int,
    text_splitter: TextSplitter,
    *vector_column_options: VectorColumnOptions,
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
        database=database,
        table_name=book_name,
        *vector_column_options,
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


async def download_and_ingest_multiple_books(
    db: database,
    books_config: List[dict],
    default_text_splitter: Optional[TextSplitter] = None,
    max_workers: int = 4
) -> None:
    """Concurrently download and ingest multiple books from Project Gutenberg.
    
    Args:
        db: The AstraDB database instance
        books_config: List of dictionaries containing:
            - book_name: Name for the table
            - book_id: Project Gutenberg ID
            - vector_column_options: List of VectorColumnOptions for this specific book
            - text_splitter: (Optional) Language-specific TextSplitter for this book
        default_text_splitter: Default splitter to use if not specified in book config
        max_workers: Maximum number of concurrent workers per book
    
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
            },
            {
                "book_name": "bras_cubas", 
                "book_id": 54829,
                "vector_column_options": [portuguese_vector_options, multilingual_vector_options],
                "text_splitter": RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ". ", "! ", "? "],
                    chunk_size=500
                )
            }
        ]
        ```
    """
    tasks = []
    
    for book_config in books_config:
        book_name = book_config["book_name"]
        book_id = book_config["book_id"]
        vector_options = book_config.get("vector_column_options", [])
        
        text_splitter = book_config.get("text_splitter", default_text_splitter)
        if text_splitter is None:
            raise ValueError(f"No text splitter provided for book {book_name} and no default splitter specified")
        
        task = asyncio.create_task(
            asyncio.to_thread(
                download_and_ingest_book,
                db,
                book_name,
                book_id,
                text_splitter,
                *vector_options,
                max_workers=max_workers
            )
        )
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    print(f"Successfully ingested {len(books_config)} books")
