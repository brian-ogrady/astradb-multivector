import uuid
import asyncio
from typing import List, Dict, Any, Callable, TypeVar, Awaitable

T = TypeVar('T')
R = TypeVar('R')

from astrapy import AsyncDatabase, AsyncTable
from astrapy.info import (
    AlterTableAddVectorize,
    ColumnType,
    CreateTableDefinition,
)

from astra_multivector.vector_column_options import VectorColumnOptions


class AsyncAstraMultiVectorTable:
    """An async class for storing and retrieving text chunks with vector embeddings.
    
    This class handles the creation of database tables with vector columns,
    creation of vector indexes, and the insertion of text chunks with their
    associated embeddings (either computed client-side or with vectorize).
    
    Example:
        ```python
        # Setup database connection
        from astrapy import AsyncDatabase
        db = AsyncDatabase(token="your-token", api_endpoint="your-endpoint")
        
        # Create embedding models
        from sentence_transformers import SentenceTransformer
        english_model = SentenceTransformer("intfloat/e5-large-v2")
        
        # Create column options
        english_options = VectorColumnOptions.from_sentence_transformer(english_model)
        
        # Create vectorize options for multilingual content
        from astrapy.info import VectorServiceOptions
        multilingual_options = VectorColumnOptions.from_vectorize(
            column_name="multi_embeddings",
            dimension=1536,
            vector_service_options=VectorServiceOptions(
                provider='openai',
                model_name='text-embedding-3-small',
                authentication={
                    "providerKey": "openaikey_astra_kms_alias",
                },
            )
        )
        
        # Create the table
        vector_table = AsyncAstraMultiVectorTable(
            database=db,
            table_name="hamlet",
            vector_column_options=[english_options, multilingual_options],
            default_concurrency_limit=5
        )
        
        # Insert text chunks
        await vector_table.insert_chunk("To be or not to be, that is the question.")
        
        # Bulk insert with custom concurrency limit
        await vector_table.bulk_insert_chunks(
            text_chunks=["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"],
            max_concurrency=10
        )
        ```
    """

    def __init__(
        self,
        db: AsyncDatabase,
        table_name: str,
        vector_column_options: List[VectorColumnOptions],
        default_concurrency_limit: int = 10,
    ) -> None:
        self.db = db
        self.name = table_name
        self.vector_column_options = vector_column_options
        self.default_concurrency_limit = default_concurrency_limit
        self.table = None
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _initialize(self) -> None:
        """Initialize the table if not already initialized."""
        async with self._init_lock:
            if not self._initialized:
                self.table = await self._create_table()
                self._initialized = True

    async def _create_table(self) -> AsyncTable:
        """Creates and configures a table with vector search capabilities."""
        schema = (
            CreateTableDefinition.builder()
            .add_column("chunk_id", ColumnType.UUID)
            .add_column("content", ColumnType.TEXT)
            .add_partition_by(["chunk_id"])
        )
        for options in self.vector_column_options:
            schema = schema.add_vector_column(
                options.column_name,
                dimension=options.dimension,
            )
        
        # Create the table asynchronously 
        # If using astrapy 2.0.0rc0, this assumes Database.create_table is async
        table = await self.db.create_table(
            self.name,
            definition=schema.build(),
            if_not_exists=True,
        )

        # Create vector indexes
        for options in self.vector_column_options:
            await table.create_vector_index(
                f"{self.name}_{options.column_name}_idx",
                column=options.column_name,
                options=options.table_vector_index_options,
                if_not_exists=True,
            )

        vectorize_options = {
            options.column_name: options.vector_service_options 
            for options in self.vector_column_options if options.vector_service_options
        }

        if len(vectorize_options) > 0:
            table = await table.alter(
                AlterTableAddVectorize(columns=vectorize_options)
            )
        
        return table
    
    async def insert_chunk(self, text_chunk: str) -> None:
        """Insert a text chunk & embeddings(s) into the table asynchronously"""
        if not self._initialized:
            await self._initialize()
            
        chunk_id = uuid.uuid4()

        insertions = {"chunk_id": chunk_id, "content": text_chunk}
        for options in self.vector_column_options:
            if options.vector_service_options:
                insertions[options.column_name] = text_chunk
            else:
                # For client-side encoding, we may need to run in a thread pool
                # since sentence-transformers may not be async-friendly
                embedding = await asyncio.to_thread(
                    lambda: options.model.encode(text_chunk).tolist()
                )
                insertions[options.column_name] = embedding
        
        await self.table.insert_one(insertions)

    async def search_by_text(
        self, 
        query_text: str, 
        vector_column: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for chunks by text similarity.
        
        Args:
            query_text: The text to search for.
            vector_column: The vector column to use for the search. If None, uses the first column.
            limit: Maximum number of results to return.
        
        Returns:
            A list of matching documents.
        """
        if not self._initialized:
            await self._initialize()
            
        if vector_column is None:
            # Use the first vector column by default
            vector_column = self.vector_column_options[0].column_name
            
        # Find the appropriate vector column options
        options = next((opt for opt in self.vector_column_options 
                        if opt.column_name == vector_column), None)
        if not options:
            raise ValueError(f"Vector column '{vector_column}' not found")
            
        if options.vector_service_options:
            # For vectorize, use the text directly
            query = query_text
        else:
            # For client-side embedding
            query = await asyncio.to_thread(
                lambda: options.model.encode(query_text).tolist()
            )
            
        # Perform the vector search using sort with the vector column
        cursor = self.table.find(
            filter={},
            sort={vector_column: query},
            limit=limit
        )
        
        results = await cursor.to_list()
        return results

    async def bulk_insert_chunks(
        self, 
        text_chunks: List[str], 
        max_concurrency: int = None, 
        batch_size: int = 100
    ) -> None:
        """Insert multiple text chunks with concurrency control.
        
        Args:
            text_chunks: List of text chunks to insert
            max_concurrency: Maximum number of concurrent operations (defaults to self.default_concurrency_limit)
            batch_size: Number of documents to insert in a single batch operation
        """
        if not self._initialized:
            await self._initialize()
        
        if max_concurrency is None:
            max_concurrency = self.default_concurrency_limit
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # Process text chunks in batches
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            
            # Process embeddings with concurrency control
            async def process_chunk(text_chunk):
                async with semaphore:
                    chunk_id = uuid.uuid4()
                    insertion = {"chunk_id": chunk_id, "content": text_chunk}
                    
                    for options in self.vector_column_options:
                        if options.vector_service_options:
                            insertion[options.column_name] = text_chunk
                        else:
                            # For client-side encoding, we need to run in a thread pool
                            embedding = await asyncio.to_thread(
                                lambda text=text_chunk: options.model.encode(text).tolist()
                            )
                            insertion[options.column_name] = embedding
                            
                    return insertion
            
            # Create tasks for all chunks in the current batch
            tasks = [process_chunk(chunk) for chunk in batch]
            
            # Wait for all tasks to complete and collect results
            batch_inserts = await asyncio.gather(*tasks)
            
            # Insert the batch
            if batch_inserts:
                await self.table.insert_many(batch_inserts)
                
    async def parallel_process_chunks(
        self,
        items: List[T],
        process_fn: Callable[[T], Awaitable[R]],
        max_concurrency: int = None
    ) -> List[R]:
        """Process items in parallel with a custom processing function.
        
        Args:
            items: List of items to process
            process_fn: Async function that takes an item and returns a processed result
            max_concurrency: Maximum number of concurrent operations
            
        Returns:
            List of processed results in the same order as the input items
        """
        if max_concurrency is None:
            max_concurrency = self.default_concurrency_limit
            
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await process_fn(item)
                
        tasks = [process_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)
        
    async def batch_search_by_text(
        self,
        queries: List[str],
        vector_column: str = None,
        limit: int = 10,
        max_concurrency: int = None
    ) -> List[List[Dict[str, Any]]]:
        """Perform multiple text similarity searches in parallel.
        
        Args:
            queries: List of text queries to search for
            vector_column: The vector column to use for the search
            limit: Maximum number of results to return per query
            max_concurrency: Maximum number of concurrent search operations
            
        Returns:
            List of search results for each query
        """
        async def search_single_query(query):
            return await self.search_by_text(
                query_text=query,
                vector_column=vector_column,
                limit=limit
            )
            
        return await self.parallel_process_chunks(
            items=queries,
            process_fn=search_single_query,
            max_concurrency=max_concurrency
        )