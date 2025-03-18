import asyncio
import uuid
import warnings
from typing import List, Dict, Any, Callable, TypeVar, Awaitable, Optional, Tuple, Union

T = TypeVar('T')
R = TypeVar('R')

from astrapy import AsyncDatabase, AsyncTable
from astrapy.info import (
    AlterTableAddVectorize,
    ColumnType,
    CreateTableDefinition,
)
from astrapy.results import TableInsertManyResult, TableInsertOneResult
from reranker import Reranker, RankedResults

from astra_multivector.vector_column_options import (
    VectorColumnOptions,
    VectorColumnType,
)


class AsyncAstraMultiVectorTable:
    """An async class for storing and retrieving text chunks with vector embeddings.
    
    This class provides asynchronous operations for working with vector embeddings in AstraDB,
    with built-in concurrency controls for efficient batch operations. It handles the creation 
    of database tables with vector columns, creation of vector indexes, and the insertion of 
    text chunks with their associated embeddings.
    
    Key features:
    - Fully asynchronous API compatible with asyncio
    - Lazy initialization (tables are created only when needed)
    - Configurable concurrency limits for batch operations
    - Support for client-side, server-side, and pre-computed embeddings
    - Thread pool offloading for CPU-intensive embedding operations
    
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
                    "providerKey": "OPENAI_API_KEY",
                },
            )
        )
        
        # Create the table
        vector_table = AsyncAstraMultiVectorTable(
            db=db,
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
        
        # Perform batch searches in parallel
        queries = ["Hamlet question", "To be or not to be", "Shakespeare plays"]
        results = await vector_table.batch_search_by_text(
            queries=queries,
            vector_column="multi_embeddings",
            limit=5
        )
        
        # Process custom operations in parallel
        async def process_item(item):
            # Custom async processing logic
            return f"Processed {item}"
            
        processed = await vector_table.parallel_process_chunks(
            items=["item1", "item2", "item3"],
            process_fn=process_item,
            max_concurrency=5
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

        table = await self.db.create_table(
            self.name,
            definition=schema.build(),
            if_not_exists=True,
        )

        for options in self.vector_column_options:
            await table.create_vector_index(
                f"{self.name}_{options.column_name}_idx",
                column=options.column_name,
                options=options.table_vector_index_options,
                if_not_exists=True,
            )
            if options.type == VectorColumnType.VECTORIZE:
                table = await table.alter(
                    AlterTableAddVectorize(columns=options.vector_service_options)
                )
        
        return table
    
    async def _get_embedding_for_column(
        self,
        text: str,
        column_options: VectorColumnOptions,
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None
    ) -> List[float]:
        """Generate an embedding for a specific column based on its type.
        
        Args:
            text: The text to generate an embedding for
            column_options: The options for the vector column
            precomputed_embeddings: Dictionary of precomputed embeddings (for PRECOMPUTED columns)
            
        Returns:
            The embedding vector as a list of floats
            
        Raises:
            ValueError: If a precomputed embedding is required but not provided
        """
        precomputed_embeddings = precomputed_embeddings or {}
        
        if column_options.type == VectorColumnType.VECTORIZE:
            return text
        elif column_options.type == VectorColumnType.SENTENCE_TRANSFORMER:
            return await asyncio.to_thread(
                lambda: column_options.model.encode(text).tolist()
            )
        elif column_options.type == VectorColumnType.PRECOMPUTED:
            column_name = column_options.column_name
            if column_name not in precomputed_embeddings:
                raise ValueError(
                    f"Precomputed embedding required for column '{column_name}' but not provided"
                )
            return precomputed_embeddings[column_name]
    
    async def _add_embedding_to_insertion(
        self,
        insertion: Dict[str, Any],
        text_chunk: str,
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Any]:
        """Add embeddings for all vector columns to the insertion dictionary.
        
        Args:
            insertion: The dictionary to add embeddings to
            text_chunk: The text to generate embeddings for
            precomputed_embeddings: Dictionary of precomputed embeddings
            
        Returns:
            The updated insertion dictionary with embeddings added
        """
        for options in self.vector_column_options:
            embedding = await self._get_embedding_for_column(
                text=text_chunk,
                column_options=options,
                precomputed_embeddings=precomputed_embeddings
            )
            insertion[options.column_name] = embedding
            
        return insertion
    
    async def insert_chunk(
        self,
        text_chunk: str,
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
        **kwargs,
    ) -> TableInsertOneResult:
        """Insert a text chunk & embeddings(s) into the table asynchronously.
    
        Args:
            text_chunk: The text content to insert and embed.
            precomputed_embeddings: Dictionary mapping column names to precomputed 
                vector embeddings for columns of type PRECOMPUTED
            **kwargs: Additional arguments passed to the underlying insert_one method
            
        Notes:
            - Client-side embedding operations are offloaded to a thread pool to avoid blocking the event loop
            - For vectorize columns, the raw text is stored directly
            - A unique UUID is generated for each chunk
            For complete details on all available parameters, see the AstraPy documentation:
            https://docs.datastax.com/en/astra-api-docs/_attachments/python-client/astrapy/index.html#astrapy.AsyncTable.insert_one
        """
        if not self._initialized:
            await self._initialize()
            
        chunk_id = uuid.uuid4()

        insertion = {"chunk_id": chunk_id, "content": text_chunk}

        insertion = await self._add_embedding_to_insertion(
            insertion,
            text_chunk,
            precomputed_embeddings or {},
        )
        
        return await self.table.insert_one(
            insertion,
            **kwargs
        )
    
    async def _process_chunk_with_semaphore(
        self,
        j: int, 
        text_chunk: str, 
        semaphore: asyncio.Semaphore,
        precomputed_embeddings: Optional[Dict[str, List[List[float]]]] = None, 
    ) -> Dict[str, Any]:
        """Process a single chunk with semaphore control.
        
        Args:
            j: Index of the chunk
            text_chunk: The text content to process
            semaphore: Semaphore for concurrency control
            precomputed_embeddings: Dictionary of precomputed embeddings
            
        Returns:
            Dictionary with chunk_id, content, and embeddings ready for insertion
        """
        async with semaphore:
            chunk_id = uuid.uuid4()
            insertion = {"chunk_id": chunk_id, "content": text_chunk}
            
            precomputed_embeddings = precomputed_embeddings or {}

            chunk_precomputed = {}
            for col in precomputed_embeddings.keys():
                chunk_precomputed[col] = precomputed_embeddings[col][j]
            
            insertion = await self._add_embedding_to_insertion(
                insertion, text_chunk, chunk_precomputed
            )
            
            return insertion

    async def bulk_insert_chunks(
        self, 
        text_chunks: List[str],
        precomputed_embeddings: Optional[Dict[str, List[List[float]]]] = None, 
        max_parallel_embeddings: int = None,
        **kwargs
    ) -> Optional[TableInsertManyResult]:
        """Insert multiple text chunks with efficient parallel embedding.
        
        Args:
            text_chunks: List of text chunks to insert
            precomputed_embeddings: Dictionary mapping column names to lists of embeddings,
                where each list corresponds to a text chunk at the same index
            max_parallel_embeddings: Maximum number of embeddings to generate simultaneously
                                (defaults to self.default_concurrency_limit)
            **kwargs: Additional arguments passed to the underlying insert_many method
                
        Returns:
            The result from the underlying insert_many operation if any rows were inserted
        """
        if not self._initialized:
            await self._initialize()
        
        precomputed_embeddings = precomputed_embeddings or {}
        
        precomputed_columns = [opt.column_name for opt in self.vector_column_options 
                            if opt.type == VectorColumnType.PRECOMPUTED]
        
        for col in precomputed_columns:
            if col not in precomputed_embeddings:
                raise ValueError(f"Precomputed embeddings required for column '{col}' but not provided")
            if len(precomputed_embeddings[col]) < len(text_chunks):
                raise ValueError(f"Not enough precomputed embeddings for column '{col}'")
        
        max_parallel_embeddings = max_parallel_embeddings or self.default_concurrency_limit
        
        semaphore = asyncio.Semaphore(max_parallel_embeddings)
        
        tasks = [
            self._process_chunk_with_semaphore(
                j, chunk, semaphore, precomputed_embeddings
            ) 
            for j, chunk in enumerate(text_chunks)
        ]
        
        batch_inserts = await asyncio.gather(*tasks)
        
        if batch_inserts:
            return await self.table.insert_many(batch_inserts, **kwargs)
    
    async def _search_column(
        self,
        column_name: str,
        query_text: str,
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
        candidates_per_column: int = 10,
        **kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Search a single vector column for similar documents.
        
        Args:
            column_name: The name of the vector column to search
            query_text: The text query to search for
            precomputed_embeddings: Optional pre-computed query embeddings for PRECOMPUTED columns
            candidates_per_column: Number of candidates to retrieve from this column
            **kwargs: Additional parameters passed to the underlying find method
            
        Returns:
            A tuple of (column_name, search_results) with the column's search results
        """
        options = next(opt for opt in self.vector_column_options if opt.column_name == column_name)
        
        query = await self._get_embedding_for_column(
            text=query_text,
            column_options=options,
            precomputed_embeddings=precomputed_embeddings
        )
        
        cursor = self.table.find(
            sort={column_name: query},
            limit=candidates_per_column,
            include_similarity=True,
            **kwargs
        )
        
        col_results = await cursor.to_list()
        return column_name, col_results
            
    async def multi_vector_similarity_search(
        self, 
        query_text: str,
        vector_columns: Optional[List[str]] = None,
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
        candidates_per_column: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search across multiple vector columns and combine results asynchronously.
        
        Args:
            query_text: The text query to search for
            vector_columns: List of vector columns to search. If None, uses all columns.
            precomputed_embeddings: Optional pre-computed query embeddings for PRECOMPUTED columns
            candidates_per_column: Number of candidates to retrieve per column (default: 10)
            **kwargs: Additional parameters passed to the underlying find method
            
        Returns:
            A list of unique matching documents from all vector columns, including information 
            about which columns returned each document, at what rank, and with what similarity score
        """
        if not self._initialized:
            await self._initialize()
            
        precomputed_embeddings = precomputed_embeddings or {}

        vector_columns = vector_columns or [opt.column_name for opt in self.vector_column_options]
        
        for col in vector_columns:
            if not any(opt.column_name == col for opt in self.vector_column_options):
                raise ValueError(f"Vector column '{col}' not found")
        
        all_results = []
        doc_id_to_result = {}
        doc_id_to_columns = {}
        
        search_tasks = [
            self._search_column(
                column_name=col,
                query_text=query_text,
                precomputed_embeddings=precomputed_embeddings,
                candidates_per_column=candidates_per_column,
                **kwargs
            ) 
            for col in vector_columns
        ]
        search_results = await asyncio.gather(*search_tasks)
        
        for col, col_results in search_results:
            for rank, doc in enumerate(col_results):
                doc_id, similarity = doc.get("chunk_id"), doc.get("$similarity")
                if not doc_id:
                    warnings.warn(f"Document without chunk_id found in column '{col}' results (rank {rank+1}). Skipping this document.")
                    continue
                
                if doc_id in doc_id_to_result:
                    doc_id_to_columns[doc_id].append({
                        "column": col,
                        "rank": rank + 1,
                        "similarity": similarity
                    })
                else:
                    doc_id_to_result[doc_id] = doc
                    doc_id_to_columns[doc_id] = [{
                        "column": col,
                        "rank": rank + 1,
                        "similarity": similarity
                    }]
                    all_results.append(doc)
        
        for doc in all_results:
            doc_id = doc.get("chunk_id")
            if doc_id and doc_id in doc_id_to_columns:
                doc["source_columns"] = doc_id_to_columns[doc_id]
        
        return all_results
    
    async def rerank_results(
        self,
        query_text: str,
        results: List[Dict[str, Any]],
        reranker: Reranker,
        limit: Optional[int] = None
    ) -> RankedResults:
        """Rerank search results using a reranker.
        
        Args:
            query_text: The original query text
            results: List of results to rerank (typically from multi_vector_similarity_search)
            reranker: Reranker instance to use for reranking
            limit: Maximum number of results to return. If None, returns all reranked results.
            
        Returns:
            The RankedResults object from the reranker, potentially limited to the specified number of results
        """
        if not results:
            return RankedResults(results=[])
            
        texts = [doc["content"] for doc in results]
        doc_ids = [doc["chunk_id"] for doc in results]
        
        ranked_results = await asyncio.to_thread(
            lambda: reranker.rank(query=query_text, docs=texts, doc_ids=doc_ids)
        )
        
        if limit is not None and limit < len(ranked_results.results):
            ranked_results.results = ranked_results.results[:limit]
            
        return ranked_results

    async def search_and_rerank(
        self,
        query_text: str,
        reranker: Reranker,
        vector_columns: Optional[List[str]] = None,
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
        candidates_per_column: int = 10,
        rerank_limit: Optional[int] = None,
        **kwargs
    ) -> RankedResults:
        """Search across multiple vector columns and rerank results asynchronously.
        
        Args:
            query_text: The text query to search for
            reranker: Reranker instance to use for reranking
            vector_columns: List of vector columns to search
            precomputed_embeddings: Pre-computed query embeddings for PRECOMPUTED columns
            candidates_per_column: Number of candidates to retrieve per column
            rerank_limit: Maximum number of results after reranking
            **kwargs: Additional parameters for the underlying find method
            
        Returns:
            Reranked search results
        """
        rerank_limit = rerank_limit or candidates_per_column
        
        search_results = await self.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=vector_columns,
            precomputed_embeddings=precomputed_embeddings,
            candidates_per_column=candidates_per_column,
            **kwargs
        )
        
        reranked_results = await self.rerank_results(
            query_text=query_text,
            results=search_results, 
            reranker=reranker,
            limit=rerank_limit
        )
        
        return reranked_results
                
    async def parallel_process_chunks(
        self,
        items: List[T],
        process_fn: Callable[[T], Awaitable[R]],
        max_concurrency: int = None
    ) -> List[R]:
        """Process items in parallel with a custom processing function.
        
        This utility method allows you to apply any async function to a list of items
        with controlled parallelism. It's useful for custom processing workflows that
        need to be executed efficiently in parallel.
        
        Args:
            items: List of items to process
            process_fn: Async function that takes an item and returns a processed result
            max_concurrency: Maximum number of concurrent operations (defaults to self.default_concurrency_limit)
            
        Returns:
            List of processed results in the same order as the input items
            
        Example:
            ```python
            # Define a custom async processing function
            async def process_doc(doc_id):
                doc = await fetch_document(doc_id)
                summary = await summarize_text(doc.content)
                return {"id": doc_id, "summary": summary}
                
            # Process multiple documents in parallel
            summaries = await table.parallel_process_chunks(
                items=doc_ids,
                process_fn=process_doc,
                max_concurrency=5
            )
            ```
        """
        max_concurrency = max_concurrency or self.default_concurrency_limit
            
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await process_fn(item)
                
        tasks = [process_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)
    
    async def _process_single_query(
        self,
        query_data: Tuple[int, str],
        vector_columns: Optional[List[str]] = None,
        precomputed_embeddings: Optional[List[Dict[str, List[float]]]] = None,
        candidates_per_column: int = 10,
        rerank: bool = False,
        reranker: Optional[Reranker] = None,
        rerank_limit: Optional[int] = None,
        **kwargs
    ) -> Union[List[Dict[str, Any]], RankedResults]:
        """Process a single query with optional reranking.
        
        Args:
            query_data: Tuple of (index, query_text)
            vector_columns: List of vector columns to search
            precomputed_embeddings: List of precomputed embeddings, one per query
            candidates_per_column: Number of candidates to retrieve per column
            rerank: Whether to rerank the results
            reranker: Reranker instance to use
            rerank_limit: Maximum number of results after reranking
            **kwargs: Additional parameters for vector search
            
        Returns:
            Either raw search results or reranked results depending on parameters
        """
        i, query = query_data
        
        query_embeddings = None
        if precomputed_embeddings and i < len(precomputed_embeddings):
            query_embeddings = precomputed_embeddings[i]
            
        if rerank and reranker:
            return await self.search_and_rerank(
                query_text=query,
                reranker=reranker,
                vector_columns=vector_columns,
                precomputed_embeddings=query_embeddings,
                candidates_per_column=candidates_per_column,
                rerank_limit=rerank_limit,
                **kwargs
            )
        else:
            return await self.multi_vector_similarity_search(
                query_text=query,
                vector_columns=vector_columns,
                precomputed_embeddings=query_embeddings,
                candidates_per_column=candidates_per_column,
                **kwargs
            )
        
    async def batch_search_by_text(
        self,
        queries: List[str],
        vector_columns: Optional[List[str]] = None,
        precomputed_embeddings: Optional[List[Dict[str, List[float]]]] = None,
        candidates_per_column: int = 10,
        rerank: bool = False,
        reranker: Optional[Reranker] = None,
        rerank_limit: Optional[int] = None,
        max_concurrency: int = None,
        **kwargs
    ) -> Union[List[List[Dict[str, Any]]], List[RankedResults]]:
        """Perform multiple text similarity searches across vector columns with optional reranking.
        
        Args:
            queries: List of text queries to search for
            vector_columns: List of vector columns to search. If None, uses all columns.
            precomputed_embeddings: Optional list of pre-computed query embeddings for 
                                PRECOMPUTED columns, one dict per query
            candidates_per_column: Number of candidates to retrieve per column (default: 10)
            rerank: Whether to rerank the results (default: False)
            reranker: Reranker instance to use for reranking. Required if rerank=True.
            rerank_limit: Maximum number of results to return after reranking.
                        If None, returns all reranked results.
            max_concurrency: Maximum number of concurrent search operations
                            (defaults to self.default_concurrency_limit)
            **kwargs: Additional parameters passed to the underlying find method
            
        Returns:
            If rerank=False: List of search results for each query
            If rerank=True: List of RankedResults objects for each query
            
        Raises:
            ValueError: If rerank=True but no reranker is provided, or if the number of
                    precomputed embeddings doesn't match the number of queries
        """
        if not self._initialized:
            await self._initialize()
            
        if rerank and not reranker:
            raise ValueError("reranker must be provided when rerank=True")
            
        if precomputed_embeddings and len(precomputed_embeddings) != len(queries):
            raise ValueError(
                f"Number of precomputed embeddings ({len(precomputed_embeddings)}) "
                f"must match number of queries ({len(queries)})"
            )
        
        async def process_fn(query_data):
            return await self._process_single_query(
                query_data=query_data,
                vector_columns=vector_columns,
                precomputed_embeddings=precomputed_embeddings,
                candidates_per_column=candidates_per_column,
                rerank=rerank,
                reranker=reranker,
                rerank_limit=rerank_limit,
                **kwargs
            )
        
        items = list(enumerate(queries))
        results = await self.parallel_process_chunks(
            items=items,
            process_fn=process_fn,
            max_concurrency=max_concurrency
        )
        
        return results
