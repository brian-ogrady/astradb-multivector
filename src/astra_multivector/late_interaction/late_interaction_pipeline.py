import asyncio
import heapq
import uuid
import time
import logging
from collections import Counter
from typing import Iterable, List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from PIL.Image import Image
from async_lru import alru_cache
from astrapy import AsyncDatabase, AsyncTable
from astrapy.constants import SortMode, VectorMetric
from astrapy.cursors import AsyncTableFindCursor
from astrapy.info import (
    ColumnType,
    CreateTableDefinition,
    TableVectorIndexOptions
)

from .models.base import LateInteractionModel
from .utils import (
    pool_query_embeddings,
    pool_doc_embeddings,
    expand_parameter,
)

logger = logging.getLogger(__name__)


class LateInteractionPipeline:
    """
    Pipeline for late interaction retrieval models in AstraDB.
    
    Handles the creation of document and token tables, indexing of documents,
    and multi-stage retrieval using late interaction models like ColBERT.
    
    Note on metadata:
        Currently, document metadata is not supported due to limitations
        in the Data API. Future versions will enhance metadata support
        with proper indexing and filtering capabilities.
    """
    
    def __init__(
        self,
        db: AsyncDatabase,
        model: LateInteractionModel,
        base_table_name: str,
        doc_pool_factor: int = 2,
        query_pool_distance: float = 0.03,
        sim_metric: str = VectorMetric.COSINE,
        default_concurrency_limit: int = 10,
        embedding_cache_size: int = 1000,
    ):
        """
        Initialize the LateInteractionPipeline.
        
        Args:
            db: AstraDB database connection
            model: Late interaction model (ColBERT, ColPali, etc.)
            base_table_name: Base name for the document and token tables
            doc_pool_factor: Factor by which to pool document embeddings (None to disable)
            query_pool_distance: Maximum cosine distance for pooling query embeddings (0.0 to disable)
            sim_metric: Similarity metric for vector search ("cosine" or "dot_product")
            default_concurrency_limit: Default concurrency limit for async operations
            embedding_cache_size: Size of the LRU cache for document embeddings
        """
        self.db = db
        self.model = model
        self.base_table_name = base_table_name
        self.doc_pool_factor = doc_pool_factor
        self.query_pool_distance = query_pool_distance
        self.sim_metric = sim_metric
        self.default_concurrency_limit = default_concurrency_limit
        self.embedding_cache_size = embedding_cache_size
        
        self.doc_table_name = f"{base_table_name}_docs"
        self.token_table_name = f"{base_table_name}_tokens"
        
        self._doc_table = None
        self._token_table = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        logger.info(f"Initialized LateInteractionPipeline with base_table_name={base_table_name}, "
                    f"model={model.__class__.__name__}, sim_metric={sim_metric}")
    
    async def initialize(self) -> None:
        """
        Initialize document and token tables for the late interaction pipeline.
        
        This method sets up the necessary database infrastructure:
        1. Creates the document table for storing document content
        2. Creates the token table for storing token-level embeddings
        3. Creates the required vector indexes for similarity search
        
        All operations use if_not_exists=True to ensure idempotency.
        """
        async with self._init_lock:
            if not self._initialized:
                logger.info(f"Initializing tables: {self.doc_table_name} and {self.token_table_name}")
                try:
                    self._doc_table = await self._create_doc_table()
                    self._token_table = await self._create_token_table()
                    self._initialized = True
                    logger.info("Table initialization completed successfully")
                except Exception as e:
                    logger.error(f"Error initializing tables: {str(e)}")
                    raise
    
    async def _create_doc_table(self) -> AsyncTable:
        """
        Create the document table for storing document content and metadata.
        
        Defines the schema for the document table with appropriate partitioning
        and sorting to optimize data access patterns for document retrieval.
        
        TODO: In future versions of the Data API, enhance this schema to include:
         1. Proper indexing for metadata fields
         2. Searchable JSON/map column type for efficient metadata filtering
         3. Consider separating commonly filtered metadata into dedicated indexed columns
         
        Returns:
            AsyncTable: The created or existing document table
        """
        schema = (
            CreateTableDefinition.builder()
            .add_column("doc_id", ColumnType.UUID)
            .add_column("page_number", ColumnType.INT)
            .add_column("content", ColumnType.TEXT)
            .add_partition_by(["doc_id"])
            .add_partition_sort({"page_number": SortMode.ASCENDING})
        )

        if self.model.supports_images:
            schema = schema.add_column("content_type", ColumnType.TEXT)
        
        logger.debug(f"Creating document table: {self.doc_table_name}")
        doc_table = await self.db.create_table(
            self.doc_table_name,
            definition=schema.build(),
            if_not_exists=True,
        )
        
        return doc_table
    
    async def _create_token_table(self) -> AsyncTable:
        """
        Create the token table for storing token-level embeddings with vector search capabilities.
        
        Defines the schema for the token table and creates the necessary vector index
        for efficient similarity search using the configured similarity metric.
        
        Returns:
            AsyncTable: The created or existing token table with vector indexing
        """
        token_column_name = "token_embedding"

        schema = (
            CreateTableDefinition.builder()
            .add_column("doc_id", ColumnType.UUID)
            .add_column("page_number", ColumnType.INT)
            .add_column("token_id", ColumnType.UUID)
            .add_vector_column(token_column_name, dimension=self.model.dim)
            .add_partition_by(["doc_id"])
            .add_partition_sort({"page_number": SortMode.ASCENDING})
        ).build()
        
        logger.debug(f"Creating token table: {self.token_table_name} with dimension {self.model.dim}")
        token_table = await self.db.create_table(
            self.token_table_name,
            definition=schema,
            if_not_exists=True,
        )
        
        index_options = TableVectorIndexOptions(
            metric=self.sim_metric
        )

        logger.debug(f"Creating vector index for {token_column_name}")
        await token_table.create_vector_index(
            f"{self.token_table_name}_{token_column_name}_idx",
            column=token_column_name,
            options=index_options,
            if_not_exists=True,
        )
        
        return token_table
    
    @staticmethod
    async def async_find(table: AsyncTable, **kwargs) -> AsyncTableFindCursor:
        """
        Find documents in the table asynchronously.
        
        Args:
            table: The table to search
            **kwargs: Additional keyword arguments to pass to the find method
            
        Returns:
            AsyncTableFindCursor: Cursor for accessing the query results
        """
        cursor: AsyncTableFindCursor = table.find(**kwargs)

        return cursor
    
    async def _validate_row(self, document_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and prepare a document row for insertion into the database.
        
        This method performs comprehensive validation and normalization of document data:
        1. Validates the presence of required fields
        2. Handles both text and image content types
        3. Validates content against model capabilities (e.g., image support)
        4. Maps document fields to the database schema
        
        Args:
            document_row: Input document dictionary containing content and metadata
            
        Returns:
            Dictionary with validated insertion data, original content, and document ID
            
        Raises:
            ValueError: If required fields are missing or the content type is unsupported
        """
        content = document_row.get('content')
        if not content:
            raise ValueError("Document row must contain nonempty 'content' field")
        
        doc_id = document_row.get('doc_id', uuid.uuid4())
        insertion = {"doc_id": doc_id}
        
        if isinstance(content, Image):
            if not self.model.supports_images:
                raise ValueError(f"Model {self.model.__class__.__name__} does not support image inputs")
                
            image_url = document_row.get('image_url')
            if not image_url:
                logger.warning(f"Image URL not provided for image document id {doc_id}. Using placeholder.")
            insertion["content"] = image_url or "image_document"
            insertion["content_type"] = "image"
        else:
            insertion["content"] = content
            if self.model.supports_images:
                insertion["content_type"] = "text"

        table_definition = await self._doc_table.definition()
        valid_columns = {col for col in table_definition.columns}

        remaining_keys = valid_columns - set(insertion.keys())
        
        for key in remaining_keys:
            if key in document_row:
                insertion[key] = document_row[key]
            else:
                logger.warning(f"Field '{key}' is not defined in the table schema and will be ignored")
                
        return {
            "validated_insertion": insertion,
            "original_content": content,
            "doc_id": doc_id
        }
            
    
    async def index_document(
        self, 
        document_row: Dict[str, Any],
        **kwargs,
    ) -> Optional[uuid.UUID]:
        """
        Index a single document by storing its content and token embeddings.
        
        This method processes a document through the complete indexing pipeline:
        1. Validates and prepares the document data
        2. Stores document content in the document table
        3. Generates token-level embeddings using the late interaction model
        4. Applies optional document embedding pooling
        5. Stores token embeddings in the token table
        
        Args:
            document_row: Dictionary containing document content and metadata
            **kwargs: Additional keyword arguments passed to insert_one
            
        Returns:
            The document ID of the indexed document
            
        Raises:
            ValueError: If document validation fails
            Exception: If any step in the indexing process fails
        """
        if not self._initialized:
            await self.initialize()
        
        logger.debug("Validating document row")
        validated = await self._validate_row(document_row)
        doc_id = validated["doc_id"]
        content = validated["original_content"]
        insertion = validated["validated_insertion"]
        
        try:
            logger.debug(f"Inserting document {doc_id} into document table")
            await self._doc_table.insert_one(insertion, **kwargs)
            
            logger.debug(f"Encoding document {doc_id}")
            doc_embeddings = await self.model.encode_doc([content])

            if self.doc_pool_factor and self.doc_pool_factor > 1:
                logger.debug(f"Pooling document embeddings with factor {self.doc_pool_factor}")
                doc_embeddings = pool_doc_embeddings(doc_embeddings, self.doc_pool_factor)
            
            logger.debug(f"Indexing token embeddings for document {doc_id}")
            await self._index_token_embeddings(doc_id, doc_embeddings[0])
            
            logger.debug("Clearing document embeddings cache")
            self._cached_doc_embeddings.cache_clear()
            logger.info(f"Successfully indexed document {doc_id}")
            
            return doc_id
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise
    
    async def _index_token_embeddings(
        self, 
        doc_ids: Union[uuid.UUID, List[uuid.UUID]], 
        embeddings: Union[torch.Tensor, List[torch.Tensor]],
        db_concurrency: int = None,
        **kwargs,
    ) -> List[List[uuid.UUID]]:
        """
        Index token embeddings for one or more documents into the token table.
        
        This method handles the conversion and storage of token-level embeddings,
        performing validation and efficient batch insertion into the database.
        
        Args:
            doc_ids: Single document ID or list of document IDs
            embeddings: 2D tensor of token-wise embeddings or list of such tensors, 
                    corresponding to doc_ids. Each tensor has shape [num_tokens, embedding_dim].
            db_concurrency: Maximum number of concurrent database insertions
            **kwargs: Additional keyword arguments for database insertion
            
        Returns:
            List of lists of token IDs, one inner list per document
            
        Raises:
            ValueError: If input validation fails
            TypeError: If embeddings are not of the expected type
        """
        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]
        
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        
        if len(doc_ids) != len(embeddings):
            raise ValueError(f"Number of doc_ids ({len(doc_ids)}) must match number of embeddings ({len(embeddings)})")
        
        for i, embedding in enumerate(embeddings):
            if not isinstance(embedding, torch.Tensor):
                raise TypeError(f"Embedding at index {i} is not a torch.Tensor")
                
            if embedding.dim() != 2:
                raise ValueError(f"Embedding at index {i} must be a 2D tensor, got {embedding.dim()}D")
                
            if embedding.shape[1] != self.model.dim:
                raise ValueError(f"Embedding dimension mismatch at index {i}. "
                                f"Expected {self.model.dim}, got {embedding.shape[1]}")
        
        logger.debug(f"Indexing token embeddings for {len(doc_ids)} documents")
        all_token_ids = []
        all_insertions = []
        
        for _, (doc_id, embedding_tensor) in enumerate(zip(doc_ids, embeddings)):
            embeddings_np = self.model._embeddings_to_numpy(embedding_tensor)
            doc_token_ids = []
            
            logger.debug(f"Preparing {len(embeddings_np)} token embeddings for document {doc_id}")
            for _, token_embedding in enumerate(embeddings_np):
                token_id = uuid.uuid4()
                doc_token_ids.append(token_id)
                
                insertion = {
                    "doc_id": doc_id,
                    "token_id": token_id,
                    "token_embedding": token_embedding.tolist()
                }
                all_insertions.append(insertion)
            
            all_token_ids.append(doc_token_ids)
                
        logger.debug(f"Bulk inserting {len(all_insertions)} token embeddings with db_concurrency={db_concurrency}")
        try:
            result = await self._token_table.insert_many(
                all_insertions,
                ordered=False,
                concurrency=5,
                chunk_size=5,
                **kwargs,
            )
            logger.debug(f"Completed token embeddings indexing for {len(doc_ids)} documents")
        except Exception as e:
            logger.error(f"Error during token embeddings insertion: {str(e)}")
            raise
        
        return all_token_ids
    
    async def bulk_index_documents(
        self,
        document_rows: Iterable[Dict[str, Any]],
        embedding_concurrency: Optional[int] = None,
        batch_size: int = 10,
        concurrency: int = 10,
        **kwargs
    ) -> List[uuid.UUID]:
        """
        Index multiple documents in batches with optimized concurrency control.
        
        This method efficiently indexes large numbers of documents by:
        1. Processing documents in manageable batches
        2. Parallelizing embedding computation with controlled concurrency
        3. Optimizing database operations for the Data API
        
        Args:
            document_rows: Iterable of document dictionaries, each containing content and metadata
            embedding_concurrency: Maximum number of concurrent embedding operations
            batch_size: Number of documents to process in a single batch
            concurrency: Maximum number of concurrent database requests for the Data API
            **kwargs: Additional arguments to pass to insert_many (e.g., ordered, timeout_ms)
            
        Returns:
            List of successfully indexed document IDs
        """
        if not self._initialized:
            logger.debug("Initializing tables before bulk indexing")
            await self.initialize()

        embedding_concurrency = embedding_concurrency or self.default_concurrency_limit
        logger.debug(f"Using embedding concurrency: {embedding_concurrency}")
        
        document_rows_list = list(document_rows)
        logger.info(f"Bulk indexing {len(document_rows_list)} documents with batch_size={batch_size}, "
                f"embedding_concurrency={embedding_concurrency}, api_concurrency={concurrency}")
        
        all_doc_ids = []
        
        for i in range(0, len(document_rows_list), batch_size):
            batch_rows = document_rows_list[i:i+batch_size]
            batch_num = i//batch_size + 1
            logger.debug(f"Processing batch {batch_num} with {len(batch_rows)} documents")
            
            validated_rows = []
            batch_contents = []
            batch_doc_ids = []
            
            for row in batch_rows:
                validated = await self._validate_row(row)
                validated_rows.append(validated["validated_insertion"])
                batch_contents.append(validated["original_content"])
                batch_doc_ids.append(validated["doc_id"])
            
            logger.debug(f"Batch {batch_num}: Inserting {len(validated_rows)} documents")
            try:
                await self._doc_table.insert_many(
                    validated_rows, 
                    concurrency=concurrency,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Error inserting documents in batch {batch_num}: {str(e)}")
                raise
            
            logger.debug(f"Batch {batch_num}: Encoding document content")
            semaphore = asyncio.Semaphore(embedding_concurrency)
            
            async def encode_with_semaphore(content, doc_id):
                async with semaphore:
                    logger.debug(f"Encoding document {doc_id}")
                    doc_embeddings = await self.model.encode_doc([content])
                    
                    if self.doc_pool_factor and self.doc_pool_factor > 1:
                        logger.debug(f"Pooling document {doc_id} embeddings with factor {self.doc_pool_factor}")
                        doc_embeddings = pool_doc_embeddings(doc_embeddings, self.doc_pool_factor)
                    
                    await self._index_token_embeddings(doc_id, doc_embeddings[0], db_concurrency=concurrency)
                    return doc_id
            
            encoding_tasks = [
                encode_with_semaphore(content, doc_id) 
                for content, doc_id in zip(batch_contents, batch_doc_ids)
            ]
            
            try:
                completed_doc_ids = await asyncio.gather(*encoding_tasks)
                all_doc_ids.extend(completed_doc_ids)
                logger.debug(f"Completed batch {batch_num}")
            except Exception as e:
                logger.error(f"Error processing embeddings in batch {batch_num}: {str(e)}")
                raise
        
        logger.debug("Clearing document embeddings cache")
        self._cached_doc_embeddings.cache_clear()
        
        logger.info(f"Completed bulk indexing of {len(all_doc_ids)} documents")
        return all_doc_ids
    
    async def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a query string into token embeddings with optional pooling.
        
        This method converts a text query into token-level embeddings using
        the underlying late interaction model, and applies pooling based on
        the configured query_pool_distance parameter.
        
        Args:
            query: Query string to encode
            
        Returns:
            Query token embeddings as a PyTorch tensor
        """
        logger.debug(f"Encoding query: {query[:50]}{'...' if len(query) > 50 else ''}")
        query_embeddings = await self.model.encode_query(query)
        
        if self.query_pool_distance > 0:
            logger.debug(f"Pooling query embeddings with distance {self.query_pool_distance}")
            query_embeddings = pool_query_embeddings(
                query_embeddings, 
                self.query_pool_distance
            )
            
        return query_embeddings
    
    async def search(
        self, 
        query: str,
        k: int = 10,
        n_ann_tokens: Optional[int] = None,
        n_maxsim_candidates: Optional[int] = None,
        **kwargs,
    ) -> List[Tuple[uuid.UUID, float, str]]:
        """
        Perform a late interaction search using a two-stage retrieval process.

        Stage 1: ANN search finds candidate tokens by retrieving the most similar token 
                vectors for each query token.
        Stage 2: MaxSim calculation computes final relevance scores between the query and 
                candidate documents for ranking.

        Args:
            query: Query string to search for
            k: Number of top results to return (default: 10)
            n_ann_tokens: Number of tokens to retrieve for each query token in the ANN stage.
                If None, automatically scales based on k (e.g., ~171 for k=10, ~514 for k=100)
            n_maxsim_candidates: Number of document candidates to consider for MaxSim scoring.
                If None, automatically scales based on k (e.g., ~20 for k=10, ~119 for k=100)
            **kwargs: Additional parameters passed to the underlying search operations
        
        Returns:
            List of tuples containing (doc_id, score, content) for top k documents, 
            sorted by relevance score in descending order.
        
        TODO: Future enhancement - When metadata indexing is supported by the Data API:
        1. Implement efficient metadata filtering at query time
        2. Pass filter_condition to search operations
        3. Create a metadata index table for common query patterns
        4. Consider hybrid retrieval that combines vector and metadata filtering
        """

        import time
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
            
        if n_ann_tokens is None:
            n_ann_tokens = expand_parameter(k, 94.9, 11.0, -1.48)
        if n_maxsim_candidates is None:
            n_maxsim_candidates = expand_parameter(k, 8.82, 1.13, -0.00471)
        
        logger.info(f"Searching for query with k={k}, n_ann_tokens={n_ann_tokens}, "
                   f"n_maxsim_candidates={n_maxsim_candidates}")
            
        try:
            Q = await self.encode_query(query)
            Q_np = self.model._embeddings_to_numpy(Q)
            
            results = await self._search_with_embeddings(
                Q, 
                Q_np, 
                k, 
                n_ann_tokens, 
                n_maxsim_candidates, 
                **kwargs,
            )
            
            logger.info(f"Search completed. Found {len(results)} results.")
            end_time = time.time()
            logger.debug(f"Search completed in {end_time - start_time} seconds")
            return results
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    async def _maxsim_score(
        self,
        Q: torch.Tensor,
        doc_embeddings: List[torch.Tensor],
        k: int,
        candidates: List[uuid.UUID],
    ) -> List[Tuple[uuid.UUID, float, str]]:
        """
        Calculate MaxSim scores between query and document embeddings and return ranked results.
        
        This method is responsible for the second stage of retrieval, computing
        precise relevance scores between the query and candidate documents using
        the late interaction model's scoring function.
        
        Args:
            Q: Query token embeddings tensor
            doc_embeddings: List of document token embeddings tensors
            k: Number of top results to return
            candidates: List of document ID candidates to score
            
        Returns:
            List of tuples with (doc_id, score, content) for the top k documents
        """

        start_time = time.time()
        logger.debug(f"Calculating MaxSim scores for {len(doc_embeddings)} documents")
        scores: torch.Tensor = self.model.score(Q, doc_embeddings)
            
        logger.debug("Converting scores and sorting results")
        score_items = [(doc_id, score.item()) for doc_id, score in zip(candidates, scores)]
        top_k = sorted(score_items, key=lambda x: x[1], reverse=True)[:k]
            
        logger.debug(f"Fetching content for {len(top_k)} top documents")
        doc_ids_str = [doc_id for doc_id, _ in top_k]
        cursor = await self.async_find(
            self._doc_table,
            filter={"doc_id": {"$in": doc_ids_str}}
        )
        docs = await cursor.to_list()
            
        logger.debug(f"Building final result set with {len(docs)} documents")
        doc_map = {doc["doc_id"]: doc for doc in docs}
                
        results = [(doc_id, score, doc_map.get(doc_id, {}).get("content", "")) 
                    for doc_id, score in top_k]
            
        end_time = time.time()
        logger.debug(f"MaxSim scoring completed in {end_time - start_time} seconds")
        return results
    
    async def _execute_token_level_searches(
        self,
        Q_np: np.ndarray,
        n_ann_tokens: int,
        **kwargs,
    ) -> List[List[Dict[str, Any]]]:
        """
        Execute token-level searches for each query token.
        
        Performs parallel vector searches for each query token embedding
        to find the most similar token embeddings in the database.
        
        Args:
            Q_np: Query token embeddings as NumPy array
            n_ann_tokens: Number of tokens to retrieve for each query token
            **kwargs: Additional search parameters passed to the database query
            
        Returns:
            List of lists containing token search results for each query token
        """
        start_time = time.time()
        required_projection: Dict[str, bool] = {"token_embedding": True, "doc_id": True}
        user_projection: Dict[str, bool] = kwargs.pop("projection", {})
        merged_projection: Dict[str, bool] = {**user_projection, **required_projection}
        
        token_search_tasks = [
            self.async_find(
                self._token_table,
                sort={
                    "token_embedding": token_embedding.tolist()
                },
                limit=n_ann_tokens,
                include_similarity=True,
                projection=merged_projection,
                **kwargs,
            )
            for token_embedding in Q_np
        ]

        cursors: List[AsyncTableFindCursor] = await asyncio.gather(*token_search_tasks)
        doc_token_results_list: List[List[Dict[str, Any]]] = (
            await asyncio.gather(*[cursor.to_list() for cursor in cursors])
        )
        end_time = time.time()
        logger.debug(f"Token-level search completed in {end_time - start_time} seconds")

        return doc_token_results_list

    async def _aggregate_token_level_results(
        self,
        doc_token_results_list: List[List[Dict[str, Any]]],
        n_maxsim_candidates: int,
    ) -> List[uuid.UUID]:
        """
        Aggregate token-level results into a ranked list of document candidates.
        
        This method processes the ANN search results for all query tokens and
        aggregates the similarity scores to identify the most promising document
        candidates for the MaxSim scoring phase.
        
        Args:
            doc_token_results_list: List of token search results for each query token
            n_maxsim_candidates: Number of top document candidates to select
            
        Returns:
            List of document IDs for the top n_maxsim_candidates documents
        """
        start_time = time.time()
        doc_scores, doc_token_scores = Counter(), {}

        logger.debug(f"Selecting top {n_maxsim_candidates} candidates for MaxSim scoring")


        for query_token_idx, doc_token_results in enumerate(doc_token_results_list):
            for result in doc_token_results:
                if not (doc_id := result.get("doc_id", "")):
                    continue

                key = (doc_id, query_token_idx)
                similarity = result.get("$similarity", 0)
                 
                if key in doc_token_scores:
                    if similarity > doc_token_scores[key]:
                        doc_scores[doc_id] += (similarity - doc_token_scores[key])
                        doc_token_scores[key] = similarity
                else:
                    doc_token_scores[key] = similarity
                    doc_scores[doc_id] += similarity

        logger.debug(f"Aggregated scores for {len(doc_scores)} unique documents")
        end_time = time.time()
        logger.debug(f"Token-level aggregation completed in {end_time - start_time} seconds")

        return heapq.nlargest(
                    n_maxsim_candidates, 
                    doc_scores.keys(),
                    key=lambda d: doc_scores[d]
                )

    async def _search_with_embeddings(
        self,
        Q: torch.Tensor,
        Q_np: np.ndarray,
        k: int,
        n_ann_tokens: int,
        n_maxsim_candidates: int,
        **kwargs,
    ) -> List[Tuple[uuid.UUID, float, str]]:
        """
        Perform a late interaction search with pre-computed query embeddings.
        
        This internal method implements the core search functionality using
        pre-computed embeddings, executing the two-stage retrieval process.
        
        Args:
            Q: Query token embeddings as PyTorch tensor
            Q_np: Query token embeddings as NumPy array
            k: Number of top results to return
            n_ann_tokens: Number of tokens to retrieve for each query token
            n_maxsim_candidates: Number of document candidates for MaxSim scoring
            **kwargs: Additional keyword arguments passed to the underlying search operations
            
        Returns:
            List of tuples with (doc_id, score, content) for top k documents
        """

        doc_token_results_list: List[List[Dict[str, Any]]] = (
            await self._execute_token_level_searches(Q_np, n_ann_tokens, **kwargs)
        )

        logger.debug(f"ANN search returned {len(doc_token_results_list) * n_ann_tokens}"
                     f"results across {len(doc_token_results_list)} query tokens")

        candidates: List[uuid.UUID] = await self._aggregate_token_level_results(
            doc_token_results_list,
            n_maxsim_candidates,
        )

        if not candidates:
            logger.info("No candidates found in first-stage retrieval")
            return []
            
        logger.debug(f"Selected {len(candidates)} candidates for MaxSim scoring")

        candidates_tuple: Tuple[str] = tuple(str(c) for c in candidates)

        doc_embeddings: List[torch.Tensor] = await self._cached_doc_embeddings(candidates_tuple)
        
        if doc_embeddings:
            return await self._maxsim_score(Q, doc_embeddings, k, candidates)
            
        logger.info("No document embeddings retrieved")
        return []
    
    @alru_cache(maxsize=1000)
    async def _cached_doc_embeddings(
        self, 
        doc_ids: Tuple[uuid.UUID]
    ) -> List[torch.Tensor]:
        """
        Load document token embeddings with caching for improved performance.
        
        This method caches document token embeddings to reduce database load
        and improve search performance for frequently accessed documents.
        
        Args:
            doc_ids: Tuple of document ID strings (tuple for hashability)
            
        Returns:
            List of token embedding tensors, one per document
        """

        return await self._load_doc_token_embeddings(doc_ids)

    async def _load_doc_token_embeddings(
        self, 
        doc_ids: Union[Tuple[uuid.UUID], List[uuid.UUID]]
    ) -> List[torch.Tensor]:
        """
        Load token embeddings for the specified documents in parallel.
        
        This method concurrently retrieves token embeddings for multiple documents,
        optimizing database access patterns with semaphore-controlled concurrency.
        
        Args:
            doc_ids: List or tuple of document IDs
            
        Returns:
            List of token embedding tensors, one per document
        """
        logger.debug(f"Loading token embeddings for {len(doc_ids)} documents")
        
        logger.debug(f"Creating parallel fetch tasks for {len(doc_ids)} documents")

        semaphore = asyncio.Semaphore(5)
        async def fetch_with_semaphore(doc_id: uuid.UUID) -> torch.Tensor:
            async with semaphore:
                return await self._fetch_token_embeddings(doc_id)

        start_time = time.time()
        tasks = [
            fetch_with_semaphore(doc_id) 
            for doc_id in doc_ids
        ]
        
        logger.debug(f"Executing {len(tasks)} token embedding fetch tasks concurrently")
        embeddings: List[torch.Tensor] = await asyncio.gather(*tasks)
        end_time = time.time()
        logger.debug(f"Token embedding fetch completed in {end_time - start_time} seconds")
        return embeddings

    async def _fetch_token_embeddings(
        self, 
        doc_id: uuid.UUID
    ) -> torch.Tensor:
        """
        Fetch token embeddings for a single document from the database.
        
        Retrieves all token embeddings for a specific document and
        converts them to the appropriate tensor format for the model.
        
        Args:
            doc_id: Document ID to fetch embeddings for
            
        Returns:
            Token embeddings tensor for the document, or an empty tensor if no embeddings found
        """
        logger.debug(f"Fetching token embeddings for document {doc_id}")
        cursor = await self.async_find(
            self._token_table,
            filter={"doc_id": doc_id},
            projection={"token_embedding": True, "token_id": True}
        )
        tokens = await cursor.to_list()
        
        if not tokens:
            logger.warning(f"No token embeddings found for document {doc_id}")
            logger.debug(f"Returning empty tensor for document {doc_id}")
            return torch.zeros((0, self.model.dim), 
                            device=self.model.to_device(torch.tensor([])).device)
            
        try:
            logger.debug(f"Processing {len(tokens)} token embeddings for document {doc_id}")
            embeddings = [token["token_embedding"] for token in tokens]
            logger.debug(f"Converting embeddings to tensor for document {doc_id}")
            return self.model._numpy_to_embeddings(np.array(embeddings))
        except Exception as e:
            logger.error(f"Error processing token embeddings for document {doc_id}: {str(e)}")
            logger.debug(f"Returning empty tensor due to error for document {doc_id}")
            return torch.zeros((0, self.model.dim), 
                            device=self.model.to_device(torch.tensor([])).device)

    async def delete_document(self, doc_id: uuid.UUID) -> Optional[bool]:
        """
        Delete a document and all its associated token embeddings.
        
        This method ensures complete cleanup of document data by:
        1. Removing the document content from the document table
        2. Removing all token embeddings from the token table
        3. Clearing any cached embeddings to maintain consistency
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if document was deleted successfully
            
        Raises:
            Exception: If an error occurs during deletion
        """
        if not self._initialized:
            logger.debug("Initializing before document deletion")
            await self.initialize()
            
        logger.info(f"Deleting document {doc_id} and its token embeddings")
        try:
            logger.debug(f"Deleting document record for {doc_id}")
            await self._doc_table.delete_many(
                filter={"doc_id": str(doc_id)}
            )
            
            logger.debug(f"Deleting token records for document {doc_id}")
            await self._token_table.delete_many(
                filter={"doc_id": doc_id}
            )
                
            logger.debug("Clearing document embeddings cache")
            self._cached_doc_embeddings.cache_clear()
            
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            raise