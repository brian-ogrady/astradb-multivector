import asyncio
import uuid
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from functools import lru_cache

import numpy as np
import torch
from PIL.Image import Image
from astrapy import AsyncDatabase, AsyncTable
from astrapy.constants import VectorMetric
from astrapy.info import (
    ColumnType,
    CreateTableDefinition,
    TableVectorIndexOptions
)

from astra_multivector import AsyncAstraMultiVectorTable
from astra_multivector import VectorColumnOptions

from .models.base import LateInteractionModel
from .utils import (
    pool_query_embeddings,
    pool_doc_embeddings,
    expand_parameter,
)


class LateInteractionPipeline:
    """
    Pipeline for late interaction retrieval models in AstraDB.
    
    Handles the creation of document and token tables, indexing of documents,
    and multi-stage retrieval using late interaction models like ColBERT.
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
    
    async def initialize(self) -> None:
        """
        Initialize document and token tables.
        
        Creates the necessary tables and vector indexes if they don't exist.
        """
        async with self._init_lock:
            if not self._initialized:
                self._doc_table = await self._create_doc_table()
                self._token_table = await self._create_token_table()
                self._initialized = True
    
    async def _create_doc_table(self) -> AsyncTable:
        """
        Create the document table for storing document metadata.
        """
        schema = (
            CreateTableDefinition.builder()
            .add_column("doc_id", ColumnType.UUID)
            .add_column("content", ColumnType.TEXT)
            .add_partition_by(["doc_id"])
        )
        
        doc_table = await self.db.create_table(
            self.doc_table_name,
            definition=schema.build(),
            if_not_exists=True,
        )
        
        return doc_table
    
    async def _create_token_table(self) -> AsyncAstraMultiVectorTable:
        """
        Create the token table for storing token-level embeddings.
        """
        index_options = TableVectorIndexOptions(metric=self.sim_metric)
        
        token_options = VectorColumnOptions.from_precomputed_embeddings(
            column_name="token_embedding",
            dimension=self.model.dim,
            table_vector_index_options=index_options
        )
        
        token_table = AsyncAstraMultiVectorTable(
            db=self.db,
            table_name=self.token_table_name,
            vector_column_options=[token_options],
            default_concurrency_limit=self.default_concurrency_limit,
        )
        
        return token_table
    
    async def index_document(
        self, 
        content: Union[str, Image],
        doc_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        """
        Index a document by storing its content and token embeddings.
        
        Args:
            content: Document content (text or image)
            doc_id: Optional document ID (generated if not provided)
            
        Returns:
            The document ID
        """
        if not self._initialized:
            await self.initialize()
        
        if doc_id is None:
            doc_id = uuid.uuid4()
            
        
        await self._doc_table.insert_one({
            "doc_id": doc_id,
            "content": content if isinstance(content, str) else "image_document",
        })
        
        doc_embeddings = await self.model.encode_doc([content])
        
        if self.doc_pool_factor and self.doc_pool_factor > 1:
            doc_embeddings = pool_doc_embeddings(doc_embeddings, self.doc_pool_factor)
        
        await self._index_token_embeddings(doc_id, doc_embeddings[0])
        
        self._cached_doc_embeddings.cache_clear()
        
        return doc_id
    
    async def _index_token_embeddings(
        self, 
        doc_id: uuid.UUID, 
        embeddings: torch.Tensor
    ) -> List[uuid.UUID]:
        """
        Index token embeddings for a document.
        
        Args:
            doc_id: Document ID
            embeddings: Token embeddings tensor
            
        Returns:
            List of token IDs
        """
        embeddings_np = self.model._embeddings_to_numpy(embeddings)
        
        tasks = []
        token_ids = []
        
        for token_idx, token_embedding in enumerate(embeddings_np):
            token_id = uuid.uuid4()
            token_ids.append(token_id)
            
            insertion = {
                "chunk_id": token_id,
                "content": f"{doc_id}:{token_idx}",
                "token_embedding": token_embedding.tolist()
            }
            
            task = self._token_table.table.insert_one(insertion)
            tasks.append(task)
        
        # Execute all insertions concurrently
        await asyncio.gather(*tasks)
        
        return token_ids
    
    async def bulk_index_documents(
        self,
        documents: List[Union[str, Image]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        doc_ids: Optional[List[uuid.UUID]] = None,
        max_concurrency: Optional[int] = None,
        batch_size: int = 10,
    ) -> List[uuid.UUID]:
        """
        Index multiple documents in batches with concurrency control.
        
        Args:
            documents: List of document contents (text or images)
            metadata_list: Optional list of metadata for each document
            doc_ids: Optional list of document IDs (generated if not provided)
            max_concurrency: Maximum number of concurrent indexing operations
            batch_size: Number of documents to process in a single batch
            
        Returns:
            List of document IDs
        """
        if not self._initialized:
            await self.initialize()
            
        if max_concurrency is None:
            max_concurrency = self.default_concurrency_limit
            
        # Generate document IDs if not provided
        if doc_ids is None:
            doc_ids = [uuid.uuid4() for _ in range(len(documents))]
            
        # Use empty dictionaries instead of None for metadata
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(documents))]
            
        # Process documents in batches
        all_doc_ids = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metadata = metadata_list[i:i+batch_size]
            batch_ids = doc_ids[i:i+batch_size]
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def index_with_semaphore(doc, meta, doc_id):
                async with semaphore:
                    return await self.index_document(doc, meta, doc_id)
            
            # Create tasks for all documents in batch
            tasks = [
                index_with_semaphore(doc, meta, doc_id) 
                for doc, meta, doc_id in zip(batch_docs, batch_metadata, batch_ids)
            ]
            
            # Execute tasks concurrently and collect results
            batch_results = await asyncio.gather(*tasks)
            all_doc_ids.extend(batch_results)
            
        return all_doc_ids
    
    async def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a query string into token embeddings.
        
        Applies query pooling if enabled.
        
        Args:
            query: Query string
            
        Returns:
            Query token embeddings tensor
        """
        # Encode query
        query_embeddings = await self.model.encode_query(query)
        
        # Apply pooling if enabled
        if self.query_pool_distance > 0:
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
        filter_condition: Optional[dict] = None,
    ) -> List[Tuple[uuid.UUID, float, str]]:
        """
        Perform a late interaction search.
        
        Uses a two-stage retrieval process:
        1. ANN search to find candidate tokens
        2. MaxSim calculation for final ranking
        
        Args:
            query: Query string
            k: Number of top results to return
            n_ann_tokens: Number of tokens to retrieve for each query token
            n_maxsim_candidates: Number of document candidates for MaxSim scoring
            filter_condition: Optional filter condition for the document search
            
        Returns:
            List of tuples with (doc_id, score, content) for top k documents
        """
        if not self._initialized:
            await self.initialize()
            
        # Compute dynamic parameters if not provided
        if n_ann_tokens is None:
            # f(1) = 105, f(10) = 171, f(100) = 514, f(500) = 998
            n_ann_tokens = expand_parameter(k, 94.9, 11.0, -1.48)
        if n_maxsim_candidates is None:
            # f(1) = 9, f(10) = 20, f(100) = 119, f(900) = 1000
            n_maxsim_candidates = expand_parameter(k, 8.82, 1.13, -0.00471)
            
        # Encode query
        Q = await self.encode_query(query)
        Q_np = self.model._embeddings_to_numpy(Q)
        
        # Perform ANN search for each query token
        return await self._search_with_embeddings(
            Q, 
            Q_np, 
            k, 
            n_ann_tokens, 
            n_maxsim_candidates, 
            filter_condition
        )
    
    async def _search_with_embeddings(
        self,
        Q: torch.Tensor,
        Q_np: np.ndarray,
        k: int,
        n_ann_tokens: int,
        n_maxsim_candidates: int,
        filter_condition: Optional[dict] = None,
    ) -> List[Tuple[uuid.UUID, float, str]]:
        """
        Perform a late interaction search with pre-computed query embeddings.
        
        Args:
            Q: Query token embeddings as PyTorch tensor
            Q_np: Query token embeddings as NumPy array
            k: Number of top results to return
            n_ann_tokens: Number of tokens to retrieve for each query token
            n_maxsim_candidates: Number of document candidates for MaxSim scoring
            filter_condition: Optional filter condition for the document search
            
        Returns:
            List of tuples with (doc_id, score, content) for top k documents
        """
        # Step 1: Run all ANN searches in parallel
        token_search_tasks = [
            self._token_table.search_by_text(
                query_text=token_embedding.tolist(),
                vector_column="token_embedding",
                limit=n_ann_tokens
            )
            for token_embedding in Q_np
        ]
        
        # Execute all searches concurrently
        token_results_list = await asyncio.gather(*token_search_tasks)
        
        # Step 2: Process results
        doc_token_scores = {}
        for token_idx, token_results in enumerate(token_results_list):
            # Process each token result
            for result in token_results:
                # Extract doc_id from content field (format: "doc_id:token_idx")
                doc_id_str, _ = result["content"].split(":")
                doc_id = uuid.UUID(doc_id_str)
                
                # Store highest similarity per query token per document
                key = (doc_id, token_idx)
                similarity = result.get("$similarity", 0)  # ANN similarity score
                doc_token_scores[key] = max(doc_token_scores.get(key, -1), similarity)
        
        # Step 3: Aggregate scores per document
        doc_scores = {}
        for (doc_id, _), similarity in doc_token_scores.items():
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + similarity
            
        # Step 4: Select top candidates for full MaxSim scoring
        candidates = sorted(
            doc_scores.keys(), 
            key=lambda d: doc_scores[d], 
            reverse=True
        )[:n_maxsim_candidates]
        
        if not candidates:
            return []  # No results found
            
        # Step 5: Use cached document token embeddings for candidates
        # Convert list to tuple for cache key
        candidates_tuple = tuple(candidates)
        doc_embeddings = await self._cached_doc_embeddings(candidates_tuple)
        
        # Step 6: Calculate full MaxSim scores
        if doc_embeddings:
            scores = self.model.score(Q, doc_embeddings)
            
            # Convert tensor scores to Python floats
            score_items = [(doc_id, score.item()) for doc_id, score in zip(candidates, scores)]
            
            # Sort by score and get top k
            top_k = sorted(score_items, key=lambda x: x[1], reverse=True)[:k]
            
            # Fetch document metadata
            cursor = await self._doc_table.find(
                filter={"doc_id": {"$in": [str(doc_id) for doc_id, _ in top_k]}}
            )
            docs = await cursor.to_list()
            
            # Create a mapping of doc_ids to docs
            doc_map = {uuid.UUID(doc["doc_id"]): doc for doc in docs}
                
            # Include content in results
            results = [(doc_id, score, doc_map.get(doc_id, {}).get("content", "")) for doc_id, score in top_k]
            return results
            
        return []
    
    @lru_cache(maxsize=1000)  # Decorator applied directly to the method
    async def _cached_doc_embeddings(
        self, 
        doc_ids: Tuple[uuid.UUID]
    ) -> List[torch.Tensor]:
        """
        Cached version of document token embeddings loading.
        
        Args:
            doc_ids: Tuple of document IDs (tuple for hashability)
            
        Returns:
            List of token embedding tensors, one per document
        """
        # Ensure doc_ids is a tuple for hashability
        doc_ids_tuple = tuple(doc_ids)  # Extra protection against non-hashable types
        
        # Use the parallelized implementation
        return await self._load_doc_token_embeddings(doc_ids_tuple)
    
    async def _load_doc_token_embeddings(
        self, 
        doc_ids: Tuple[uuid.UUID]
    ) -> List[torch.Tensor]:
        """
        Load token embeddings for the specified documents in parallel.
        
        Args:
            doc_ids: Tuple of document IDs
            
        Returns:
            List of token embedding tensors, one per document
        """
        # Convert tuple to list for processing
        doc_ids_list = list(doc_ids)
        
        # Create tasks to fetch embeddings for each document in parallel
        tasks = [
            self._fetch_token_embeddings(doc_id) 
            for doc_id in doc_ids_list
        ]
        
        # Execute all tasks concurrently
        return await asyncio.gather(*tasks)
    
    async def _fetch_token_embeddings(
        self, 
        doc_id: uuid.UUID
    ) -> torch.Tensor:
        """
        Fetch token embeddings for a single document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Token embeddings tensor for the document
        """
        # Query tokens for this document
        cursor = await self._token_table.table.find(
            filter={"content": {"$regex": f"^{doc_id}:"}},
            projection={"token_embedding": 1, "content": 1}
        )
        tokens = await cursor.to_list()
        
        if not tokens:
            # Return empty tensor if no tokens found
            return torch.zeros((0, self.model.dim), device=self.model.to_device(torch.tensor([])).device)
            
        # Extract token index and embeddings, then sort by index
        token_data = sorted(
            [(int(token["content"].split(":")[1]), token["token_embedding"]) 
             for token in tokens],
            key=lambda x: x[0]
        )
            
        # Convert to tensor
        embeddings = [t[1] for t in token_data]
        return self.model._numpy_to_embeddings(np.array(embeddings))
        
    async def get_document(self, doc_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        if not self._initialized:
            await self.initialize()
            
        # Query document
        cursor = await self._doc_table.find(
            filter={"doc_id": str(doc_id)},
            limit=1
        )
        docs = await cursor.to_list()
        
        if not docs:
            return None
            
        doc = docs[0]
        
        # Prepare result
        result = {
            "doc_id": uuid.UUID(doc["doc_id"]),
            "content": doc["content"]
        }
        
        return result
        
    async def delete_document(self, doc_id: uuid.UUID) -> bool:
        """
        Delete a document and its token embeddings.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        if not self._initialized:
            await self.initialize()
            
        # Delete document
        doc_result = await self._doc_table.delete(
            filter={"doc_id": str(doc_id)}
        )
        
        # Delete token embeddings
        token_cursor = await self._token_table.table.find(
            filter={"content": {"$regex": f"^{doc_id}:"}},
            projection={"chunk_id": 1}
        )
        tokens = await token_cursor.to_list()
        
        if tokens:
            # Create delete tasks for each token
            delete_tasks = []
            for token in tokens:
                task = self._token_table.table.delete(
                    filter={"chunk_id": token["chunk_id"]}
                )
                delete_tasks.append(task)
                
            # Execute all deletes concurrently
            await asyncio.gather(*delete_tasks)
            
        # Clear the cache since we've modified documents
        self._cached_doc_embeddings.cache_clear()
            
        return doc_result.deleted_count > 0
