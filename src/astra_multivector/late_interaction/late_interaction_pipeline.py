import asyncio
import uuid
import json
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

import numpy as np
import torch
from PIL.Image import Image
from astrapy import AsyncDatabase, AsyncTable
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
        sim_metric: str = "cosine",
        default_concurrency_limit: int = 10,
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
        """
        self.db = db
        self.model = model
        self.base_table_name = base_table_name
        self.doc_pool_factor = doc_pool_factor
        self.query_pool_distance = query_pool_distance
        self.sim_metric = sim_metric
        self.default_concurrency_limit = default_concurrency_limit
        
        # Table names
        self.doc_table_name = f"{base_table_name}_docs"
        self.token_table_name = f"{base_table_name}_tokens"
        
        # Initialize tables
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
                # Create document table
                self._doc_table = await self._create_doc_table()
                
                # Create token table for token-level embeddings
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
            .add_column("metadata", ColumnType.TEXT)
            .add_partition_by(["doc_id"])
        )
        
        # Create the table
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
        # Create vector column options with table-specific settings
        index_options = TableVectorIndexOptions(metric=self.sim_metric)
        
        # Since we're using custom models, we'll create a placeholder options
        # We won't be using the model from VectorColumnOptions, just the dimension
        token_options = VectorColumnOptions(
            column_name="token_embedding",
            dimension=self.model.dim,
            table_vector_index_options=index_options
        )
        
        # Create the token table using AsyncAstraMultiVectorTable
        token_table = AsyncAstraMultiVectorTable(
            db=self.db,
            table_name=self.token_table_name,
            vector_column_options=[token_options],
            default_concurrency_limit=self.default_concurrency_limit,
        )
        
        # Make sure the table is initialized
        await token_table._initialize()
        
        return token_table
    
    async def index_document(
        self, 
        content: Union[str, Image],
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        """
        Index a document by storing its content, metadata, and token embeddings.
        
        Args:
            content: Document content (text or image)
            metadata: Optional metadata for the document (will be stored as JSON)
            doc_id: Optional document ID (generated if not provided)
            
        Returns:
            The document ID
        """
        if not self._initialized:
            await self.initialize()
        
        # Generate document ID if not provided
        if doc_id is None:
            doc_id = uuid.uuid4()
            
        # Convert metadata to JSON string if provided
        metadata_json = None
        if metadata is not None:
            metadata_json = json.dumps(metadata)
        
        # Insert document into doc table
        await self._doc_table.insert_one({
            "doc_id": doc_id,
            "content": content if isinstance(content, str) else "image_document",
            "metadata": metadata_json
        })
        
        # Encode document content to token embeddings
        doc_embeddings = await self.model.encode_doc([content])
        
        # Apply pooling if enabled
        if self.doc_pool_factor and self.doc_pool_factor > 1:
            doc_embeddings = pool_doc_embeddings(doc_embeddings, self.doc_pool_factor)
        
        # Insert token embeddings into token table
        await self._index_token_embeddings(doc_id, doc_embeddings[0])
        
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
        # Convert embeddings to list of numpy arrays for storage
        embeddings_np = self.model._embeddings_to_numpy(embeddings)
        
        # Process each token embedding
        tasks = []
        token_ids = []
        
        for token_idx, token_embedding in enumerate(embeddings_np):
            token_id = uuid.uuid4()
            token_ids.append(token_id)
            
            # Prepare insertion
            insertion = {
                "chunk_id": token_id,
                "content": f"{doc_id}:{token_idx}",  # Store doc_id:token_idx as content
                "token_embedding": token_embedding.tolist()
            }
            
            # Create task for insertion
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
            
        # Use empty metadata if not provided
        if metadata_list is None:
            metadata_list = [None] * len(documents)
            
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
    ) -> List[Tuple[uuid.UUID, float, Optional[Dict[str, Any]]]]:
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
            List of tuples with (doc_id, score, metadata) for top k documents
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
    ) -> List[Tuple[uuid.UUID, float, Optional[Dict[str, Any]]]]:
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
            List of tuples with (doc_id, score, metadata) for top k documents
        """
        # Step 1: ANN search for each query token
        doc_token_scores = {}
        
        # Process each query token embedding
        token_search_tasks = []
        for token_idx, token_embedding in enumerate(Q_np):
            # Create task for ANN search
            task = self._token_table.search_by_text(
                query_text=token_embedding.tolist(),
                vector_column="token_embedding",
                limit=n_ann_tokens
            )
            token_search_tasks.append((token_idx, task))
            
        # Collect ANN search results
        for token_idx, task in token_search_tasks:
            token_results = await task
            
            # Process each token result
            for result in token_results:
                # Extract doc_id from content field (format: "doc_id:token_idx")
                doc_id_str, _ = result["content"].split(":")
                doc_id = uuid.UUID(doc_id_str)
                
                # Store highest similarity per query token per document
                key = (doc_id, token_idx)
                similarity = result.get("$similarity", 0)  # ANN similarity score
                doc_token_scores[key] = max(doc_token_scores.get(key, -1), similarity)
        
        # Step 2: Aggregate scores per document
        doc_scores = {}
        for (doc_id, _), similarity in doc_token_scores.items():
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + similarity
            
        # Step 3: Select top candidates for full MaxSim scoring
        candidates = sorted(
            doc_scores.keys(), 
            key=lambda d: doc_scores[d], 
            reverse=True
        )[:n_maxsim_candidates]
        
        if not candidates:
            return []  # No results found
            
        # Step 4: Load document token embeddings for candidates
        doc_embeddings = await self._load_doc_token_embeddings(candidates)
        
        # Step 5: Calculate full MaxSim scores
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
            
            # Create a mapping of doc_id to metadata
            metadata_map = {}
            for doc in docs:
                doc_id = uuid.UUID(doc["doc_id"])
                metadata_json = doc.get("metadata")
                metadata = json.loads(metadata_json) if metadata_json else None
                metadata_map[doc_id] = metadata
                
            # Include metadata in results
            results = [(doc_id, score, metadata_map.get(doc_id)) for doc_id, score in top_k]
            return results
            
        return []
    
    async def _load_doc_token_embeddings(
        self, 
        doc_ids: List[uuid.UUID]
    ) -> List[torch.Tensor]:
        """
        Load token embeddings for the specified documents.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            List of token embedding tensors, one per document
        """
        doc_embeddings = []
        
        # Fetch token embeddings for each document
        for doc_id in doc_ids:
            # Query tokens for this document
            cursor = await self._token_table.table.find(
                filter={"content": {"$regex": f"^{doc_id}:"}},
                projection={"token_embedding": 1, "content": 1}
            )
            tokens = await cursor.to_list()
            
            if not tokens:
                continue
                
            # Extract token index and embeddings
            token_data = []
            for token in tokens:
                _, token_idx_str = token["content"].split(":")
                token_idx = int(token_idx_str)
                embedding = token["token_embedding"]
                token_data.append((token_idx, embedding))
                
            # Sort by token index
            token_data.sort(key=lambda x: x[0])
            
            # Convert to tensor
            embeddings = [t[1] for t in token_data]
            embeddings_tensor = self.model._numpy_to_embeddings(np.array(embeddings))
            
            doc_embeddings.append(embeddings_tensor)
            
        return doc_embeddings
        
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
        
        # Parse metadata
        metadata = None
        if doc.get("metadata"):
            try:
                metadata = json.loads(doc["metadata"])
            except json.JSONDecodeError:
                metadata = doc["metadata"]
                
        # Prepare result
        result = {
            "doc_id": uuid.UUID(doc["doc_id"]),
            "content": doc["content"],
            "metadata": metadata
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
            
        return doc_result.deleted_count > 0
