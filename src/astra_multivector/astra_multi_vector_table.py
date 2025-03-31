import uuid
import warnings
from typing import Any, Dict, List, Optional, Union

from astrapy import Database, Table
from astrapy.info import (
    AlterTableAddVectorize,
    ColumnType,
    CreateTableDefinition,
)
from astrapy.results import TableInsertManyResult, TableInsertOneResult
from rerankers import Reranker
from rerankers.results import RankedResults

from astra_multivector.vector_column_options import (
    VectorColumnOptions,
    VectorColumnType,
)


class AstraMultiVectorTable:
    """A class for storing and retrieving text chunks with vector embeddings.
    
    This class handles the creation of database tables with vector columns,
    creation of vector indexes, and the insertion of text chunks with their
    associated embeddings. It supports multiple vector columns for the same content,
    allowing for:
    
    - Client-side embeddings using sentence-transformers models
    - Server-side embeddings using Astra Vectorize
    - Pre-computed embeddings supplied by the user
    
    The table supports advanced search capabilities including:
    - Basic vector similarity search per column
    - Multi-vector search across all columns with combined results
    - Integration with reranking models for improved accuracy
    - Batch operations for efficiency
    
    Example:
        ```python
        # Setup database connection
        from astrapy import Database
        db = Database(token="your-token", api_endpoint="your-endpoint")
        
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
        vector_table = AstraMultiVectorTable(
            db=db,
            table_name="hamlet",
            vector_column_options=[english_options, multilingual_options]
        )
        
        # Insert text chunks
        vector_table.insert_chunk("To be or not to be, that is the question.")
        
        # Search across all vector columns
        results = vector_table.multi_vector_similarity_search(
            query_text="question about existence",
            candidates_per_column=10
        )
        
        # Add reranking for improved relevance
        from reranker import Reranker
        reranker = Reranker.from_pretrained("BAAI/bge-reranker-base")
        ranked_results = vector_table.search_and_rerank(
            query_text="question about existence",
            reranker=reranker
        )
        ```
    """

    def __init__(
        self,
        db: Database,
        table_name: str,
        vector_column_options: List[VectorColumnOptions],
    ) -> None:
        self.db = db
        self.name = table_name
        self.vector_column_options = vector_column_options
        self.table = self._create_table()

    def _create_table(self) -> Table:
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
        
        table = self.db.create_table(
            self.name,
            definition=schema.build(),
            if_not_exists=True,
        )

        for options in self.vector_column_options:
            table.create_vector_index(
                f"{self.name}_{options.column_name}_idx",
                column=options.column_name,
                options=options.table_vector_index_options,
                if_not_exists=True,
            )
            if options.type == VectorColumnType.VECTORIZE:
                table = table.alter(
                    AlterTableAddVectorize(columns=options.vector_service_options)
                )

        return table
    
    def insert_chunk(
            self,
            text_chunk: str,
            precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
            **kwargs,
        ) -> TableInsertOneResult:
        """Insert a text chunk & embeddings(s) into the table
    
        Args:
            text_chunk: The text content to insert
            precomputed_embeddings: Dictionary mapping column names to precomputed 
                vector embeddings for columns of type PRECOMPUTED
            **kwargs: Additional arguments passed directly to the DataAPI's insert_one method
                
        Note:
            For complete details on all available parameters, see the AstraPy documentation:
            https://docs.datastax.com/en/astra-api-docs/_attachments/python-client/astrapy/index.html#astrapy.Table.insert_one
        """
        chunk_id = uuid.uuid4()
        precomputed_embeddings = precomputed_embeddings or {}

        insertion = {"chunk_id": chunk_id, "content": text_chunk}

        for options in self.vector_column_options:
            if options.type == VectorColumnType.VECTORIZE:
                insertion[options.column_name] = text_chunk
            elif options.type == VectorColumnType.SENTENCE_TRANSFORMER:
                insertion[options.column_name] = options.model.encode(text_chunk).tolist()
            elif options.type == VectorColumnType.PRECOMPUTED:
                if options.column_name not in precomputed_embeddings:
                    raise ValueError(
                        f"""Precomputed embeddings required for column 
                        '{options.column_name}' but not provided"""
                    )
                insertion[options.column_name] = precomputed_embeddings[options.column_name]
        
        return self.table.insert_one(insertion, **kwargs)
        
    def bulk_insert_chunks(
            self,
            text_chunks: List[str],
            precomputed_embeddings: Optional[Dict[str, List[List[float]]]] = None,
            batch_size: int = 100,
            **kwargs,
        ) -> List[TableInsertManyResult]:
        """Insert multiple text chunks in a single operation.
    
        Args:
            text_chunks: List of text chunks to insert
            precomputed_embeddings: Dictionary mapping column names to lists of embeddings,
                where each list corresponds to a text chunk at the same index
            batch_size: Number of text chunks to insert in each batch
            **kwargs: Additional arguments passed directly to the DataAPI's insert_many method.
                Common options include:
                - chunk_size: How many rows to include in each API request (default: system-optimized)
                - concurrency: Maximum number of concurrent requests (default: system-optimized)
                - ordered: If True, rows are processed sequentially, stopping on first error
                
        Note:
            For detailed information on all available options, refer to the AstraPy DataAPI 
            documentation for the insert_many method.
            https://docs.datastax.com/en/astra-api-docs/_attachments/python-client/astrapy/index.html#astrapy.Table.insert_many
        """
        if not text_chunks:
            return []
        
        precomputed_embeddings = precomputed_embeddings or {}

        results = []

        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            batch_inserts = []

            for batch_idx, text_chunk in enumerate(batch):
                chunk_id = uuid.uuid4()
                insertion = {"chunk_id": chunk_id, "content": text_chunk}

                global_idx = i + batch_idx
                
                for options in self.vector_column_options:
                    if options.type == VectorColumnType.VECTORIZE:
                        insertion[options.column_name] = text_chunk
                    elif options.type == VectorColumnType.SENTENCE_TRANSFORMER:
                        insertion[options.column_name] = options.model.encode(text_chunk).tolist()
                    elif options.type == VectorColumnType.PRECOMPUTED:
                        if options.column_name not in precomputed_embeddings:
                            raise ValueError(
                                f"Precomputed embeddings required for column '{options.column_name}' but not provided"
                            )
                        column_embeddings = precomputed_embeddings[options.column_name]
                        if global_idx >= len(column_embeddings):
                            raise ValueError(
                                f"Not enough precomputed embeddings provided for column '{options.column_name}'"
                            )
                        insertion[options.column_name] = column_embeddings[global_idx]
                            
                batch_inserts.append(insertion)
            
            if batch_inserts:
                result = self.table.insert_many(batch_inserts, **kwargs)
                results.append(result)

        return results
    
    def multi_vector_similarity_search(
        self, 
        query_text: str,
        vector_columns: Optional[List[str]] = None,
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
        candidates_per_column: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search across multiple vector columns and combine results.
        
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
        precomputed_embeddings = precomputed_embeddings or {}

        vector_columns = vector_columns or [opt.column_name for opt in self.vector_column_options]
        
        for col in vector_columns:
            if not any(opt.column_name == col for opt in self.vector_column_options):
                raise ValueError(f"Vector column '{col}' not found")
        
        all_results = []
        doc_id_to_result = {}
        doc_id_to_columns = {}
        
        for col in vector_columns:
            options = next(opt for opt in self.vector_column_options if opt.column_name == col)
            
            if options.type == VectorColumnType.VECTORIZE:
                query = query_text
            elif options.type == VectorColumnType.SENTENCE_TRANSFORMER:
                embedding = options.model.encode(query_text)
                query = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            elif options.type == VectorColumnType.PRECOMPUTED:
                if col not in precomputed_embeddings:
                    raise ValueError(f"Precomputed embedding required for column '{col}'")
                query = precomputed_embeddings[col]

            kwargs['filter'] = kwargs.get('filter', {})
            
            col_results = self.table.find(
                sort={col: query},
                limit=candidates_per_column,
                include_similarity=True,
                **{k: v for k, v in kwargs.items() if k != 'limit'}
            ).to_list()
            
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
    
    def rerank_results(
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
            return RankedResults(query=query_text, results=[])
            
        texts = [doc["content"] for doc in results]
        doc_ids = [doc["chunk_id"] for doc in results]
        
        ranked_results = reranker.rank(query=query_text, docs=texts, doc_ids=doc_ids)
        
        if limit is not None and limit < len(ranked_results.results):
            ranked_results = RankedResults(
                query=ranked_results.query,
                results=ranked_results.results[:limit],
            )
            
        return ranked_results
    
    def search_and_rerank(
        self,
        query_text: str,
        reranker: Reranker,
        vector_columns: Optional[List[str]] = None,
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
        candidates_per_column: int = 10,
        rerank_limit: Optional[int] = None,
        **kwargs
    ) -> RankedResults:
        """Search across multiple vector columns and rerank results.
        
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
        
        search_results = self.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=vector_columns,
            precomputed_embeddings=precomputed_embeddings,
            candidates_per_column=candidates_per_column,
            **kwargs
        )
        
        reranked_results = self.rerank_results(
            query_text=query_text,
            results=search_results, 
            reranker=reranker,
            limit=rerank_limit
        )
        
        return reranked_results
        
    def batch_search_by_text(
        self,
        queries: List[str],
        vector_columns: Optional[List[str]] = None,
        precomputed_embeddings: Optional[List[Dict[str, List[float]]]] = None,
        candidates_per_column: int = 10,
        rerank: bool = True,
        reranker: Optional[Reranker] = None,
        rerank_limit: Optional[int] = None,
        **kwargs
    ) -> Union[List[List[Dict[str, Any]]], List[RankedResults]]:
        """Perform multiple text similarity searches across vector columns with optional reranking.
        
        Args:
            queries: List of text queries to search for
            vector_columns: List of vector columns to search. If None, uses all columns.
            precomputed_embeddings: Optional list of pre-computed query embeddings for 
                                   PRECOMPUTED columns, one dict per query
            candidates_per_column: Number of candidates to retrieve per column (default: 10)
            rerank: Whether to rerank the results (default: True)
            reranker: Reranker instance to use for reranking. Required if rerank=True.
            rerank_limit: Maximum number of results to return after reranking.
                         If None, returns all reranked results.
            **kwargs: Additional parameters passed to the underlying find method
            
        Returns:
            If rerank=False: List of search results for each query
            If rerank=True: List of RankedResults objects for each query
        """
        if rerank and not reranker:
            raise ValueError("reranker must be provided when rerank=True")
            
        if precomputed_embeddings and len(precomputed_embeddings) != len(queries):
            raise ValueError(
                f"Number of precomputed embeddings ({len(precomputed_embeddings)}) "
                f"must match number of queries ({len(queries)})"
            )
            
        results = []
        for i, query in enumerate(queries):
            query_embeddings = None
            if precomputed_embeddings and i < len(precomputed_embeddings):
                query_embeddings = precomputed_embeddings[i]
                
            query_results = self.multi_vector_similarity_search(
                query_text=query,
                vector_columns=vector_columns,
                precomputed_embeddings=query_embeddings,
                candidates_per_column=candidates_per_column,
                **kwargs
            )
            
            if rerank and reranker and query_results:
                ranked_results = self.rerank_results(
                    query_text=query,
                    results=query_results,
                    reranker=reranker,
                    limit=rerank_limit
                )
                results.append(ranked_results)
            else:
                results.append(query_results)
            
        return results
