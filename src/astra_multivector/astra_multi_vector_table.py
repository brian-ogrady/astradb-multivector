import uuid
from typing import List, Dict, Any

from astrapy import Database, Table
from astrapy.info import (
    AlterTableAddVectorize,
    ColumnType,
    CreateTableDefinition,
)

from astra_multivector.table.vector_column_options import VectorColumnOptions


class AstraMultiVectorTable:
    """A class for storing and retrieving text chunks with vector embeddings.
    
    This class handles the creation of database tables with vector columns,
    creation of vector indexes, and the insertion of text chunks with their
    associated embeddings (either computed client-side or with vectorize).
    
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
                    "providerKey": "openaikey_astra_kms_alias",
                },
            )
        )
        
        # Create the table
        vector_table = GutenbergTextVectorTable(
            database=db,
            table_name="hamlet",
            english_options,
            multilingual_options
        )
        
        # Insert text chunks
        vector_table.insert_chunk("To be or not to be, that is the question.")
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

        vectorize_options = {
            options.column_name: options.vector_service_options 
            for options in self.vector_column_options if options.vector_service_options
        }

        if len(vectorize_options) > 0:
            table = table.alter(
                AlterTableAddVectorize(columns=vectorize_options)
            )
        
        return table
    
    def insert_chunk(self, text_chunk: str) -> None:
        """Insert a text chunk & embeddings(s) into the table"""
        chunk_id = uuid.uuid4()

        insertions = {"chunk_id": chunk_id, "content": text_chunk}
        for options in self.vector_column_options:
            if options.vector_service_options:
                insertions[options.column_name] = text_chunk
            else:
                insertions[options.column_name] = options.model.encode(text_chunk).tolist()
        
        self.table.insert_one(insertions)
        
    def bulk_insert_chunks(self, text_chunks: List[str], batch_size: int = 100) -> None:
        """Insert multiple text chunks in batches.
        
        Args:
            text_chunks: List of text chunks to insert
            batch_size: Number of documents to insert in a single batch operation
        """
        # Process text chunks in batches
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            
            batch_inserts = []
            for text_chunk in batch:
                chunk_id = uuid.uuid4()
                insertion = {"chunk_id": chunk_id, "content": text_chunk}
                
                for options in self.vector_column_options:
                    if options.vector_service_options:
                        insertion[options.column_name] = text_chunk
                    else:
                        insertion[options.column_name] = options.model.encode(text_chunk).tolist()
                        
                batch_inserts.append(insertion)
            
            # Insert the batch
            if batch_inserts:
                self.table.insert_many(batch_inserts)
    
    def search_by_text(
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
            query = options.model.encode(query_text).tolist()
            
        # Perform the vector search using sort with the vector column
        results = self.table.find(
            filter={},
            sort={vector_column: query},
            limit=limit
        )
        
        return results
        
    def batch_search_by_text(
        self,
        queries: List[str],
        vector_column: str = None,
        limit: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """Perform multiple text similarity searches.
        
        Args:
            queries: List of text queries to search for
            vector_column: The vector column to use for the search
            limit: Maximum number of results to return per query
            
        Returns:
            List of search results for each query
        """
        results = []
        for query in queries:
            query_results = self.search_by_text(
                query_text=query,
                vector_column=vector_column,
                limit=limit
            )
            results.append(query_results)
            
        return results
