import uuid

from astrapy import Database, Table
from astrapy.info import (
    AlterTableAddVectorize,
    ColumnType,
    CreateTableDefinition,
)

from .vector_column_options import VectorColumnOptions


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
        database: Database,
        table_name: str,
        *vector_column_options: VectorColumnOptions,
    ) -> None:
        self.db = database
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
            schema.add_vector_column(
                options.column_name,
                dimension=options.model.get_sentence_embedding_dimension(),
            )
        
        table = self.db.create_table(
            self.name,
            definition=schema.build(),
            if_not_exists=False,
        )

        for options in self.vector_column_options:
            table.create_vector_index(
                f"{self.name}_{options.column_name}_idx",
                column=options.column_name,
                options=options.table_vector_index_options,
                if_not_exists=False,
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
                insertions[options.column_name] = options.model.encode(text_chunk)
        
        self.table.insert_one(insertions)
