from typing import Optional

from astrapy.info import (
    TableVectorIndexOptions,
    VectorServiceOptions,
)
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class VectorColumnOptions(BaseModel):
    column_name: str
    dimension: int
    model: Optional[SentenceTransformer] = None
    vector_service_options: Optional[VectorServiceOptions] = None
    table_vector_index_options: Optional[TableVectorIndexOptions] = None

    @classmethod
    def from_sentence_transformer(
        cls, 
        model: SentenceTransformer, 
        column_name: Optional[str] = None,
        table_vector_index_options: Optional[TableVectorIndexOptions] = None,
    ):
        """Create options for client-side embedding with SentenceTransformer
        Example:
            ```python
            # Using the default column name (derived from model name)
            model = SentenceTransformer('intfloat/e5-large-v2')
            options = VectorColumnOptions.from_sentence_transformer(model)
            
            # With custom column name and index options
            index_options = TableVectorIndexOptions(metric='dot_product')
            options = VectorColumnOptions.from_sentence_transformer(
                model=model,
                column_name="embedding",
                table_vector_index_options=index_options
            )
            ```
        """
        if column_name is None:
            column_name = model.model_name_or_path.replace("/", "_").replace("-", "_")
        
        return cls(
            column_name=column_name,
            dimension=model.get_sentence_embedding_dimension(),
            model=model,
            table_vector_index_options=table_vector_index_options
        )
    
    @classmethod
    def from_vectorize(
        cls,
        column_name: str,
        dimension: int,
        vector_service_options: VectorServiceOptions,
        table_vector_index_options: Optional[TableVectorIndexOptions] = None,
    ):
        """Create options for Vectorize
        Example:
            ```python
            # Creating options for OpenAI embeddings with Vectorize
            vector_options = VectorServiceOptions(
                    provider='openai',
                    model_name='text-embedding-3-small',
                    authentication={
                        "providerKey": "openaikey_astra_kms_alias",
                    },
            )
            index_options = TableVectorIndexOptions(metric='cosine')
            
            options = VectorColumnOptions.from_vectorize(
                column_name="openai_embeddings",
                dimension=1536,
                vector_service_options=vector_options,
                table_vector_index_options=index_options
            )
            ```
        """
        return cls(
            column_name=column_name,
            dimension=dimension,
            vector_service_options=vector_service_options,
            table_vector_index_options=table_vector_index_options
        )
