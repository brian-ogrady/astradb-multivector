import re
from enum import Enum, auto
from typing import Optional

from astrapy.info import (
    TableVectorIndexOptions,
    VectorServiceOptions,
)
from pydantic import BaseModel, PrivateAttr
from sentence_transformers import SentenceTransformer


class VectorColumnType(Enum):
    SENTENCE_TRANSFORMER = auto()
    VECTORIZE = auto()  
    PRECOMPUTED = auto()


class VectorColumnOptions(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    column_name: str
    dimension: int
    model: Optional[SentenceTransformer] = None
    vector_service_options: Optional[VectorServiceOptions] = None
    table_vector_index_options: Optional[TableVectorIndexOptions] = None
    _type: VectorColumnType = PrivateAttr(default=VectorColumnType.PRECOMPUTED)

    def __init__(self, *, _type: VectorColumnType, **kwargs):
        super().__init__(**kwargs)
        self._type = _type

    @property
    def type(self) -> VectorColumnType:
        """Read-only property to get the type"""
        return self._type

    @classmethod
    def from_precomputed_embeddings(
        cls,
        column_name: str,
        dimension: int,
        table_vector_index_options: Optional[TableVectorIndexOptions] = None,
    ) -> "VectorColumnOptions":
        """Create options for pre-computed embeddings where neither
        SentenceTransformer nor Vectorize will be used for embedding generation.
    
        When using these options, embeddings must be provided directly during insertion.
        """
        instance = cls(
            column_name=column_name,
            dimension=dimension,
            table_vector_index_options=table_vector_index_options,
            _type=VectorColumnType.PRECOMPUTED,
        )
        return instance

    @classmethod
    def from_sentence_transformer(
        cls, 
        model: SentenceTransformer, 
        column_name: Optional[str] = None,
        table_vector_index_options: Optional[TableVectorIndexOptions] = None,
    ) -> "VectorColumnOptions":
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
        base_model_name = re.sub(r'[^\w\d]', '_', getattr(model.model_card_data, "base_model", "default_model"), flags=re.UNICODE)
        column_name = column_name or base_model_name
        
        instance = cls(
            column_name=column_name,
            dimension=model.get_sentence_embedding_dimension(),
            model=model,
            table_vector_index_options=table_vector_index_options,
            _type=VectorColumnType.SENTENCE_TRANSFORMER,
        )
        return instance
    
    @classmethod
    def from_vectorize(
        cls,
        column_name: str,
        dimension: int,
        vector_service_options: VectorServiceOptions,
        table_vector_index_options: Optional[TableVectorIndexOptions] = None,
    ) -> "VectorColumnOptions":
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
        instance = cls(
            column_name=column_name,
            dimension=dimension,
            vector_service_options=vector_service_options,
            table_vector_index_options=table_vector_index_options,
            _type=VectorColumnType.VECTORIZE,
        )
        return instance
    
    def to_dict(self) -> dict:
        return {
            "column_name": self.column_name,
            "dimension": self.dimension,
            "type": self._type.name,
            "model": self.model.model_name if self.model else None,
            "vector_service_options": self.vector_service_options.as_dict() if self.vector_service_options else None,
            "table_vector_index_options": self.table_vector_index_options.as_dict() if self.table_vector_index_options else None,
        }

