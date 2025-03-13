import unittest
from unittest.mock import MagicMock, patch

from astrapy.constants import VectorMetric
from astrapy.info import TableVectorIndexOptions, VectorServiceOptions
from sentence_transformers import SentenceTransformer

from astra_multivector import VectorColumnOptions


class TestVectorColumnOptions(unittest.TestCase):
    
    def test_from_sentence_transformer_with_default_name(self):
        # Mock SentenceTransformer
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.model_card_data.base_model = "test-model"
        
        # Test with default column name
        options = VectorColumnOptions.from_sentence_transformer(model=mock_model)
        
        # Assertions
        self.assertEqual(options.column_name, "test_model")
        self.assertEqual(options.dimension, 768)
        self.assertEqual(options.model, mock_model)
        self.assertIsNone(options.vector_service_options)
        self.assertIsNone(options.table_vector_index_options)
    
    def test_from_sentence_transformer_with_custom_name(self):
        # Mock SentenceTransformer
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        
        # Create custom index options
        index_options = TableVectorIndexOptions(metric=VectorMetric.COSINE)
        
        # Test with custom column name and index options
        options = VectorColumnOptions.from_sentence_transformer(
            model=mock_model,
            column_name="custom_embeddings",
            table_vector_index_options=index_options
        )
        
        # Assertions
        self.assertEqual(options.column_name, "custom_embeddings")
        self.assertEqual(options.dimension, 768)
        self.assertEqual(options.model, mock_model)
        self.assertIsNone(options.vector_service_options)
        self.assertEqual(options.table_vector_index_options, index_options)
    
    def test_from_vectorize(self):
        # Create vectorize options
        vector_options = VectorServiceOptions(
            provider="openai",
            model_name="text-embedding-3-small",
            authentication={"providerKey": "test-key"}
        )
        
        # Create index options
        index_options = TableVectorIndexOptions(metric=VectorMetric.COSINE)
        
        # Test creating options for Vectorize
        options = VectorColumnOptions.from_vectorize(
            column_name="openai_embeddings",
            dimension=1536,
            vector_service_options=vector_options,
            table_vector_index_options=index_options
        )
        
        # Assertions
        self.assertEqual(options.column_name, "openai_embeddings")
        self.assertEqual(options.dimension, 1536)
        self.assertIsNone(options.model)
        self.assertEqual(options.vector_service_options, vector_options)
        self.assertEqual(options.table_vector_index_options, index_options)
    
    def test_model_config(self):
        # Verify that arbitrary_types_allowed is set to True
        self.assertTrue(VectorColumnOptions.model_config.get("arbitrary_types_allowed"))


if __name__ == "__main__":
    unittest.main()