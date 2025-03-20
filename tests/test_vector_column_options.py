import unittest
from unittest.mock import MagicMock, patch

from astrapy.constants import VectorMetric
from astrapy.info import TableVectorIndexOptions, VectorServiceOptions
from sentence_transformers import SentenceTransformer

from astra_multivector import VectorColumnOptions
from astra_multivector.vector_column_options import VectorColumnType


class TestVectorColumnOptions(unittest.TestCase):
    
    def test_from_sentence_transformer_with_default_name(self):
        # Mock SentenceTransformer
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768

        mock_card_data = MagicMock()
        mock_card_data.base_model = "test-model"
        mock_model.model_card_data = mock_card_data
        
        # Test with default column name
        options = VectorColumnOptions.from_sentence_transformer(model=mock_model)
        
        # Assertions
        self.assertEqual(options.column_name, "test_model")
        self.assertEqual(options.dimension, 768)
        self.assertEqual(options.model, mock_model)
        self.assertIsNone(options.vector_service_options)
        self.assertIsNone(options.table_vector_index_options)
        
        # Verify type is correctly set
        self.assertEqual(options.type, VectorColumnType.SENTENCE_TRANSFORMER)
    
    def test_from_sentence_transformer_with_custom_name(self):
        # Mock SentenceTransformer
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.model_card_data = MagicMock(base_model="test-model")
        
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
        
        # Verify type is correctly set
        self.assertEqual(options.type, VectorColumnType.SENTENCE_TRANSFORMER)
    
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
        
        # Verify type is correctly set
        self.assertEqual(options.type, VectorColumnType.VECTORIZE)
    
    def test_from_precomputed_embeddings(self):
        # Create index options
        index_options = TableVectorIndexOptions(metric=VectorMetric.DOT_PRODUCT)
        
        # Test creating options for precomputed embeddings
        options = VectorColumnOptions.from_precomputed_embeddings(
            column_name="precomputed_embeddings",
            dimension=512,
            table_vector_index_options=index_options
        )
        
        # Assertions
        self.assertEqual(options.column_name, "precomputed_embeddings")
        self.assertEqual(options.dimension, 512)
        self.assertIsNone(options.model)
        self.assertIsNone(options.vector_service_options)
        self.assertEqual(options.table_vector_index_options, index_options)
        
        # Verify type is correctly set
        self.assertEqual(options.type, VectorColumnType.PRECOMPUTED)
    
    def test_to_dict(self):
        # Create test options with all fields set
        # Mock SentenceTransformer
        mock_model = SentenceTransformer("all-MiniLM-L6-v2")  # Load a lightweight real model
        mock_model.model_name = "test-model"  # Manually override model_name

        # Create model_card_data attribute
        class FakeModelCard:
            base_model = "test-model"

        mock_model.model_card_data = FakeModelCard()
        
        index_options = TableVectorIndexOptions(metric=VectorMetric.COSINE)
        
        options = VectorColumnOptions.from_sentence_transformer(
            model=mock_model,
            column_name="embeddings",
            table_vector_index_options=index_options
        )
        
        # Call to_dict
        result = options.to_dict()
        

        # Assertions
        self.assertIsInstance(result, dict)
        self.assertEqual(result["column_name"], "embeddings")
        self.assertEqual(result["dimension"], 384)
        self.assertEqual(result["type"], "SENTENCE_TRANSFORMER")
        self.assertEqual(result["model"], "test-model")
        self.assertIsNone(result["vector_service_options"])
        self.assertIsInstance(result["table_vector_index_options"], dict)
        self.assertEqual(result["table_vector_index_options"]["metric"], "cosine")
        
        # Test with vectorize options
        vector_options = VectorServiceOptions(
            provider="openai",
            model_name="text-embedding-3-small",
            authentication={"providerKey": "test-key"}
        )
        
        vectorize_options = VectorColumnOptions.from_vectorize(
            column_name="vectorize_embeddings",
            dimension=1536,
            vector_service_options=vector_options
        )
        
        result = vectorize_options.to_dict()
        
        self.assertEqual(result["column_name"], "vectorize_embeddings")
        self.assertEqual(result["type"], "VECTORIZE")
        self.assertIsNone(result["model"])
        self.assertIsInstance(result["vector_service_options"], dict)
        self.assertEqual(result["vector_service_options"]["provider"], "openai")
        
        # Test with minimal options
        minimal_options = VectorColumnOptions.from_precomputed_embeddings(
            column_name="minimal",
            dimension=512
        )
        
        result = minimal_options.to_dict()
        
        self.assertEqual(result["column_name"], "minimal")
        self.assertEqual(result["type"], "PRECOMPUTED")
        self.assertIsNone(result["model"])
        self.assertIsNone(result["vector_service_options"])
        self.assertIsNone(result["table_vector_index_options"])
    
    def test_type_property(self):
        # Create options using different factory methods
        st_options = VectorColumnOptions.from_sentence_transformer(
            model=MagicMock(spec=SentenceTransformer, 
                          get_sentence_embedding_dimension=lambda: 768,
                          model_card_data=MagicMock(base_model="test"))
        )
        
        vector_options = VectorColumnOptions.from_vectorize(
            column_name="vectorize",
            dimension=1536,
            vector_service_options=VectorServiceOptions(
                provider="openai",
                model_name="text-embedding-3-small",
                authentication={"key": "test"}
            )
        )
        
        precomputed_options = VectorColumnOptions.from_precomputed_embeddings(
            column_name="precomputed",
            dimension=512
        )
        
        # Verify types are correctly set
        self.assertEqual(st_options.type, VectorColumnType.SENTENCE_TRANSFORMER)
        self.assertEqual(vector_options.type, VectorColumnType.VECTORIZE)
        self.assertEqual(precomputed_options.type, VectorColumnType.PRECOMPUTED)
        
        # Verify type is read-only
        with self.assertRaises(AttributeError):
            st_options.type = VectorColumnType.PRECOMPUTED
    
    def test_validation_errors(self):
        # Test missing required parameters
        with self.assertRaises(TypeError):
            # Missing column_name
            VectorColumnOptions.from_precomputed_embeddings(dimension=512)
        
        with self.assertRaises(TypeError):
            # Missing dimension
            VectorColumnOptions.from_precomputed_embeddings(column_name="test")
        
        with self.assertRaises(TypeError):
            # Missing model
            VectorColumnOptions.from_sentence_transformer(column_name="test")
        
        with self.assertRaises(TypeError):
            # Missing vector_service_options
            VectorColumnOptions.from_vectorize(
                column_name="test",
                dimension=512
            )
        
        # Test invalid parameter types
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.model_card_data = MagicMock(base_model="test-model")
        
        with self.assertRaises(ValueError):
            # Invalid table_vector_index_options
            VectorColumnOptions.from_sentence_transformer(
                model=mock_model,
                table_vector_index_options="invalid"  # Should be TableVectorIndexOptions
            )
    
    def test_model_config(self):
        # Verify that arbitrary_types_allowed is set to True
        self.assertTrue(VectorColumnOptions.model_config.get("arbitrary_types_allowed"))


if __name__ == "__main__":
    unittest.main()