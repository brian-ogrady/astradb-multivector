import unittest
from unittest.mock import MagicMock

from astrapy.constants import VectorMetric
from astrapy.info import TableVectorIndexOptions, VectorServiceOptions
from sentence_transformers import SentenceTransformer

from astra_multivector import VectorColumnOptions
from astra_multivector.vector_column_options import VectorColumnType


class TestVectorColumnOptions(unittest.TestCase):
    
    def test_from_sentence_transformer_with_default_name(self):
        """Test VectorColumnOptions.from_sentence_transformer with default column name.
        
        Verifies that:
        - Column name defaults to the model's base_model name with hyphens replaced by underscores
        - Dimension is correctly extracted from the model
        - Model is properly stored
        - Vector service options and table vector index options are None by default
        - Type is set to SENTENCE_TRANSFORMER
        """
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768

        mock_card_data = MagicMock()
        mock_card_data.base_model = "test-model"
        mock_model.model_card_data = mock_card_data
        
        options = VectorColumnOptions.from_sentence_transformer(model=mock_model)
        
        self.assertEqual(options.column_name, "test_model")
        self.assertEqual(options.dimension, 768)
        self.assertEqual(options.model, mock_model)
        self.assertIsNone(options.vector_service_options)
        self.assertIsNone(options.table_vector_index_options)
        self.assertEqual(options.type, VectorColumnType.SENTENCE_TRANSFORMER)
    
    def test_from_sentence_transformer_with_custom_name(self):
        """Test VectorColumnOptions.from_sentence_transformer with custom column name and index options.
        
        Verifies that:
        - Custom column name overrides the default
        - Custom table vector index options are properly used
        - Dimension is correctly extracted from the model
        - Vector service options default to None
        - Type is set to SENTENCE_TRANSFORMER
        """
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.model_card_data = MagicMock(base_model="test-model")
        
        index_options = TableVectorIndexOptions(metric=VectorMetric.COSINE)
        
        options = VectorColumnOptions.from_sentence_transformer(
            model=mock_model,
            column_name="custom_embeddings",
            table_vector_index_options=index_options
        )
        
        self.assertEqual(options.column_name, "custom_embeddings")
        self.assertEqual(options.dimension, 768)
        self.assertEqual(options.model, mock_model)
        self.assertIsNone(options.vector_service_options)
        self.assertEqual(options.table_vector_index_options, index_options)
        self.assertEqual(options.type, VectorColumnType.SENTENCE_TRANSFORMER)
    
    def test_from_vectorize(self):
        """Test VectorColumnOptions.from_vectorize factory method.
        
        Verifies that:
        - Column name is set correctly
        - Dimension is set correctly
        - Model is None
        - Vector service options are properly stored
        - Table vector index options are properly stored
        - Type is set to VECTORIZE
        """
        vector_options = VectorServiceOptions(
            provider="openai",
            model_name="text-embedding-3-small",
            authentication={"providerKey": "test-key"}
        )
        
        index_options = TableVectorIndexOptions(metric=VectorMetric.COSINE)
        
        options = VectorColumnOptions.from_vectorize(
            column_name="openai_embeddings",
            dimension=1536,
            vector_service_options=vector_options,
            table_vector_index_options=index_options
        )
        
        self.assertEqual(options.column_name, "openai_embeddings")
        self.assertEqual(options.dimension, 1536)
        self.assertIsNone(options.model)
        self.assertEqual(options.vector_service_options, vector_options)
        self.assertEqual(options.table_vector_index_options, index_options)
        self.assertEqual(options.type, VectorColumnType.VECTORIZE)
    
    def test_from_precomputed_embeddings(self):
        """Test VectorColumnOptions.from_precomputed_embeddings factory method.
        
        Verifies that:
        - Column name is set correctly
        - Dimension is set correctly
        - Model is None
        - Vector service options are None
        - Table vector index options are properly stored
        - Type is set to PRECOMPUTED
        """
        index_options = TableVectorIndexOptions(metric=VectorMetric.DOT_PRODUCT)
        
        options = VectorColumnOptions.from_precomputed_embeddings(
            column_name="precomputed_embeddings",
            dimension=512,
            table_vector_index_options=index_options
        )
        
        self.assertEqual(options.column_name, "precomputed_embeddings")
        self.assertEqual(options.dimension, 512)
        self.assertIsNone(options.model)
        self.assertIsNone(options.vector_service_options)
        self.assertEqual(options.table_vector_index_options, index_options)
        self.assertEqual(options.type, VectorColumnType.PRECOMPUTED)
    
    def test_to_dict(self):
        """Test VectorColumnOptions.to_dict method.
        
        Verifies that:
        - Method correctly serializes all options into a dictionary format
        - All expected fields are present with correct values
        - Different types of options are correctly represented
        - Nested objects like table_vector_index_options are also serialized
        - Tests with multiple types of VectorColumnOptions
        """
        mock_model = SentenceTransformer("all-MiniLM-L6-v2")
        mock_model.model_name = "test-model"

        class FakeModelCard:
            base_model = "test-model"

        mock_model.model_card_data = FakeModelCard()
        
        index_options = TableVectorIndexOptions(metric=VectorMetric.COSINE)
        
        options = VectorColumnOptions.from_sentence_transformer(
            model=mock_model,
            column_name="embeddings",
            table_vector_index_options=index_options
        )
        
        result = options.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["column_name"], "embeddings")
        self.assertEqual(result["dimension"], 384)
        self.assertEqual(result["type"], "SENTENCE_TRANSFORMER")
        self.assertEqual(result["model"], "test-model")
        self.assertIsNone(result["vector_service_options"])
        self.assertIsInstance(result["table_vector_index_options"], dict)
        self.assertEqual(result["table_vector_index_options"]["metric"], "cosine")
        
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
        """Test the type property of VectorColumnOptions.
        
        Verifies that:
        - The type property returns the correct VectorColumnType enum value for each factory method
        - The type property is read-only and cannot be modified
        """
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
        
        self.assertEqual(st_options.type, VectorColumnType.SENTENCE_TRANSFORMER)
        self.assertEqual(vector_options.type, VectorColumnType.VECTORIZE)
        self.assertEqual(precomputed_options.type, VectorColumnType.PRECOMPUTED)
        
        with self.assertRaises(AttributeError):
            st_options.type = VectorColumnType.PRECOMPUTED
    
    def test_validation_errors(self):
        """Test validation errors in VectorColumnOptions factory methods.
        
        Verifies that:
        - Required parameters cannot be omitted (column_name, dimension, model)
        - TypeError is raised when required parameters are missing
        - ValueError is raised when parameters have incorrect types
        - Tests all factory methods for proper validation
        """
        with self.assertRaises(TypeError):
            VectorColumnOptions.from_precomputed_embeddings(dimension=512)
        
        with self.assertRaises(TypeError):
            VectorColumnOptions.from_precomputed_embeddings(column_name="test")
        
        with self.assertRaises(TypeError):
            VectorColumnOptions.from_sentence_transformer(column_name="test")
        
        with self.assertRaises(TypeError):
            VectorColumnOptions.from_vectorize(
                column_name="test",
                dimension=512
            )
        
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.model_card_data = MagicMock(base_model="test-model")
        
        with self.assertRaises(ValueError):
            VectorColumnOptions.from_sentence_transformer(
                model=mock_model,
                table_vector_index_options="invalid"
            )
    
    def test_model_config(self):
        """Test the model_config settings of VectorColumnOptions.
        
        Verifies that the Pydantic model configuration has arbitrary_types_allowed
        set to True, which is necessary for storing non-serializable objects like
        SentenceTransformer models.
        """
        self.assertTrue(VectorColumnOptions.model_config.get("arbitrary_types_allowed"))


if __name__ == "__main__":
    unittest.main()
