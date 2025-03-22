#!/usr/bin/env python3
"""
Unit tests for the ColPaliModel class.
"""

import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from PIL import Image

from astra_multivector.late_interaction import ColPaliModel


class MockColPali:
    """
    Mock implementation of ColPali for testing purposes.
    
    Simulates the ColPali model with minimal functionality needed for tests.
    Provides embeddings of the correct shape and handles both query and image inputs.
    """
    def __init__(self, device="cpu"):
        self.dim = 768
        
        param_tensor = torch.tensor([1.0], device=device)
        self._params = [param_tensor]
    
    def eval(self):
        """
        Mock the eval method to mimic model behavior.
        
        Returns:
            Self to allow method chaining
        """
        return self
    
    def __call__(self, **kwargs):
        """
        Mock forward pass through the model.
        
        Returns different outputs based on input type (query vs image).
        
        Args:
            **kwargs: Input tensors and model parameters
            
        Returns:
            List of embedding tensors
        """
        if 'pixel_values' in kwargs:  # Image input
            batch_size = kwargs.get('pixel_values', torch.zeros(1, 3, 224, 224)).shape[0]
            return [torch.randn(5, 768) for _ in range(batch_size)]
        else:  # Query input
            return [torch.randn(3, 768)]
    
    def parameters(self):
        """
        Return model parameters for device detection.
        
        Returns:
            Iterator of parameter tensors
        """
        return iter(self._params)


class MockColQwen2(MockColPali):
    """
    Mock implementation of ColQwen2 for testing purposes.
    
    Inherits from MockColPali to provide the same functionality
    but with a different class name for testing model selection logic.
    """
    pass


class MockProcessor:
    """
    Mock implementation of ColPali processor for testing purposes.
    
    Provides methods to process both queries and images, returning
    appropriate tensor shapes for model input.
    """
    def __init__(self):
        pass
    
    def process_queries(self, queries):
        """
        Process text queries into model inputs.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dictionary of input tensors
        """
        return {
            'input_ids': torch.ones(len(queries), 10, dtype=torch.long),
            'attention_mask': torch.ones(len(queries), 10, dtype=torch.long)
        }
    
    def process_images(self, images):
        """
        Process images into model inputs.
        
        Args:
            images: List of PIL images
            
        Returns:
            Dictionary of input tensors
        """
        return {
            'pixel_values': torch.randn(len(images), 3, 224, 224)
        }


def mock_model_from_pretrained(model_name, **kwargs):
    """
    Mock the from_pretrained method for ColPali and ColQwen2.
    
    Args:
        model_name: Name of the model to load
        **kwargs: Additional arguments
        
    Returns:
        MockColPali instance
    """
    return MockColPali()


def mock_processor_from_pretrained(model_name, **kwargs):
    """
    Mock the from_pretrained method for processors.
    
    Args:
        model_name: Name of the processor to load
        **kwargs: Additional arguments
        
    Returns:
        MockProcessor instance
    """
    return MockProcessor()


class TestColPaliModel(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the ColPaliModel class.
    
    Tests the functionality of the ColPaliModel implementation of the
    LateInteractionModel interface, including initialization, query and
    image encoding, device handling, and error cases.
    """
    
    @patch('astra_multivector.late_interaction.models.colpali.ColPali.from_pretrained', mock_model_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColQwen2.from_pretrained', mock_model_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColPaliProcessor.from_pretrained', mock_processor_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColQwen2Processor.from_pretrained', mock_processor_from_pretrained)
    def setUp(self):
        """
        Set up test fixtures with mocked dependencies.
        
        Creates a ColPaliModel instance with mock implementations of
        ColPali and its processor to avoid actual model loading.
        Also creates a test image for encoding tests.
        """
        self.model = ColPaliModel(
            model_name='vidore/colpali-v0.1',
            device='cpu'
        )
        
        self.test_image = MagicMock(spec=Image.Image)
        self.test_image.width = 224
        self.test_image.height = 224
    
    def test_init_standard_model(self):
        """
        Test initialization with standard ColPali model.
        
        Verifies that model attributes and dependencies are correctly
        initialized when using a standard ColPali model name.
        """
        self.assertEqual(self.model._model_name, 'vidore/colpali-v0.1')
        self.assertEqual(self.model._device, 'cpu')
        self.assertIsInstance(self.model.colpali, MockColPali)
        self.assertIsInstance(self.model.processor, MockProcessor)
    
    @patch('astra_multivector.late_interaction.models.colpali.ColPali.from_pretrained', mock_model_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColQwen2.from_pretrained', mock_model_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColPaliProcessor.from_pretrained', mock_processor_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColQwen2Processor.from_pretrained', mock_processor_from_pretrained)
    def test_init_qwen_model(self):
        """
        Test initialization with ColQwen2 model.
        
        Verifies that model selection logic correctly identifies
        and initializes a ColQwen2 model based on the model name.
        """
        model = ColPaliModel(
            model_name='vidore/colqwen2-v0.1',
            device='cpu'
        )
        
        self.assertEqual(model._model_name, 'vidore/colqwen2-v0.1')
    
    @patch('astra_multivector.late_interaction.models.colpali.ColPali.from_pretrained')
    @patch('astra_multivector.late_interaction.models.colpali.logger.warning')
    def test_device_fallback(self, mock_warning, mock_from_pretrained):
        """
        Test device fallback mechanism.
        
        Verifies that when model loading fails on a specified device,
        the model falls back to automatic device mapping and logs a warning.
        """
        mock_from_pretrained.side_effect = [
            RuntimeError("Could not load model on specified device"),
            MockColPali()
        ]
        
        with patch('astra_multivector.late_interaction.models.colpali.ColPaliProcessor.from_pretrained', 
                  return_value=MockProcessor()):
            model = ColPaliModel(
                model_name='vidore/colpali-v0.1',
                device='cuda:0'
            )
        
        mock_warning.assert_called_once()
        self.assertIn("Could not load model", mock_warning.call_args[0][0])
        
        self.assertEqual(mock_from_pretrained.call_args_list[1][1]['device_map'], "auto")
        
    @patch('astra_multivector.late_interaction.models.colpali.asyncio.to_thread')
    async def test_encode_query(self, mock_to_thread):
        """
        Test asynchronous query encoding.
        
        Verifies that encode_query correctly offloads work to a thread
        and returns the expected result.
        """
        sample_output = torch.randn(3, 768)
        mock_to_thread.return_value = sample_output
        
        result = await self.model.encode_query("test query")
        
        mock_to_thread.assert_called_once()
        self.assertEqual(mock_to_thread.call_args[0][0], self.model.encode_query_sync)
        self.assertEqual(mock_to_thread.call_args[0][1], "test query")
        
        self.assertIs(result, sample_output)
    
    def test_encode_query_sync(self):
        """
        Test synchronous query encoding.
        
        Verifies that encode_query_sync correctly processes query text
        through the model and returns valid embeddings.
        """
        expected_output = torch.randn(3, 768)
        with patch.object(self.model.colpali, '__call__', return_value=[expected_output]):
            result = self.model.encode_query_sync("test query")
            
            self.assertEqual(result.shape,(3, 768))
            self.assertEqual(result.dtype, expected_output.dtype)
            self.assertEqual(result.device, expected_output.device)

            self.assertTrue(torch.isfinite(result[0]).all())
            self.assertFalse(torch.all(result == 0))
    
    def test_encode_query_sync_empty(self):
        """
        Test synchronous query encoding with empty input.
        
        Verifies that encode_query_sync correctly handles empty queries
        by returning an empty tensor with the correct dimensions.
        """
        result = self.model.encode_query_sync("")
        
        self.assertEqual(result.shape, (0, self.model.dim))
    
    @patch('astra_multivector.late_interaction.models.colpali.asyncio.to_thread')
    async def test_encode_doc(self, mock_to_thread):
        """
        Test asynchronous document encoding.
        
        Verifies that encode_doc correctly offloads work to a thread
        and returns the expected result.
        """
        sample_output = [torch.randn(5, 768)]
        mock_to_thread.return_value = sample_output
        
        result = await self.model.encode_doc([self.test_image])
        
        mock_to_thread.assert_called_once()
        self.assertEqual(mock_to_thread.call_args[0][0], self.model.encode_doc_sync)
        self.assertEqual(len(mock_to_thread.call_args[0][1]), 1)
        
        self.assertIs(result, sample_output)
    
    def test_encode_doc_sync(self):
        """
        Test synchronous document encoding.
        
        Verifies that encode_doc_sync correctly processes images
        through the model and returns valid embeddings.
        """
        embeddings = [torch.randn(5, 768)]
        
        with patch.object(self.model.colpali, '__call__', return_value=embeddings):
            result = self.model.encode_doc_sync([self.test_image])
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].shape, embeddings[0].shape)
            self.assertEqual(result[0].dtype, embeddings[0].dtype)
            self.assertEqual(result[0].device, embeddings[0].device)
            self.assertTrue(torch.isfinite(result[0]).all())
            self.assertFalse(torch.all(result[0] == 0))
    
    def test_encode_doc_sync_empty(self):
        """
        Test document encoding with empty input.
        
        Verifies that encode_doc_sync correctly handles empty input
        by returning an empty list.
        """
        result = self.model.encode_doc_sync([])
        
        self.assertEqual(result, [])
    
    def test_encode_doc_sync_invalid_images(self):
        """
        Test document encoding with invalid images.
        
        Verifies that encode_doc_sync correctly handles invalid images
        (with zero dimensions) by returning empty embeddings and logging warnings.
        """
        invalid_image = MagicMock(spec=Image.Image)
        invalid_image.width = 0
        invalid_image.height = 0
        
        with patch('astra_multivector.late_interaction.models.colpali.logger.warning') as mock_warning:
            result = self.model.encode_doc_sync([invalid_image])
        
        self.assertEqual(mock_warning.call_count, 2)
        self.assertIn("invalid", mock_warning.call_args[0][0].lower())
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (0, self.model.dim))
    
    def test_encode_doc_sync_mixed_images(self):
        """
        Test document encoding with mix of valid and invalid images.
        
        Verifies that encode_doc_sync correctly processes a mix of valid
        and invalid images, producing proper embeddings for valid images
        and empty embeddings for invalid ones.
        """
        valid_image = MagicMock(spec=Image.Image)
        valid_image.width = 224
        valid_image.height = 224
        
        invalid_image = MagicMock(spec=Image.Image)
        invalid_image.width = 0
        invalid_image.height = 0
        
        embedding = torch.randn(5, 768)
        with patch.object(self.model.colpali, '__call__', return_value=[embedding]):
            with patch('astra_multivector.late_interaction.models.colpali.logger.warning'):
                result = self.model.encode_doc_sync([valid_image, invalid_image])
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape,(5, 768))
        self.assertEqual(result[0].dtype, embedding[0].dtype)
        self.assertEqual(result[0].device, embedding[0].device)
        self.assertTrue(torch.isfinite(result[0]).all())
        self.assertFalse(torch.all(result[0] == 0))
        self.assertEqual(result[1].shape, (0, self.model.dim))
    
    def test_encode_doc_non_image_input(self):
        """
        Test document encoding with non-image input.
        
        Verifies that encode_doc_sync correctly raises TypeError
        when given non-image inputs.
        """
        with self.assertRaises(TypeError):
            self.model.encode_doc_sync(["not an image"])
    
    def test_to_device(self):
        """
        Test tensor device movement.
        
        Verifies that to_device correctly moves tensors, dictionaries
        of tensors, and handles None input.
        """
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.model.to_device(tensor)
        self.assertEqual(result.device, torch.device('cpu'))
        
        self.assertIsNone(self.model.to_device(None))
        
        tensor_dict = {
            'a': torch.tensor([1.0]),
            'b': torch.tensor([2.0]),
            'nested': {
                'c': torch.tensor([3.0])
            }
        }
        result_dict = self.model.to_device(tensor_dict)
        self.assertEqual(result_dict['a'].device, torch.device('cpu'))
        self.assertEqual(result_dict['b'].device, torch.device('cpu'))
        self.assertEqual(result_dict['nested']['c'].device, torch.device('cpu'))
        
        with self.assertRaises(TypeError):
            self.model.to_device(123)
    
    def test_properties(self):
        """
        Test model properties.
        
        Verifies that model properties return the expected values.
        """
        self.assertEqual(self.model.dim, 768)
        self.assertEqual(self.model.model_name, 'vidore/colpali-v0.1')
        self.assertTrue(self.model.supports_images)
    
    def test_str(self):
        """
        Test string representation.
        
        Verifies that the model's string representation is correct.
        """
        expected_fragments = [
            "ColPaliModel",
            "model=vidore/colpali-v0.1",
            "dim=768",
            "device=cpu",
            "supports_images=True"
        ]
        
        model_str = str(self.model)
        for fragment in expected_fragments:
            self.assertIn(fragment, model_str)


if __name__ == "__main__":
    unittest.main()