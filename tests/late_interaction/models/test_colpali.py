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


# Mock the ColPali dependencies
class MockColPali:
    def __init__(self, device="cpu"):
        self.dim = 768
        
        # Create a parameter tensor on the specified device
        param_tensor = torch.tensor([1.0], device=device)
        
        # Store the parameter in an attribute
        self._params = [param_tensor]
    
    def eval(self):
        return self
    
    def __call__(self, **kwargs):
        # Return different outputs depending on the input
        if 'pixel_values' in kwargs:  # Image input
            batch_size = kwargs.get('pixel_values', torch.zeros(1, 3, 224, 224)).shape[0]
            return [torch.randn(5, 768) for _ in range(batch_size)]
        else:  # Query input
            return [torch.randn(3, 768)]
    
    def parameters(self):
        # Return an iterator of parameters, not a list
        return iter(self._params)

class MockColQwen2(MockColPali):
    pass


class MockProcessor:
    def __init__(self):
        pass
    
    def process_queries(self, queries):
        # Return a dictionary with tokenized query
        return {
            'input_ids': torch.ones(len(queries), 10, dtype=torch.long),
            'attention_mask': torch.ones(len(queries), 10, dtype=torch.long)
        }
    
    def process_images(self, images):
        # Return a dictionary with processed images
        return {
            'pixel_values': torch.randn(len(images), 3, 224, 224)
        }


# Patched from_pretrained methods
def mock_model_from_pretrained(model_name, **kwargs):
    return MockColPali()


def mock_processor_from_pretrained(model_name, **kwargs):
    return MockProcessor()


class TestColPaliModel(unittest.IsolatedAsyncioTestCase):
    """Tests for the ColPaliModel class."""
    
    @patch('astra_multivector.late_interaction.models.colpali.ColPali.from_pretrained', mock_model_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColQwen2.from_pretrained', mock_model_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColPaliProcessor.from_pretrained', mock_processor_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColQwen2Processor.from_pretrained', mock_processor_from_pretrained)
    def setUp(self):
        """Set up test fixtures."""
        # Create a model with mocked dependencies
        self.model = ColPaliModel(
            model_name='vidore/colpali-v0.1',
            device='cpu'
        )
        
        # Create a test image for input
        self.test_image = MagicMock(spec=Image.Image)
        self.test_image.width = 224
        self.test_image.height = 224
    
    def test_init_standard_model(self):
        """Test initialization with standard ColPali model."""
        # Check model attributes
        self.assertEqual(self.model._model_name, 'vidore/colpali-v0.1')
        self.assertEqual(self.model._device, 'cpu')
        self.assertIsInstance(self.model.colpali, MockColPali)
        self.assertIsInstance(self.model.processor, MockProcessor)
    
    @patch('astra_multivector.late_interaction.models.colpali.ColPali.from_pretrained', mock_model_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColQwen2.from_pretrained', mock_model_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColPaliProcessor.from_pretrained', mock_processor_from_pretrained)
    @patch('astra_multivector.late_interaction.models.colpali.ColQwen2Processor.from_pretrained', mock_processor_from_pretrained)
    def test_init_qwen_model(self):
        """Test initialization with ColQwen2 model."""
        model = ColPaliModel(
            model_name='vidore/colqwen2-v0.1',
            device='cpu'
        )
        
        # Check that the right model class was selected
        self.assertEqual(model._model_name, 'vidore/colqwen2-v0.1')
    
    @patch('astra_multivector.late_interaction.models.colpali.ColPali.from_pretrained')
    @patch('astra_multivector.late_interaction.models.colpali.logger.warning')
    def test_device_fallback(self, mock_warning, mock_from_pretrained):
        """Test fallback to auto device mapping when specified device fails."""
        # First attempt should raise an error
        mock_from_pretrained.side_effect = [
            RuntimeError("Could not load model on specified device"),
            MockColPali()  # Second attempt succeeds
        ]
        
        # Create model with mocked errors
        with patch('astra_multivector.late_interaction.models.colpali.ColPaliProcessor.from_pretrained', 
                  return_value=MockProcessor()):
            model = ColPaliModel(
                model_name='vidore/colpali-v0.1',
                device='cuda:0'  # This will "fail" in our mock
            )
        
        # Check that warning was logged and fallback was used
        mock_warning.assert_called_once()
        self.assertIn("Could not load model", mock_warning.call_args[0][0])
        
        # Second call should have device_map="auto"
        self.assertEqual(mock_from_pretrained.call_args_list[1][1]['device_map'], "auto")
        
    @patch('astra_multivector.late_interaction.models.colpali.asyncio.to_thread')
    async def test_encode_query(self, mock_to_thread):
        """Test query encoding."""
        # Set up the mock to return a sample tensor
        sample_output = torch.randn(3, 768)
        mock_to_thread.return_value = sample_output
        
        # Call the method
        result = await self.model.encode_query("test query")
        
        # Check that to_thread was called with the correct function and arguments
        mock_to_thread.assert_called_once()
        self.assertEqual(mock_to_thread.call_args[0][0], self.model.encode_query_sync)
        self.assertEqual(mock_to_thread.call_args[0][1], "test query")
        
        # Check that the result is correct
        self.assertIs(result, sample_output)
    
    def test_encode_query_sync(self):
        """Test synchronous query encoding."""
        # Patch the colpali model call to return a known tensor
        expected_output = torch.randn(3, 768)
        with patch.object(self.model.colpali, '__call__', return_value=[expected_output]):
            # Call the method
            result = self.model.encode_query_sync("test query")
            
            # Check result
            self.assertEqual(result.shape,(3, 768))  # Check shape
            self.assertEqual(result.dtype, expected_output.dtype)  # Check data type
            self.assertEqual(result.device, expected_output.device)  # Check device

            # Check that the tensor contains finite values (no NaN or Inf)
            self.assertTrue(torch.isfinite(result[0]).all())

            # Check that tensor is not all zeros
            self.assertFalse(torch.all(result == 0))
    
    def test_encode_query_sync_empty(self):
        """Test synchronous query encoding with empty input."""
        # Call with empty query
        result = self.model.encode_query_sync("")
        
        # Should return empty tensor with correct dimensions
        self.assertEqual(result.shape, (0, self.model.dim))
    
    @patch('astra_multivector.late_interaction.models.colpali.asyncio.to_thread')
    async def test_encode_doc(self, mock_to_thread):
        """Test document encoding."""
        # Set up the mock to return a sample list of tensors
        sample_output = [torch.randn(5, 768)]
        mock_to_thread.return_value = sample_output
        
        # Call the method with test image
        result = await self.model.encode_doc([self.test_image])
        
        # Check that to_thread was called with the correct function and arguments
        mock_to_thread.assert_called_once()
        self.assertEqual(mock_to_thread.call_args[0][0], self.model.encode_doc_sync)
        self.assertEqual(len(mock_to_thread.call_args[0][1]), 1)
        
        # Check that the result is correct
        self.assertIs(result, sample_output)
    
    def test_encode_doc_sync(self):
        """Test synchronous document encoding."""
        # Patch the colpali model call to return known tensors
        embeddings = [torch.randn(5, 768)]
        
        with patch.object(self.model.colpali, '__call__', return_value=embeddings):
            # Call the method
            result = self.model.encode_doc_sync([self.test_image])
            
            # Check result
            self.assertEqual(len(result), 1)

            self.assertEqual(result[0].shape, embeddings[0].shape)  # Check shape
            self.assertEqual(result[0].dtype, embeddings[0].dtype)  # Check data type
            self.assertEqual(result[0].device, embeddings[0].device)  # Check device

            # Check that the tensor contains finite values (no NaN or Inf)
            self.assertTrue(torch.isfinite(result[0]).all())

            # Check that tensor is not all zeros
            self.assertFalse(torch.all(result[0] == 0))
    
    def test_encode_doc_sync_empty(self):
        """Test document encoding with empty input."""
        # Call with empty list
        result = self.model.encode_doc_sync([])
        
        # Should return empty list
        self.assertEqual(result, [])
    
    def test_encode_doc_sync_invalid_images(self):
        """Test document encoding with invalid images."""
        # Create an invalid image (zero dimensions)
        invalid_image = MagicMock(spec=Image.Image)
        invalid_image.width = 0
        invalid_image.height = 0
        
        # Call with invalid image
        with patch('astra_multivector.late_interaction.models.colpali.logger.warning') as mock_warning:
            result = self.model.encode_doc_sync([invalid_image])
        
        # Should log a warning
        self.assertEqual(mock_warning.call_count, 2)
        self.assertIn("invalid", mock_warning.call_args[0][0].lower())
        
        # Should return an empty embedding
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (0, self.model.dim))
    
    def test_encode_doc_sync_mixed_images(self):
        """Test document encoding with mix of valid and invalid images."""
        # Create valid and invalid images
        valid_image = MagicMock(spec=Image.Image)
        valid_image.width = 224
        valid_image.height = 224
        
        invalid_image = MagicMock(spec=Image.Image)
        invalid_image.width = 0
        invalid_image.height = 0
        
        # Patch model to return embedding for valid image
        embedding = torch.randn(5, 768)
        with patch.object(self.model.colpali, '__call__', return_value=[embedding]):
            with patch('astra_multivector.late_interaction.models.colpali.logger.warning'):
                # Call with mixed images
                result = self.model.encode_doc_sync([valid_image, invalid_image])
        
        # Should return embeddings for both, with empty for invalid
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape,(5, 768))  # Check shape
        self.assertEqual(result[0].dtype, embedding[0].dtype)  # Check data type
        self.assertEqual(result[0].device, embedding[0].device)  # Check device

        # Check that the tensor contains finite values (no NaN or Inf)
        self.assertTrue(torch.isfinite(result[0]).all())

        # Check that tensor is not all zeros
        self.assertFalse(torch.all(result[0] == 0))
        self.assertEqual(result[1].shape, (0, self.model.dim))
    
    def test_encode_doc_non_image_input(self):
        """Test document encoding with non-image input."""
        # Call with string input
        with self.assertRaises(TypeError):
            self.model.encode_doc_sync(["not an image"])
    
    def test_to_device(self):
        """Test tensor device movement."""
        # Test with tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.model.to_device(tensor)
        self.assertEqual(result.device, torch.device('cpu'))
        
        # Test with None
        self.assertIsNone(self.model.to_device(None))
        
        # Test with dict
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
        
        # Test with invalid type
        with self.assertRaises(TypeError):
            self.model.to_device(123)
    
    def test_properties(self):
        """Test model properties."""
        # Test dim property
        self.assertEqual(self.model.dim, 768)
        
        # Test model_name property
        self.assertEqual(self.model.model_name, 'vidore/colpali-v0.1')
        
        # Test supports_images property (should be True)
        self.assertTrue(self.model.supports_images)
    
    def test_str(self):
        """Test string representation."""
        # Test __str__ method
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