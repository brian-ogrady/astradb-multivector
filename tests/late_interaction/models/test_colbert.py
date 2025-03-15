#!/usr/bin/env python3
"""
Unit tests for the ColBERTModel class.
"""

import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

from astra_multivector.late_interaction import ColBERTModel
from astra_multivector.late_interaction.models.colbert import _get_module_device


# Mock the ColBERT dependencies
class MockColBERTConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockCheckpoint:
    def __init__(self, checkpoint, colbert_config=None, device=None):
        self.checkpoint = checkpoint
        self.config = colbert_config
        self.device = device
        self.parameters = MagicMock(return_value=[torch.tensor([1.0], device=device)])
        
        # Mock query tokenizer and encoder
        self.queryFromText = MagicMock(return_value=[torch.randn(3, 128)])
        
        # Mock document tokenizer and encoder
        self.doc_tokenizer = MagicMock()
        self.doc_tokenizer.tensorize = MagicMock(
            return_value=(torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]))
        )
        self.doc = MagicMock(
            return_value=(torch.randn(1, 3, 128), torch.ones(1, 3, 1))
        )


class MockCollectionEncoder:
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint


@patch('astra_multivector.late_interaction.models.colbert.Checkpoint', MockCheckpoint)
@patch('astra_multivector.late_interaction.models.colbert.ColBERTConfig', MockColBERTConfig)
@patch('astra_multivector.late_interaction.models.colbert.CollectionEncoder', MockCollectionEncoder)
class TestColBERTModel(unittest.TestCase):
    """Tests for the ColBERTModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a model with mocked dependencies
        self.model = ColBERTModel(
            model_name='test_colbert', 
            tokens_per_query=32,
            device='cpu'
        )
    
    def test_init(self):
        """Test model initialization."""
        # Check that model attributes are set correctly
        self.assertEqual(self.model._model_name, 'test_colbert')
        self.assertEqual(self.model._tokens_per_query, 32)
        self.assertEqual(self.model._device, 'cpu')
        
        # Check that ColBERT components are initialized
        self.assertIsInstance(self.model.config, MockColBERTConfig)
        self.assertIsInstance(self.model.checkpoint, MockCheckpoint)
        self.assertIsInstance(self.model.encoder, MockCollectionEncoder)
    
    @patch('astra_multivector.late_interaction.models.colbert.asyncio.to_thread')
    async def test_encode_query(self, mock_to_thread):
        """Test query encoding."""
        # Set up the mock to return a sample tensor
        sample_output = torch.randn(3, 128)
        mock_to_thread.return_value = sample_output
        
        # Call the method
        result = await self.model.encode_query("test query")
        
        # Check that to_thread was called with the correct arguments
        mock_to_thread.assert_called_once()
        
        # Check that the result is correct
        self.assertIs(result, sample_output)
    
    def test_encode_query_sync(self):
        """Test synchronous query encoding."""
        # Set up the mock checkpoint
        expected_output = torch.randn(3, 128)
        self.model.checkpoint.queryFromText.return_value = [expected_output]
        
        # Call the method
        result = self.model.encode_query_sync("test query")
        
        # Check that the checkpoint method was called with the correct arguments
        self.model.checkpoint.queryFromText.assert_called_once_with(["test query"])
        
        # Check that the result is correct
        torch.testing.assert_close(result, expected_output)
    
    @patch('astra_multivector.late_interaction.models.colbert.asyncio.to_thread')
    async def test_encode_doc(self, mock_to_thread):
        """Test document encoding."""
        # Set up the mock to return a sample list of tensors
        sample_output = [torch.randn(5, 128)]
        mock_to_thread.return_value = sample_output
        
        # Call the method
        result = await self.model.encode_doc(["test document"])
        
        # Check that to_thread was called with the correct arguments
        mock_to_thread.assert_called_once()
        
        # Check that the result is correct
        self.assertIs(result, sample_output)
    
    def test_encode_doc_sync(self):
        """Test synchronous document encoding."""
        # Set up expected outputs for mock methods
        embeddings = torch.randn(1, 3, 128)
        mask = torch.ones(1, 3, 1)
        expected_output = [embeddings[0, :2]]  # First 2 tokens
        
        # Update the mock
        self.model.checkpoint.doc.return_value = (embeddings, mask)
        
        # Call the method
        result = self.model.encode_doc_sync(["test document"])
        
        # Check that document encoder was called
        self.model.checkpoint.doc_tokenizer.tensorize.assert_called_once()
        self.model.checkpoint.doc.assert_called_once()
        
        # Check results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape[1], embeddings.shape[2])
    
    def test_to_device(self):
        """Test tensor device movement."""
        # Create a tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        # Call the method
        result = self.model.to_device(tensor)
        
        # Check result device (should be the same as model)
        self.assertEqual(result.device, torch.device('cpu'))
    
    def test_properties(self):
        """Test model properties."""
        # Test dim property
        self.assertEqual(self.model.dim, 128)
        
        # Test model_name property
        self.assertEqual(self.model.model_name, 'test_colbert')
        
        # Test supports_images property (should inherit False from base class)
        self.assertFalse(self.model.supports_images)
    
    def test_str(self):
        """Test string representation."""
        # Test __str__ method
        expected_str = "ColBERTModel(model=test_colbert, dim=128, device=cpu)"
        self.assertEqual(str(self.model), expected_str)
    
    def test_error_handling(self):
        """Test error handling for unsupported inputs."""
        # Test error when passing non-text inputs
        with self.assertRaises(TypeError):
            # Should raise TypeError when non-string chunks are provided
            self.model.encode_doc_sync([MagicMock()])


class TestHelperFunctions(unittest.TestCase):
    """Tests for helper functions in the colbert module."""
    
    def test_get_module_device(self):
        """Test the _get_module_device helper function."""
        # Create a mock module with parameters on CPU
        mock_module = MagicMock()
        param = torch.nn.Parameter(torch.tensor([1.0], device='cpu'))
        mock_module.parameters.return_value = iter([param])
        
        # Test function
        device = _get_module_device(mock_module)
        self.assertEqual(device, torch.device('cpu'))
        
        # Create a mock module with parameters on CUDA
        # Skip if CUDA is not available
        if torch.cuda.is_available():
            mock_module = MagicMock()
            param = torch.nn.Parameter(torch.tensor([1.0], device='cuda:0'))
            mock_module.parameters.return_value = iter([param])
            
            # Test function
            device = _get_module_device(mock_module)
            self.assertEqual(device, torch.device('cuda:0'))


if __name__ == "__main__":
    unittest.main()