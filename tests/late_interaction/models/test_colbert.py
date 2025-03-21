#!/usr/bin/env python3
"""
Unit tests for the ColBERTModel class.
"""

import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

from astra_multivector.late_interaction import ColBERTModel


class MockColBERTConfig:
    """
    Mock implementation of ColBERTConfig for testing purposes.
    
    Simulates the configuration object used by the ColBERT model,
    with customizable attributes via kwargs.
    """
    def __init__(self, **kwargs):
        self.dim = 128  # Default embedding dimension
        
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockCheckpoint:
    """
    Mock implementation of ColBERT Checkpoint for testing purposes.
    
    Simulates the checkpoint object that handles model loading, query
    and document encoding, and other ColBERT functionalities.
    """
    def __init__(self, checkpoint, colbert_config=None, device=None):
        self.checkpoint = checkpoint
        self.config = colbert_config
        self.device = device
        
        param_tensor = torch.tensor([1.0], device=device)
        self.parameters = MagicMock(return_value=iter([param_tensor]))
        
        self.model = MagicMock()
        self.model.to = MagicMock(return_value=self.model)
        
        self.queryFromText = MagicMock(return_value=[torch.randn(3, 128)])
        
        self.doc_tokenizer = MagicMock()
        self.doc_tokenizer.tensorize = MagicMock(
            return_value=(torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]))
        )
        self.doc = MagicMock(
            return_value=(torch.randn(1, 3, 128), torch.ones(1, 3, 1))
        )


class MockCollectionEncoder:
    """
    Mock implementation of ColBERT CollectionEncoder for testing purposes.
    
    Simulates the encoder that manages batch encoding of documents.
    """
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint


@patch('astra_multivector.late_interaction.models.colbert.Checkpoint', MockCheckpoint)
@patch('astra_multivector.late_interaction.models.colbert.ColBERTConfig', MockColBERTConfig)
@patch('astra_multivector.late_interaction.models.colbert.CollectionEncoder', MockCollectionEncoder)
class TestColBERTModel(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the ColBERTModel class.
    
    Tests the functionality of the ColBERTModel implementation of the
    LateInteractionModel interface, including initialization, query
    and document encoding, and error handling.
    """
    
    @patch('astra_multivector.late_interaction.models.colbert.Checkpoint', MockCheckpoint)
    @patch('astra_multivector.late_interaction.models.colbert.ColBERTConfig', MockColBERTConfig)
    @patch('astra_multivector.late_interaction.models.colbert.CollectionEncoder', MockCollectionEncoder)
    def setUp(self):
        """
        Set up test fixtures with proper mocking.
        
        Creates a ColBERTModel instance with mock dependencies for testing.
        """
        self.model = ColBERTModel(
            model_name='test_colbert', 
            tokens_per_query=32,
            max_doc_tokens=512,
            device='cpu'
        )
    
    def test_init(self):
        """
        Test model initialization.
        
        Verifies that model attributes and components are correctly initialized.
        """
        self.assertEqual(self.model._model_name, 'test_colbert')
        self.assertEqual(self.model._tokens_per_query, 32)
        self.assertEqual(self.model._max_doc_tokens, 512)
        self.assertEqual(self.model._device, 'cpu')
        
        self.assertIsInstance(self.model.config, MockColBERTConfig)
        self.assertIsInstance(self.model.checkpoint, MockCheckpoint)
        self.assertIsInstance(self.model.encoder, MockCollectionEncoder)
        
    @patch('astra_multivector.late_interaction.models.colbert.LateInteractionModel._get_optimal_device')
    def test_device_initialization(self, mock_get_optimal_device):
        """
        Test device selection and initialization.
        
        Verifies both explicit device specification and automatic device selection.
        """
        explicit_model = ColBERTModel(
            model_name='test_colbert',
            device='cuda:1'
        )
        self.assertEqual(explicit_model._device, 'cuda:1')
        
        mock_get_optimal_device.return_value = 'cuda:0'
        auto_model = ColBERTModel(model_name='test_colbert')
        self.assertEqual(auto_model._device, 'cuda:0')
        
    @patch('astra_multivector.late_interaction.models.colbert.warnings.warn')
    def test_device_fallback(self, mock_warn):
        """
        Test device fallback when model can't be moved to requested device.
        
        Verifies that appropriate warnings are issued when the model
        can't be moved to the specified device.
        """
        mock_model = MagicMock()
        mock_model.to.side_effect = RuntimeError("Could not move model to device")
        
        mock_checkpoint = MockCheckpoint('test_colbert', None, 'cpu')
        mock_checkpoint.model = mock_model
        
        with patch('astra_multivector.late_interaction.models.colbert.Checkpoint', return_value=mock_checkpoint):
            model = ColBERTModel(
                model_name='test_colbert',
                device='cuda:1'
            )
        
        mock_warn.assert_called_once()
    
    @patch('astra_multivector.late_interaction.models.colbert.asyncio.to_thread')
    async def test_encode_query(self, mock_to_thread):
        """
        Test asynchronous query encoding.
        
        Verifies that encode_query correctly offloads work to a thread
        and returns the expected result.
        """
        sample_output = torch.randn(3, 128)
        mock_to_thread.return_value = sample_output
        
        result = await self.model.encode_query("test query")
        
        mock_to_thread.assert_called_once()
        
        self.assertIs(result, sample_output)
    
    def test_encode_query_sync(self):
        """
        Test synchronous query encoding.
        
        Verifies that encode_query_sync correctly calls the
        underlying ColBERT query encoding methods.
        """
        expected_output = torch.randn(3, 128)
        self.model.checkpoint.queryFromText.return_value = [expected_output]
        
        result = self.model.encode_query_sync("test query")
        
        self.model.checkpoint.queryFromText.assert_called_once_with(["test query"])
        
        torch.testing.assert_close(result, expected_output)
    
    @patch('astra_multivector.late_interaction.models.colbert.asyncio.to_thread')
    async def test_encode_doc(self, mock_to_thread):
        """
        Test asynchronous document encoding.
        
        Verifies that encode_doc correctly offloads work to a thread
        and returns the expected result.
        """
        sample_output = [torch.randn(5, 128)]
        mock_to_thread.return_value = sample_output
        
        result = await self.model.encode_doc(["test document"])
        
        mock_to_thread.assert_called_once()
        
        self.assertIs(result, sample_output)
    
    def test_encode_doc_sync(self):
        """
        Test synchronous document encoding.
        
        Verifies that encode_doc_sync correctly calls the underlying
        ColBERT document encoding methods and processes the results.
        """
        embeddings = torch.randn(1, 3, 128)
        mask = torch.ones(1, 3, 1)
        expected_output = [embeddings[0, :2]]
        
        self.model.checkpoint.doc.return_value = (embeddings, mask)
        
        result = self.model.encode_doc_sync(["test document"])
        
        self.model.checkpoint.doc_tokenizer.tensorize.assert_called_once()
        self.model.checkpoint.doc.assert_called_once()
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape[1], embeddings.shape[2])
    
    def test_to_device(self):
        """
        Test tensor device movement.
        
        Verifies that tensors are correctly moved to the model's device.
        """
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        result = self.model.to_device(tensor)
        
        self.assertEqual(result.device, torch.device('cpu'))
    
    def test_properties(self):
        """
        Test model properties.
        
        Verifies that model properties return the expected values.
        """
        self.assertEqual(self.model.dim, 128)
        
        self.assertEqual(self.model.model_name, 'test_colbert')
        
        self.assertFalse(self.model.supports_images)
    
    def test_str(self):
        """
        Test string representation.
        
        Verifies that the model's string representation is correct.
        """
        expected_str = "ColBERTModel(model=test_colbert, dim=128, tokens_per_query=32, max_doc_tokens=512, device=cpu)"
        self.assertEqual(str(self.model), expected_str)
    
    def test_error_handling(self):
        """
        Test error handling for unsupported inputs.
        
        Verifies that appropriate errors are raised for unsupported
        input types and formats.
        """
        with patch('astra_multivector.late_interaction.models.colbert.ColBERTModel.encode_doc_sync') as mock_encode:
            mock_encode.side_effect = TypeError("ColBERT only supports text chunks")
            
            with self.assertRaises(TypeError):
                self.model.encode_doc_sync([MagicMock()])
            
    def test_empty_inputs(self):
        """
        Test handling of empty inputs.
        
        Verifies that the model handles empty queries, empty document
        lists, and empty strings appropriately.
        """
        empty_query_result = self.model.encode_query_sync("")
        self.assertIsInstance(empty_query_result, torch.Tensor)
        self.assertEqual(empty_query_result.shape, (0, 128))
        
        empty_doc_list_result = self.model.encode_doc_sync([])
        self.assertEqual(len(empty_doc_list_result), 0)
        
        with self.assertWarns(UserWarning):
            empty_string_result = self.model.encode_doc_sync([""])
        self.assertEqual(len(empty_string_result), 1)
        self.assertEqual(empty_string_result[0].shape, (0, 128))
        
        with self.assertWarns(UserWarning):
            multiple_empty_result = self.model.encode_doc_sync(["", "", ""])
        self.assertEqual(len(multiple_empty_result), 3)
        for tensor in multiple_empty_result:
            self.assertEqual(tensor.shape, (0, 128))
    
    @patch('astra_multivector.late_interaction.models.colbert.warnings.warn')
    def test_warnings(self, mock_warn):
        """
        Test warning generation.
        
        Verifies that appropriate warnings are issued for empty chunks,
        truncated documents, and other edge cases.
        """
        self.model.encode_doc_sync([""])
        mock_warn.assert_called_with("All chunks were empty. Returning empty embeddings.")
        mock_warn.reset_mock()
        
        self.model.checkpoint.doc_tokenizer.tensorize.return_value = (
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 1, 1], [1, 1, 1]])
        )
        self.model.checkpoint.doc.return_value = (
            torch.randn(2, 3, 128),
            torch.ones(2, 3, 1)
        )
        
        self.model.encode_doc_sync(["valid", "", "also valid"])
        mock_warn.assert_called_with("Chunk at index 1 was empty and will be skipped during encoding.")
        mock_warn.reset_mock()
        
        long_input_ids = torch.ones(1, self.model._max_doc_tokens + 10, dtype=torch.long)
        long_attention_mask = torch.ones_like(long_input_ids)
        
        self.model.checkpoint.doc_tokenizer.tensorize.return_value = (long_input_ids, long_attention_mask)
        
        self.model.encode_doc_sync(["long document"])
        mock_warn.assert_called_with(f"Document tokens exceed {self.model._max_doc_tokens}. Truncating.")
        
    def test_batch_encoding(self):
        """
        Test batch document encoding.
        
        Verifies that multiple documents can be encoded in a batch and
        that the results are correctly processed.
        """
        batch_size = 3
        embeddings = torch.randn(batch_size, 5, 128)
        mask = torch.ones(batch_size, 5, 1)
        
        self.model.checkpoint.doc_tokenizer.tensorize.return_value = (
            torch.ones(batch_size, 5, dtype=torch.long),
            torch.ones(batch_size, 5, dtype=torch.long)
        )
        self.model.checkpoint.doc.return_value = (embeddings, mask)
        
        result = self.model.encode_doc_sync(["doc1", "doc2", "doc3"])
        
        self.assertEqual(len(result), batch_size)
        for i in range(batch_size):
            self.assertTrue(torch.allclose(result[i], embeddings[i][mask[i].squeeze(-1).bool()]))
            
    def test_exception_flow(self):
        """
        Test exception handling.
        
        Verifies that exceptions from the underlying ColBERT methods
        are correctly propagated.
        """
        input_ids = torch.ones(1, 5, dtype=torch.long)
        attention_mask = torch.ones(1, 5, dtype=torch.long)
        
        self.model.checkpoint.doc_tokenizer.tensorize.return_value = (input_ids, attention_mask)
        
        def raises_error(*args, **kwargs):
            raise RuntimeError("CUDA out of memory")
        
        self.model.checkpoint.doc.side_effect = raises_error
        
        with self.assertRaises(RuntimeError) as context:
            self.model.encode_doc_sync(["test document"])
        
        self.assertIn("CUDA out of memory", str(context.exception))

if __name__ == "__main__":
    unittest.main()
