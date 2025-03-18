#!/usr/bin/env python3
"""
Unit tests for the LateInteractionModel base class.
"""

import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from astra_multivector.late_interaction import LateInteractionModel


# Create a concrete subclass for testing the abstract base class
class TestModel(LateInteractionModel):
    """Concrete implementation of LateInteractionModel for testing."""
    
    def __init__(self, dim=4, device=None):
        super().__init__(device=device)
        self._dim = dim
        self._model_name = "test_model"
    
    async def encode_query(self, q):
        # Return a tensor of shape [num_tokens, dim]
        return torch.randn(5, self._dim)
    
    async def encode_doc(self, chunks):
        # Return list of tensors of shape [num_tokens, dim]
        return [torch.randn(10, self._dim) for _ in range(len(chunks))]
    
    def to_device(self, T):
        # Just return the tensor as is
        return T
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def model_name(self):
        return self._model_name


class TestLateInteractionModel(unittest.TestCase):
    """Tests for the LateInteractionModel base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        
    @patch('astra_multivector.late_interaction.models.base.LateInteractionModel._get_optimal_device')
    def test_device_initialization(self, mock_get_optimal_device):
        """Test that device is initialized correctly."""
        # Test with default device
        mock_get_optimal_device.return_value = "cpu"
        model = TestModel()
        self.assertEqual(model._device, "cpu")
        
        # Test with specified device
        model = TestModel(device="cuda:1")
        self.assertEqual(model._device, "cuda:1")
        
        # Verify the optimal device method was called once
        mock_get_optimal_device.assert_called_once()
    
    def test_score_single(self):
        """Test the score method with a single document."""
        # Create query and document embeddings
        Q = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=torch.float32)
        
        D = [torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Perfect match with Q[0]
            [0.0, 0.0, 0.0, 0.0],  # No match
            [0.0, 0.0, 0.5, 0.5],  # Partial match with Q[2]
        ], dtype=torch.float32)]
        
        # Normalize tensors
        Q = Q / torch.norm(Q, dim=1, keepdim=True)
        D = [d / torch.norm(d, dim=1, keepdim=True) for d in D]
        
        # Calculate scores
        scores = self.model.score(Q, D)
        
        # Check shape
        self.assertEqual(scores.shape, torch.Size([1]))
        
        # Calculate expected score manually:
        # For Q[0]: max_sim = 1.0 (D[0][0])
        # For Q[1]: max_sim = 0.0 (no match)
        # For Q[2]: max_sim = 0.5 (D[0][2])
        # Total: 1.0 + 0.0 + 0.5 = 1.5
        expected_score = 1.5
        self.assertAlmostEqual(scores[0].item(), expected_score, places=5)
    
    def test_score_multiple(self):
        """Test the score method with multiple documents."""
        # Create query and document embeddings
        Q = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=torch.float32)
        
        D = [
            torch.tensor([
                [1.0, 0.0, 0.0, 0.0],  # Perfect match with Q[0]
                [0.0, 0.0, 0.0, 0.0],  # No match
            ], dtype=torch.float32),
            torch.tensor([
                [0.0, 0.0, 0.0, 0.0],  # No match
                [0.0, 1.0, 0.0, 0.0],  # Perfect match with Q[1]
            ], dtype=torch.float32),
            torch.tensor([
                [0.7, 0.7, 0.0, 0.0],  # Partial match with Q[0] and Q[1]
            ], dtype=torch.float32)
        ]
        
        # Normalize tensors
        Q = Q / torch.norm(Q, dim=1, keepdim=True)
        D = [d / torch.norm(d, dim=1, keepdim=True) for d in D]
        
        # Calculate scores
        scores = self.model.score(Q, D)
        
        # Check shape
        self.assertEqual(scores.shape, torch.Size([3]))
        
        # Calculate expected scores manually
        # Doc 1: max_sim(Q[0], D1) = 1.0, max_sim(Q[1], D1) = 0.0, total = 1.0
        # Doc 2: max_sim(Q[0], D2) = 0.0, max_sim(Q[1], D2) = 1.0, total = 1.0
        # Doc 3: max_sim(Q[0], D3) = 0.7*0.7, max_sim(Q[1], D3) = 0.7*0.7, total = 0.98
        expected_scores = [1.0, 1.0, 0.98]
        for i, expected in enumerate(expected_scores):
            self.assertAlmostEqual(scores[i].item(), expected, places=5)
    
    def test_score_empty(self):
        """Test the score method with empty/zero documents."""
        # Create query embeddings
        Q = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=torch.float32)
        Q = Q / torch.norm(Q, dim=1, keepdim=True)
        
        # Empty document list
        empty_D = []
        with self.assertRaises(RuntimeError):
            # Should raise RuntimeError due to empty list in pad_sequence
            self.model.score(Q, empty_D)
        
        # Zero embeddings
        zero_D = [torch.zeros(3, 4)]
        # This should not raise an exception now that we handle zeros properly
        scores = self.model.score(Q, zero_D)
        # Should have zero similarity
        self.assertAlmostEqual(scores[0].item(), 0.0, places=5)
    
    def test_embeddings_to_numpy(self):
        """Test conversion from embeddings to numpy."""
        # Create a tensor
        embeddings = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ], dtype=torch.float32)
        
        # Convert to numpy
        np_embeddings = self.model._embeddings_to_numpy(embeddings)
        
        # Check type
        self.assertIsInstance(np_embeddings, np.ndarray)
        
        # Check values
        np.testing.assert_array_equal(
            np_embeddings,
            np.array([
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ], dtype=np.float32)
        )
    
    def test_numpy_to_embeddings(self):
        """Test conversion from numpy to embeddings."""
        # Create a numpy array
        array = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ], dtype=np.float32)
        
        # Convert to tensor
        embeddings = self.model._numpy_to_embeddings(array)
        
        # Check type
        self.assertIsInstance(embeddings, torch.Tensor)
        
        # Check values
        torch.testing.assert_close(
            embeddings,
            torch.tensor([
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ], dtype=torch.float32)
        )
    
    def test_supports_images_default(self):
        """Test that supports_images property defaults to False."""
        self.assertFalse(self.model.supports_images)
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.backends.mps.is_available')
    def test_get_optimal_device(self, mock_mps_available, mock_cuda_count, mock_cuda_available):
        """Test device selection logic with different configurations."""
        # Test case 1: CUDA available with multiple GPUs
        mock_cuda_available.return_value = True
        mock_cuda_count.return_value = 2
        mock_mps_available.return_value = False
        
        device = LateInteractionModel._get_optimal_device()
        self.assertEqual(device, "auto")
        
        # Test case 2: CUDA available with single GPU
        mock_cuda_available.return_value = True
        mock_cuda_count.return_value = 1
        mock_mps_available.return_value = False
        
        device = LateInteractionModel._get_optimal_device()
        self.assertEqual(device, "cuda")
        
        # Test case 3: MPS available, no CUDA
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        
        # Need to mock has_mps attribute
        with patch.object(torch, 'has_mps', True, create=True):
            device = LateInteractionModel._get_optimal_device()
            self.assertEqual(device, "mps")
        
        # Test case 4: No accelerators available
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        
        device = LateInteractionModel._get_optimal_device()
        self.assertEqual(device, "cpu")
        
        # Test case 5: User-specified device
        device = LateInteractionModel._get_optimal_device("cuda:1")
        self.assertEqual(device, "cuda:1")
        
    def test_score_with_padding(self):
        """Test the score method handles documents of different lengths."""
        # Create query embeddings
        Q = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=torch.float32)
        
        # Create documents with different lengths
        D = [
            torch.tensor([  # 3 tokens
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ], dtype=torch.float32),
            torch.tensor([  # 1 token
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        ]
        
        # Normalize tensors
        Q = Q / torch.norm(Q, dim=1, keepdim=True)
        D = [d / torch.norm(d, dim=1, keepdim=True) for d in D]
        
        # Calculate scores
        scores = self.model.score(Q, D)
        
        # Check shape
        self.assertEqual(scores.shape, torch.Size([2]))
        
        # Expected scores:
        # Doc 1: max_sim(Q[0], D1) = 1.0, max_sim(Q[1], D1) = 1.0, total = 2.0
        # Doc 2: max_sim(Q[0], D2) = 1.0, max_sim(Q[1], D2) = 0.0, total = 1.0
        self.assertAlmostEqual(scores[0].item(), 2.0, places=5)
        self.assertAlmostEqual(scores[1].item(), 1.0, places=5)
        
    def test_get_actual_device(self):
        """Test getting the actual device from a PyTorch module."""
        # Create a small test model
        test_model = torch.nn.Linear(10, 5)
        
        # CPU device case
        test_model = test_model.to("cpu")
        device = LateInteractionModel._get_actual_device(test_model)
        self.assertEqual(device.type, "cpu")
        
        # Skip GPU tests if not available to maintain test isolation
        if torch.cuda.is_available():
            # CUDA device case
            test_model = test_model.to("cuda")
            device = LateInteractionModel._get_actual_device(test_model)
            self.assertEqual(device.type, "cuda")
            
    def test_score_with_nan_inf(self):
        """Test the score method handles NaN and Inf values gracefully."""
        # Create query embeddings
        Q = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=torch.float32)
        Q = Q / torch.norm(Q, dim=1, keepdim=True)
        
        # Create document embeddings with NaN
        D_nan = [torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, float('nan'), 0.0],  # Contains NaN
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=torch.float32)]
        
        # Normalize tensor while preserving NaN
        D_nan[0][0:1] = D_nan[0][0:1] / torch.norm(D_nan[0][0:1], dim=1, keepdim=True)
        
        # Create document embeddings with Inf
        D_inf = [torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, float('inf'), 0.0],  # Contains Inf
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=torch.float32)]
        
        # Normalize tensor while preserving Inf
        D_inf[0][0:1] = D_inf[0][0:1] / torch.norm(D_inf[0][0:1], dim=1, keepdim=True)
        
        # Create document embeddings with both NaN and Inf
        D_mixed = [torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, float('nan'), 0.0, 0.0],  # Contains NaN
            [0.0, 0.0, float('inf'), 0.0],  # Contains Inf
        ], dtype=torch.float32)]
        
        # Normalize the first row only to preserve NaN and Inf
        D_mixed[0][0:1] = D_mixed[0][0:1] / torch.norm(D_mixed[0][0:1], dim=1, keepdim=True)
        
        # Test with NaN values
        # The operation should not crash, and the result might contain NaN
        try:
            scores_nan = self.model.score(Q, D_nan)
            self.assertEqual(scores_nan.shape, torch.Size([1]))
            
            # At a minimum, the result should be a valid tensor without raising an exception
            self.assertTrue(torch.is_tensor(scores_nan))
            
            # Some PyTorch operations may propagate NaN, others might zero them out
            # We just verify the code doesn't crash, whichever behavior it implements
        except Exception as e:
            self.fail(f"score() raised {type(e).__name__} with NaN values: {str(e)}")
        
        # Test with Inf values
        try:
            scores_inf = self.model.score(Q, D_inf)
            self.assertEqual(scores_inf.shape, torch.Size([1]))
            self.assertTrue(torch.is_tensor(scores_inf))
            
            # Check specific handling of Inf based on PyTorch's default behavior
            # If the model has special handling for Inf that converts it to a finite value,
            # we should get a reasonable score
            if not torch.isnan(scores_inf[0]) and not torch.isinf(scores_inf[0]):
                self.assertTrue(scores_inf[0].item() >= 0.0)
        except Exception as e:
            self.fail(f"score() raised {type(e).__name__} with Inf values: {str(e)}")
            
        # Test with mixed NaN and Inf values
        try:
            scores_mixed = self.model.score(Q, D_mixed)
            self.assertEqual(scores_mixed.shape, torch.Size([1]))
            self.assertTrue(torch.is_tensor(scores_mixed))
        except Exception as e:
            self.fail(f"score() raised {type(e).__name__} with mixed NaN/Inf values: {str(e)}")
            
        # Test query tensor with NaN/Inf
        Q_invalid = torch.tensor([
            [1.0, float('nan'), 0.0, 0.0],
            [0.0, 1.0, float('inf'), 0.0],
        ], dtype=torch.float32)
        
        # Valid document embeddings
        D_valid = [torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=torch.float32)]
        D_valid[0] = D_valid[0] / torch.norm(D_valid[0], dim=1, keepdim=True)
        
        # Test with invalid query embeddings
        try:
            scores_invalid_query = self.model.score(Q_invalid, D_valid)
            self.assertTrue(torch.is_tensor(scores_invalid_query))
        except Exception as e:
            # Some PyTorch operations might raise exceptions with NaN/Inf in specific contexts
            # If that happens, we just verify it's a known numerical error type
            self.assertIn(type(e).__name__, ["RuntimeError", "ValueError"], 
                          f"Unexpected exception type: {type(e).__name__}")
            
            # Check that the error message is related to numerical issues
            error_msg = str(e).lower()
            has_numerical_terms = any(term in error_msg for term in 
                                      ["nan", "inf", "numerical", "valid"])
            self.assertTrue(has_numerical_terms, 
                            f"Error doesn't seem related to numerical issues: {error_msg}")


if __name__ == "__main__":
    unittest.main()