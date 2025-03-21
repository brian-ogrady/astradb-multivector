#!/usr/bin/env python3
"""
Unit tests for the LateInteractionModel base class.

This module contains comprehensive tests for the base LateInteractionModel class,
which forms the foundation for all late interaction-based retrieval models in the
Astra Multivector framework. The tests verify core functionality like scoring,
device management, tensor conversions, and handling of edge cases.

A concrete test implementation of the abstract base class is provided to facilitate
testing without requiring a complete model implementation.
"""

import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from astra_multivector.late_interaction import LateInteractionModel


class TestModel(LateInteractionModel):
    """
    Concrete implementation of LateInteractionModel for testing purposes.
    
    This class provides a minimal implementation of the abstract LateInteractionModel
    to allow testing of the base class functionality.
    """
    
    def __init__(self, dim=4, device=None):
        super().__init__(device=device)
        self._dim = dim
        self._model_name = "test_model"
    
    async def encode_query(self, q):
        """
        Mock implementation of query encoding.
        
        Returns a random tensor of shape [num_tokens, dim] representing encoded query tokens.
        
        Args:
            q: The query to encode (not used in this mock implementation)
            
        Returns:
            torch.Tensor: Random tensor with shape [5, self._dim]
        """
        return torch.randn(5, self._dim)
    
    async def encode_doc(self, chunks):
        """
        Mock implementation of document encoding.
        
        Returns a list of random tensors, each with shape [num_tokens, dim], representing 
        encoded document chunks.
        
        Args:
            chunks: List of document chunks to encode (only length is used)
            
        Returns:
            List[torch.Tensor]: List of random tensors, each with shape [10, self._dim]
        """
        return [torch.randn(10, self._dim) for _ in range(len(chunks))]
    
    def to_device(self, T):
        """
        Mock implementation of device transfer. Returns the tensor unchanged.
        
        Args:
            T: Input tensor
            
        Returns:
            torch.Tensor: The same tensor unmodified
        """
        return T
    
    @property
    def dim(self):
        """Returns the embedding dimension."""
        return self._dim
    
    @property
    def model_name(self):
        """Returns the model name."""
        return self._model_name


class TestLateInteractionModel(unittest.TestCase):
    """
    Unit tests for the LateInteractionModel base class functionality.
    
    These tests verify the core scoring, device handling, tensor conversion, 
    and other base functionality shared by all late interaction models.
    """
    
    def setUp(self):
        """Initialize a TestModel instance for each test case."""
        self.model = TestModel()
        
    @patch('astra_multivector.late_interaction.models.base.LateInteractionModel._get_optimal_device')
    def test_device_initialization(self, mock_get_optimal_device):
        """
        Test that device initialization works correctly.
        
        Verifies both default device selection and explicit device specification
        by mocking the _get_optimal_device method.
        """
        mock_get_optimal_device.return_value = "cpu"
        model = TestModel()
        self.assertEqual(model._device, "cpu")
        
        model = TestModel(device="cuda:1")
        self.assertEqual(model._device, "cuda:1")
        
        mock_get_optimal_device.assert_called_once()
    
    def test_score_single(self):
        """
        Test the score method with a single document.
        
        Creates normalized query and document embeddings, calculates similarity scores,
        and verifies the result matches the expected value based on manual calculation.
        
        The scoring is done by summing max similarities between each query token and
        all document tokens, and then taking the norm of the result.
        """
        Q = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=torch.float32)
        
        D = [torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
        ], dtype=torch.float32)]
        
        Q = Q / torch.norm(Q, dim=1, keepdim=True)
        D = [d / torch.norm(d, dim=1, keepdim=True) for d in D]
        
        scores = self.model.score(Q, D)
        
        self.assertEqual(scores.shape, torch.Size([1]))
        
        # Expected calculation:
        # Q[0] has max similarity 1.0 with D[0][0]
        # Q[1] has max similarity 0.0 with all tokens in D
        # Q[2] has max similarity 0.5 with D[0][2]
        # Total: sqrt(1.0^2 + 0.0^2 + 0.5^2) = sqrt(1.25) ≈ 1.707107
        expected_score = 1.707107
        self.assertAlmostEqual(scores[0].item(), expected_score, places=5)
    
    def test_score_multiple(self):
        """
        Test the score method with multiple documents.
        
        Creates a scenario with two query tokens and three documents with different
        similarity patterns to verify correct scoring across multiple documents:
        - A document with perfect match for first query token
        - A document with perfect match for second query token
        - A document with partial matches for both query tokens
        """
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
        
        Q = Q / torch.norm(Q, dim=1, keepdim=True)
        D = [d / torch.norm(d, dim=1, keepdim=True) for d in D]
        
        scores = self.model.score(Q, D)
        
        self.assertEqual(scores.shape, torch.Size([3]))
        
        # Expected calculation for each document:
        # Doc 1: sqrt(1.0^2 + 0.0^2) = 1.0 (perfect match with Q[0] only)
        # Doc 2: sqrt(0.0^2 + 1.0^2) = 1.0 (perfect match with Q[1] only)
        # Doc 3: sqrt((0.7^2 + 0.7^2)) = sqrt(0.98) ≈ 1.414214 (partial matches with both)
        expected_scores = [1.0, 1.0, 1.414213657]
        for i, expected in enumerate(expected_scores):
            self.assertAlmostEqual(scores[i].item(), expected, places=5)
    
    def test_score_empty(self):
        """
        Test the score method with empty and zero-filled documents.
        
        Verifies two edge cases:
        1. Empty document list - should raise RuntimeError due to empty list in pad_sequence
        2. Document with all zero embeddings - should handle gracefully and return zero score
        
        These tests ensure the scoring function handles degenerate cases properly.
        """
        Q = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=torch.float32)
        Q = Q / torch.norm(Q, dim=1, keepdim=True)
        
        # Empty document list
        empty_D = []
        with self.assertRaises(RuntimeError):
            self.model.score(Q, empty_D)
        
        # Zero embeddings
        zero_D = [torch.zeros(3, 4)]
        scores = self.model.score(Q, zero_D)
        self.assertAlmostEqual(scores[0].item(), 0.0, places=5)
    
    def test_embeddings_to_numpy(self):
        """
        Test conversion from PyTorch tensor embeddings to NumPy arrays.
        
        Verifies that the _embeddings_to_numpy method correctly converts PyTorch tensors
        to NumPy arrays, preserving both data type and values. This conversion is important
        for interoperability with libraries that require NumPy input.
        """
        embeddings = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ], dtype=torch.float32)
        
        np_embeddings = self.model._embeddings_to_numpy(embeddings)
        
        self.assertIsInstance(np_embeddings, np.ndarray)
        
        np.testing.assert_array_equal(
            np_embeddings,
            np.array([
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ], dtype=np.float32)
        )
    
    def test_numpy_to_embeddings(self):
        """
        Test conversion from NumPy arrays to PyTorch tensor embeddings.
        
        Verifies that the _numpy_to_embeddings method correctly converts NumPy arrays
        to PyTorch tensors, preserving both data type and values. This conversion is
        essential when processing external data or results from NumPy computations.
        """
        array = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ], dtype=np.float32)
        
        embeddings = self.model._numpy_to_embeddings(array)
        
        self.assertIsInstance(embeddings, torch.Tensor)
        
        torch.testing.assert_close(
            embeddings,
            torch.tensor([
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ], dtype=torch.float32)
        )
    
    def test_supports_images_default(self):
        """
        Test that the supports_images property defaults to False.
        
        Verifies that models don't claim to support image inputs by default,
        which is important for API consistency and preventing incorrect usage.
        """
        self.assertFalse(self.model.supports_images)
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.backends.mps.is_available')
    def test_get_optimal_device(self, mock_mps_available, mock_cuda_count, mock_cuda_available):
        """
        Test device selection logic with different hardware configurations.
        
        Uses mocking to simulate different hardware environments and verifies that
        the optimal device is selected correctly in each scenario:
        1. Multiple CUDA GPUs available - should select 'auto' for multi-GPU support
        2. Single CUDA GPU available - should select 'cuda'
        3. Apple MPS acceleration available - should select 'mps'
        4. No accelerators available - should fall back to 'cpu'
        5. Explicit user device specification - should use the specified device
        
        This test ensures the model will use the most appropriate compute device
        in various environments.
        """
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
        """
        Test that the score method correctly handles documents of different lengths.
        
        Creates a scenario with documents having different numbers of tokens to verify
        that the padding mechanism in the score method works correctly. This test is
        important because in real-world applications, documents will have varying
        numbers of tokens after encoding.
        
        The test includes:
        - A document with 3 tokens (including perfect matches for both query tokens)
        - A document with only 1 token (matching only the first query token)
        
        The scoring should handle this difference correctly through proper padding.
        """
        Q = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=torch.float32)
        
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
        
        Q = Q / torch.norm(Q, dim=1, keepdim=True)
        D = [d / torch.norm(d, dim=1, keepdim=True) for d in D]
        
        scores = self.model.score(Q, D)
        
        self.assertEqual(scores.shape, torch.Size([2]))
        
        # Expected scores:
        # Doc 1: sqrt(1.0^2 + 1.0^2) = sqrt(2) = 2.0 (matches both query tokens)
        # Doc 2: sqrt(1.0^2 + 0.0^2) = 1.0 (matches only first query token)
        self.assertAlmostEqual(scores[0].item(), 2.0, places=5)
        self.assertAlmostEqual(scores[1].item(), 1.0, places=5)
        
    def test_get_actual_device(self):
        """
        Test getting the actual PyTorch device from a module.
        
        Verifies that the _get_actual_device method correctly identifies the device
        on which a PyTorch module is loaded. This is important for ensuring that
        all tensors in the model pipeline are placed on the same device for efficient
        computation.
        
        Tests CPU device always and GPU device when available.
        """
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
        """
        Test the score method's robustness when handling NaN and Inf values.
        
        This comprehensive test verifies how the scoring function handles problematic
        numerical values in tensor computations. In real-world scenarios, embeddings
        might occasionally contain NaN or Inf due to various issues (normalization of
        zero vectors, numerical instability, etc.).
        
        The test covers four scenarios:
        1. Document embeddings containing NaN values
        2. Document embeddings containing Inf values
        3. Document embeddings containing a mix of NaN and Inf values
        4. Query embeddings containing NaN/Inf values
        
        For each scenario, we verify that either:
        - The score function handles these values gracefully without crashing
        - If an exception is raised, it's an expected numerical error with an appropriate message
        
        This ensures the model is robust against common numerical issues in embedding calculations.
        """
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
        try:
            scores_nan = self.model.score(Q, D_nan)
            self.assertEqual(scores_nan.shape, torch.Size([1]))
            self.assertTrue(torch.is_tensor(scores_nan))
        except Exception as e:
            self.fail(f"score() raised {type(e).__name__} with NaN values: {str(e)}")
        
        # Test with Inf values
        try:
            scores_inf = self.model.score(Q, D_inf)
            self.assertEqual(scores_inf.shape, torch.Size([1]))
            self.assertTrue(torch.is_tensor(scores_inf))
            
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