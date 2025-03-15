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
    
    def __init__(self, dim=4):
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
        with self.assertRaises(Exception):
            # Should raise exception with empty document list
            self.model.score(Q, empty_D)
        
        # Zero embeddings
        zero_D = [torch.zeros(3, 4)]
        with self.assertRaises(Exception):
            # Should raise exception with zero document embeddings
            self.model.score(Q, zero_D)
    
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


if __name__ == "__main__":
    unittest.main()