#!/usr/bin/env python3
"""
Unit tests for late_interaction utility functions.
"""

import unittest
import torch
import numpy as np

from src.astra_multivector.late_interaction.utils import (
    expand_parameter,
    pool_query_embeddings,
    pool_doc_embeddings
)


class TestExpandParameter(unittest.TestCase):
    """Tests for the expand_parameter utility function."""

    def test_expand_parameter_basic(self):
        """Test basic parameter expansion functionality."""
        # f(x) = a + b*x + c*x*log(x)
        a, b, c = 10, 2, 0.5
        
        # Test with various inputs
        self.assertEqual(expand_parameter(1, a, b, c), 12)  # 10 + 2*1 + 0.5*1*log(1) = 12
        self.assertEqual(expand_parameter(10, a, b, c), 31)  # 10 + 2*10 + 0.5*10*log(10) = 31.5 -> 31
        self.assertEqual(expand_parameter(100, a, b, c), 240)  # 10 + 2*100 + 0.5*100*log(100) = 240.6 -> 240
    
    def test_expand_parameter_edge_cases(self):
        """Test parameter expansion with edge cases."""
        # Test with zero and negative inputs
        self.assertEqual(expand_parameter(0, 10, 2, 0.5), 0)
        self.assertEqual(expand_parameter(-5, 10, 2, 0.5), 0)
        
        # Test with negative coefficients
        self.assertEqual(expand_parameter(10, -10, 2, 0.5), 11)  # max(10, -10 + 2*10 + 0.5*10*log(10)) = max(10, 11.5) = 11

    def test_expand_parameter_typical_usage(self):
        """Test parameter expansion with values typical for search parameters."""
        # Values from the late_interaction_pipeline.py
        # f(1) = 105, f(10) = 171, f(100) = 514, f(500) = 998
        tokens_coef = (94.9, 11.0, -1.48)
        
        self.assertEqual(expand_parameter(1, *tokens_coef), 106)
        self.assertEqual(expand_parameter(10, *tokens_coef), 171)
        self.assertEqual(expand_parameter(100, *tokens_coef), 513)
        self.assertEqual(expand_parameter(500, *tokens_coef), 992)

        # f(1) = 9, f(10) = 20, f(100) = 119, f(900) = 1000
        candidates_coef = (8.82, 1.13, -0.00471)
        
        self.assertEqual(expand_parameter(1, *candidates_coef), 10)
        self.assertEqual(expand_parameter(10, *candidates_coef), 20)
        self.assertEqual(expand_parameter(100, *candidates_coef), 119)
        self.assertEqual(expand_parameter(900, *candidates_coef), 998)


class TestPoolQueryEmbeddings(unittest.TestCase):
    """Tests for the pool_query_embeddings function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create 10 embeddings of dimension 4
        # Make some of them similar to test pooling
        self.embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.98, 0.1, 0.1, 0.1],  # Similar to first
            [0.96, 0.2, 0.1, 0.1],  # Similar to first
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.98, 0.1, 0.1],  # Similar to fourth
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.2, 0.2, 0.2, 0.9],   # Similar to seventh
            [0.5, 0.5, 0.5, 0.5],
            [0.4, 0.6, 0.4, 0.4],   # Somewhat similar to ninth
        ], dtype=torch.float32)
        
        # Normalize embeddings
        self.embeddings = self.embeddings / torch.norm(self.embeddings, dim=1, keepdim=True)
    
    def test_pool_query_embeddings_disabled(self):
        """Test query pooling when disabled (max_distance=0)."""
        pooled = pool_query_embeddings(self.embeddings, max_distance=0)
        self.assertEqual(pooled.shape, self.embeddings.shape)
        torch.testing.assert_close(pooled, self.embeddings)
    
    def test_pool_query_embeddings_high_threshold(self):
        """Test query pooling with high threshold (should pool similar embeddings)."""
        # Use high threshold to pool very similar embeddings
        pooled = pool_query_embeddings(self.embeddings, max_distance=0.05)
        
        # Should pool [0,1,2], [3,4], [6,7], and leave [5], [8,9] separate
        # Total should be 6 embeddings
        self.assertEqual(pooled.shape[0], 6)
    
    def test_pool_query_embeddings_low_threshold(self):
        """Test query pooling with low threshold (minimal pooling)."""
        # Use low threshold to only pool very similar embeddings
        pooled = pool_query_embeddings(self.embeddings, max_distance=0.01)
        
        # Should only pool extremely similar embeddings
        # Expect minimal pooling, almost all embeddings preserved
        self.assertGreaterEqual(pooled.shape[0], 9)
    
    def test_pool_query_embeddings_normalized(self):
        """Test that pooled embeddings remain normalized."""
        pooled = pool_query_embeddings(self.embeddings, max_distance=0.1)
        
        # Check that all vectors still have unit norm (are normalized)
        norms = torch.norm(pooled, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)


class TestPoolDocEmbeddings(unittest.TestCase):
    """Tests for the pool_doc_embeddings function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create document embeddings: tensor of size [50, 4]
        self.doc_embeddings = torch.randn(50, 4)
        # Normalize embeddings
        self.doc_embeddings = self.doc_embeddings / torch.norm(self.doc_embeddings, dim=1, keepdim=True)
        
        # Create list of document embeddings
        self.doc_list = [
            torch.randn(30, 4),  # First document: 30 tokens
            torch.randn(20, 4),  # Second document: 20 tokens
        ]
        # Normalize embeddings in list
        self.doc_list = [d / torch.norm(d, dim=1, keepdim=True) for d in self.doc_list]
    
    def test_pool_doc_embeddings_single_doc_disabled(self):
        """Test document pooling is disabled when pool_factor <= 1."""
        # Should return input unchanged
        pooled = pool_doc_embeddings(self.doc_embeddings, pool_factor=1)
        torch.testing.assert_close(pooled, self.doc_embeddings)
    
    def test_pool_doc_embeddings_single_doc(self):
        """Test document pooling for a single document."""
        # Pool by factor of 2 (should reduce to ~25 tokens)
        pooled = pool_doc_embeddings(self.doc_embeddings, pool_factor=2)
        
        # Check that size is reduced
        self.assertLess(pooled.shape[0], self.doc_embeddings.shape[0])
        # Number of tokens should be approximately original / pool_factor
        self.assertAlmostEqual(pooled.shape[0], self.doc_embeddings.shape[0] / 2, delta=5)
    
    def test_pool_doc_embeddings_list(self):
        """Test document pooling for a list of documents."""
        # Pool by factor of 2
        pooled_list = pool_doc_embeddings(self.doc_list, pool_factor=2)
        
        # Check that we still have a list of the same length
        self.assertEqual(len(pooled_list), len(self.doc_list))
        
        # Check that each document is pooled
        for i, (original, pooled) in enumerate(zip(self.doc_list, pooled_list)):
            # Check that size is reduced
            self.assertLess(pooled.shape[0], original.shape[0])
            # Number of tokens should be approximately original / pool_factor
            self.assertAlmostEqual(pooled.shape[0], original.shape[0] / 2, delta=3)
    
    def test_pool_doc_embeddings_large_factor(self):
        """Test document pooling with a large pool factor."""
        # Pool by factor of 10 (should reduce to ~5 tokens)
        pooled = pool_doc_embeddings(self.doc_embeddings, pool_factor=10)
        
        # Check that size is significantly reduced
        self.assertLess(pooled.shape[0], 10)
        # Number of tokens should be approximately original / pool_factor
        self.assertAlmostEqual(pooled.shape[0], self.doc_embeddings.shape[0] / 10, delta=3)


if __name__ == "__main__":
    unittest.main()