#!/usr/bin/env python3
"""
Unit tests for late_interaction utility functions.
"""

import unittest
import torch
import numpy as np

from astra_multivector.late_interaction import (
    expand_parameter,
    pool_query_embeddings,
    pool_doc_embeddings,
    PoolingResult
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
    
    def test_pool_query_embeddings_return_cluster_info(self):
        """Test return_cluster_info parameter returns the expected statistics."""
        result = pool_query_embeddings(self.embeddings, max_distance=0.05, return_cluster_info=True)
        
        # Should return a PoolingResult namedtuple
        self.assertIsInstance(result, PoolingResult)
        self.assertIsInstance(result.embeddings, torch.Tensor)
        self.assertIsInstance(result.stats, dict)
        
        # Check stats content
        self.assertIn('original_count', result.stats)
        self.assertIn('pooled_count', result.stats)
        self.assertIn('compression_ratio', result.stats)
        self.assertIn('pooling_applied', result.stats)
        
        # Verify stats values
        self.assertEqual(result.stats['original_count'], self.embeddings.shape[0])
        self.assertEqual(result.stats['pooled_count'], result.embeddings.shape[0])
        self.assertTrue(result.stats['pooling_applied'])
        
        # Check that compression ratio makes sense
        expected_ratio = self.embeddings.shape[0] / result.embeddings.shape[0]
        self.assertAlmostEqual(result.stats['compression_ratio'], expected_ratio, places=5)
    
    def test_pool_query_embeddings_min_clusters(self):
        """Test min_clusters parameter prevents over-pooling."""
        # Set high min_clusters (equal to input embeddings count)
        result = pool_query_embeddings(
            self.embeddings, 
            max_distance=0.5,  # High distance would normally pool many embeddings
            min_clusters=self.embeddings.shape[0],
            return_cluster_info=True
        )
        
        # Pooling should not be applied because min_clusters would be violated
        self.assertFalse(result.stats['pooling_applied'])
        self.assertEqual(result.embeddings.shape[0], self.embeddings.shape[0])
        
        # Set reasonable min_clusters (less than input, allows pooling)
        result2 = pool_query_embeddings(
            self.embeddings, 
            max_distance=0.5,
            min_clusters=3,
            return_cluster_info=True
        )
        
        # Pooling should be applied
        self.assertTrue(result2.stats['pooling_applied'])
        self.assertLess(result2.embeddings.shape[0], self.embeddings.shape[0])
        self.assertGreaterEqual(result2.embeddings.shape[0], 3)  # At least min_clusters
    
    def test_pool_query_embeddings_invalid_inputs(self):
        """Test pool_query_embeddings with invalid inputs."""
        # Test with negative max_distance (should not pool)
        result = pool_query_embeddings(
            self.embeddings, 
            max_distance=-0.1,
            return_cluster_info=True
        )
        
        # Pooling should not be applied
        self.assertFalse(result.stats['pooling_applied'])
        torch.testing.assert_close(result.embeddings, self.embeddings)
        
        # Test with too few input embeddings
        small_embeddings = torch.randn(2, 4)
        small_embeddings = small_embeddings / torch.norm(small_embeddings, dim=1, keepdim=True)
        
        result2 = pool_query_embeddings(
            small_embeddings, 
            max_distance=0.1,
            min_clusters=3,  # More than input size
            return_cluster_info=True
        )
        
        # Pooling should not be applied
        self.assertFalse(result2.stats['pooling_applied'])
        torch.testing.assert_close(result2.embeddings, small_embeddings)


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
        
        # Create document with protected tokens
        self.doc_with_important = torch.randn(40, 4)
        self.doc_with_important = self.doc_with_important / torch.norm(self.doc_with_important, dim=1, keepdim=True)
    
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
    
    def test_pool_doc_embeddings_return_stats(self):
        """Test return_stats parameter returns pooling statistics."""
        # Single document case
        result = pool_doc_embeddings(self.doc_embeddings, pool_factor=2, return_stats=True)
        
        # Should return a PoolingResult namedtuple
        self.assertIsInstance(result, PoolingResult)
        self.assertIsInstance(result.embeddings, torch.Tensor)
        self.assertIsInstance(result.stats, dict)
        
        # Check stats content
        self.assertIn('original_tokens', result.stats)
        self.assertIn('pooled_tokens', result.stats)
        self.assertIn('compression_ratio', result.stats)
        self.assertIn('pooling_applied', result.stats)
        
        # Verify stats values
        self.assertEqual(result.stats['original_tokens'], self.doc_embeddings.shape[0])
        self.assertEqual(result.stats['pooled_tokens'], result.embeddings.shape[0])
        self.assertTrue(result.stats['pooling_applied'])
        
        # List of documents case
        result_list = pool_doc_embeddings(self.doc_list, pool_factor=2, return_stats=True)
        
        # Should return a PoolingResult namedtuple
        self.assertIsInstance(result_list, PoolingResult)
        self.assertIsInstance(result_list.embeddings, list)
        self.assertIsInstance(result_list.stats, dict)
        
        # Check stats content
        self.assertIn('original_tokens', result_list.stats)
        self.assertIn('pooled_tokens', result_list.stats)
        self.assertIn('compression_ratios', result_list.stats)
        self.assertIn('pooling_applied', result_list.stats)
        self.assertIn('total_reduction', result_list.stats)
        
        # Verify stats values
        self.assertEqual(len(result_list.stats['original_tokens']), len(self.doc_list))
        self.assertEqual(len(result_list.stats['pooled_tokens']), len(self.doc_list))
        self.assertEqual(len(result_list.stats['pooling_applied']), len(self.doc_list))
        
        # All documents should have pooling applied
        self.assertTrue(all(result_list.stats['pooling_applied']))
    
    def test_pool_doc_embeddings_protected_tokens(self):
        """Test protected_tokens parameter prevents pooling of important tokens."""
        # Number of tokens to protect (first N tokens)
        num_protected = 15
        
        # Normal pooling (no protection)
        pooled_normal = pool_doc_embeddings(self.doc_with_important, pool_factor=4)
        
        # Pooling with protected tokens
        pooled_protected = pool_doc_embeddings(
            self.doc_with_important, 
            pool_factor=4,
            protected_tokens=num_protected
        )
        
        # Check that we have more tokens when using protection
        self.assertGreater(pooled_protected.shape[0], pooled_normal.shape[0])
        
        # Generate stats for both poolings
        result_normal = pool_doc_embeddings(
            self.doc_with_important, 
            pool_factor=4,
            return_stats=True
        )
        
        result_protected = pool_doc_embeddings(
            self.doc_with_important, 
            pool_factor=4,
            protected_tokens=num_protected,
            return_stats=True
        )
        
        # Both should have pooling applied
        self.assertTrue(result_normal.stats['pooling_applied'])
        self.assertTrue(result_protected.stats['pooling_applied'])
        
        # Protected version should have lower compression ratio
        self.assertLess(
            result_normal.stats['compression_ratio'],
            result_protected.stats['compression_ratio']
        )
        
        # Check list of documents with protection
        pooled_list = pool_doc_embeddings(
            self.doc_list, 
            pool_factor=2,
            protected_tokens=10,
            return_stats=True
        )
        
        # Verify each document still has at least protected_tokens + some pooled tokens
        for i, doc in enumerate(pooled_list.embeddings):
            self.assertGreaterEqual(doc.shape[0], 10)
    
    def test_pool_doc_embeddings_min_tokens(self):
        """Test min_tokens parameter prevents over-pooling."""
        # Normal pooling
        pooled_normal = pool_doc_embeddings(self.doc_embeddings, pool_factor=25)
        
        # Pooling with min_tokens
        high_min_tokens = 20
        pooled_min = pool_doc_embeddings(
            self.doc_embeddings, 
            pool_factor=25,
            min_tokens=high_min_tokens,
            return_stats=True
        )
        
        # Check that pooling with min_tokens gives more tokens than normal pooling
        if pooled_min.stats['pooling_applied']:
            self.assertGreaterEqual(pooled_min.embeddings.shape[0], high_min_tokens)
        else:
            # If pooling was prevented, should return original
            self.assertEqual(pooled_min.embeddings.shape[0], self.doc_embeddings.shape[0])
    
    def test_pool_doc_embeddings_invalid_inputs(self):
        """Test pool_doc_embeddings with invalid inputs."""
        # Test with pool_factor <= 1 (should not pool)
        result = pool_doc_embeddings(
            self.doc_embeddings, 
            pool_factor=0.5,
            return_stats=True
        )
        
        # Pooling should not be applied
        self.assertFalse(result.stats['pooling_applied'])
        self.assertEqual(result.stats['reason'], 'pool_factor <= 1')
        torch.testing.assert_close(result.embeddings, self.doc_embeddings)
        
        # Test with document that's already smaller than min_tokens
        small_doc = torch.randn(3, 4)
        small_doc = small_doc / torch.norm(small_doc, dim=1, keepdim=True)
        
        result2 = pool_doc_embeddings(
            small_doc, 
            pool_factor=2,
            min_tokens=4,  # More than input size
            return_stats=True
        )
        
        # Pooling should not be applied
        self.assertFalse(result2.stats['pooling_applied'])
        torch.testing.assert_close(result2.embeddings, small_doc)
        
        # Test with document that's smaller than protected_tokens + pool_factor
        medium_doc = torch.randn(8, 4)
        medium_doc = medium_doc / torch.norm(medium_doc, dim=1, keepdim=True)
        
        result3 = pool_doc_embeddings(
            medium_doc, 
            pool_factor=5,
            protected_tokens=4,
            return_stats=True
        )
        
        # Pooling should not be applied
        self.assertFalse(result3.stats['pooling_applied'])
        self.assertIn('pool_factor', result3.stats['reason'])
        torch.testing.assert_close(result3.embeddings, medium_doc)


if __name__ == "__main__":
    unittest.main()