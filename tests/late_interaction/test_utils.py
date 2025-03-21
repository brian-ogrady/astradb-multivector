#!/usr/bin/env python3
"""
Unit tests for late_interaction utility functions.
"""

import unittest
import torch

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
        a, b, c = 10, 2, 0.5
        
        self.assertEqual(expand_parameter(1, a, b, c), 12)
        self.assertEqual(expand_parameter(10, a, b, c), 41)
        self.assertEqual(expand_parameter(100, a, b, c), 440)
    
    def test_expand_parameter_edge_cases(self):
        """Test parameter expansion with edge cases."""
        self.assertEqual(expand_parameter(0, 10, 2, 0.5), 0)
        self.assertEqual(expand_parameter(-5, 10, 2, 0.5), 0)
        
        self.assertEqual(expand_parameter(10, -10, 2, 0.5), 21)

    def test_expand_parameter_typical_usage(self):
        """Test parameter expansion with values typical for search parameters."""
        tokens_coef = (94.9, 11.0, -1.48)
        
        self.assertEqual(expand_parameter(1, *tokens_coef), 105)
        self.assertEqual(expand_parameter(10, *tokens_coef), 170)
        self.assertEqual(expand_parameter(100, *tokens_coef), 513)
        self.assertEqual(expand_parameter(500, *tokens_coef), 996)

        candidates_coef = (8.82, 1.13, -0.00471)
        
        self.assertEqual(expand_parameter(1, *candidates_coef), 9)
        self.assertEqual(expand_parameter(10, *candidates_coef), 20)
        self.assertEqual(expand_parameter(100, *candidates_coef), 119)
        self.assertEqual(expand_parameter(900, *candidates_coef), 996)


class TestPoolQueryEmbeddings(unittest.TestCase):
    """Tests for the pool_query_embeddings function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.98, 0.1, 0.1, 0.1],
            [0.96, 0.2, 0.1, 0.1],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.98, 0.1, 0.1],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.2, 0.2, 0.2, 0.9],
            [0.5, 0.5, 0.5, 0.5],
            [0.4, 0.6, 0.4, 0.4],
        ], dtype=torch.float32)
        
        self.embeddings = self.embeddings / torch.norm(self.embeddings, dim=1, keepdim=True)
    
    def test_pool_query_embeddings_disabled(self):
        """Test query pooling when disabled (max_distance=0)."""
        pooled = pool_query_embeddings(self.embeddings, max_distance=0)
        self.assertEqual(pooled.shape, self.embeddings.shape)
        torch.testing.assert_close(pooled, self.embeddings)
    
    def test_pool_query_embeddings_high_threshold(self):
        """Test query pooling with high threshold (should pool similar embeddings).
        
        With max_distance=0.05, the pooling should merge:
        - Embeddings [0,1,2] (first cluster)
        - Embeddings [3,4] (second cluster)
        - Embeddings [6,7] (third cluster)
        
        While leaving [5], [8,9] separate, resulting in 6 total embeddings.
        """
        pooled = pool_query_embeddings(self.embeddings, max_distance=0.05)
        self.assertEqual(pooled.shape[0], 6)
    
    def test_pool_query_embeddings_low_threshold(self):
        """Test query pooling with low threshold (minimal pooling).
        
        With a very low max_distance=0.01, only extremely similar embeddings
        should be pooled together. We expect minimal pooling to occur,
        preserving at least 9 out of the original 10 embeddings.
        """
        pooled = pool_query_embeddings(self.embeddings, max_distance=0.01)
        self.assertGreaterEqual(pooled.shape[0], 9)
    
    def test_pool_query_embeddings_normalized(self):
        """Test that pooled embeddings remain normalized.
        
        After pooling operations, all resulting embeddings should still have unit norm.
        This ensures that the pooling process preserves the normalization property,
        which is crucial for similarity computations using dot products.
        """
        pooled = pool_query_embeddings(self.embeddings, max_distance=0.1)
        
        norms = torch.norm(pooled, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)
    
    def test_pool_query_embeddings_return_cluster_info(self):
        """Test return_cluster_info parameter returns the expected statistics.
        
        When return_cluster_info=True, the function should return a PoolingResult
        namedtuple containing both the pooled embeddings and statistics about the
        pooling process, including:
        - original_count: Number of embeddings before pooling
        - pooled_count: Number of embeddings after pooling
        - compression_ratio: Ratio of original to pooled embedding count
        - pooling_applied: Boolean indicating if pooling was performed
        """
        result = pool_query_embeddings(self.embeddings, max_distance=0.05, return_cluster_info=True)
        
        self.assertIsInstance(result, PoolingResult)
        self.assertIsInstance(result.embeddings, torch.Tensor)
        self.assertIsInstance(result.stats, dict)
        
        self.assertIn('original_count', result.stats)
        self.assertIn('pooled_count', result.stats)
        self.assertIn('compression_ratio', result.stats)
        self.assertIn('pooling_applied', result.stats)
        
        self.assertEqual(result.stats['original_count'], self.embeddings.shape[0])
        self.assertEqual(result.stats['pooled_count'], result.embeddings.shape[0])
        self.assertTrue(result.stats['pooling_applied'])
        
        expected_ratio = self.embeddings.shape[0] / result.embeddings.shape[0]
        self.assertAlmostEqual(result.stats['compression_ratio'], expected_ratio, places=5)
    
    def test_pool_query_embeddings_min_clusters(self):
        """Test min_clusters parameter prevents over-pooling.
        
        The min_clusters parameter sets a lower bound on the number of clusters
        after pooling. This test verifies:
        1. When min_clusters equals the input embedding count, no pooling occurs
           even with a high max_distance that would normally pool aggressively
        2. When min_clusters is reasonably less than the input count, pooling occurs
           but respects the minimum number of clusters specified
        """
        result = pool_query_embeddings(
            self.embeddings, 
            max_distance=0.5,
            min_clusters=self.embeddings.shape[0],
            return_cluster_info=True
        )
        
        self.assertFalse(result.stats['pooling_applied'])
        self.assertEqual(result.embeddings.shape[0], self.embeddings.shape[0])
        
        result2 = pool_query_embeddings(
            self.embeddings, 
            max_distance=0.5,
            min_clusters=3,
            return_cluster_info=True
        )
        
        self.assertTrue(result2.stats['pooling_applied'])
        self.assertLess(result2.embeddings.shape[0], self.embeddings.shape[0])
        self.assertGreaterEqual(result2.embeddings.shape[0], 3)
    
    def test_pool_query_embeddings_invalid_inputs(self):
        """Test pool_query_embeddings with invalid inputs.
        
        This test verifies that the function handles invalid inputs gracefully:
        1. When max_distance is negative, pooling should not be applied
        2. When the number of input embeddings is too small compared to min_clusters,
           pooling should not be applied
        
        In both cases, the function should return the original embeddings unchanged
        and indicate that pooling was not applied in the stats.
        """
        result = pool_query_embeddings(
            self.embeddings, 
            max_distance=-0.1,
            return_cluster_info=True
        )
        
        self.assertFalse(result.stats['pooling_applied'])
        torch.testing.assert_close(result.embeddings, self.embeddings)
        
        small_embeddings = torch.randn(2, 4)
        small_embeddings = small_embeddings / torch.norm(small_embeddings, dim=1, keepdim=True)
        
        result2 = pool_query_embeddings(
            small_embeddings, 
            max_distance=0.1,
            min_clusters=3,
            return_cluster_info=True
        )
        
        self.assertFalse(result2.stats['pooling_applied'])
        torch.testing.assert_close(result2.embeddings, small_embeddings)


class TestPoolDocEmbeddings(unittest.TestCase):
    """Tests for the pool_doc_embeddings function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.doc_embeddings = torch.randn(50, 4)
        self.doc_embeddings = self.doc_embeddings / torch.norm(self.doc_embeddings, dim=1, keepdim=True)
        
        self.doc_list = [
            torch.randn(30, 4),
            torch.randn(20, 4),
        ]
        self.doc_list = [d / torch.norm(d, dim=1, keepdim=True) for d in self.doc_list]
        
        self.doc_with_important = torch.randn(40, 4)
        self.doc_with_important = self.doc_with_important / torch.norm(self.doc_with_important, dim=1, keepdim=True)
    
    def test_pool_doc_embeddings_single_doc_disabled(self):
        """Test document pooling is disabled when pool_factor <= 1.
        
        When pool_factor is set to 1 or less, no pooling should occur and
        the function should return the original embeddings unchanged.
        """
        pooled = pool_doc_embeddings(self.doc_embeddings, pool_factor=1)
        torch.testing.assert_close(pooled, self.doc_embeddings)
    
    def test_pool_doc_embeddings_single_doc(self):
        """Test document pooling for a single document.
        
        With pool_factor=2, the number of tokens should be reduced to approximately
        half of the original count (50 → ~25 tokens). This test verifies that
        pooling reduces the document size by the expected factor, within a
        reasonable margin of error (delta=5).
        """
        pooled = pool_doc_embeddings(self.doc_embeddings, pool_factor=2)
        
        self.assertLess(pooled.shape[0], self.doc_embeddings.shape[0])
        self.assertAlmostEqual(pooled.shape[0], self.doc_embeddings.shape[0] / 2, delta=5)
    
    def test_pool_doc_embeddings_list(self):
        """Test document pooling for a list of documents.
        
        When given a list of documents, the function should:
        1. Return a list of the same length as the input
        2. Pool each document individually by the specified factor
        3. Preserve the order of documents in the list
        
        This test verifies correct behavior with a list of two documents having
        different token counts (30 and 20), both pooled with factor=2.
        """
        pooled_list = pool_doc_embeddings(self.doc_list, pool_factor=2)
        
        self.assertEqual(len(pooled_list), len(self.doc_list))
        
        for i, (original, pooled) in enumerate(zip(self.doc_list, pooled_list)):
            self.assertLess(pooled.shape[0], original.shape[0])
            self.assertAlmostEqual(pooled.shape[0], original.shape[0] / 2, delta=3)
    
    def test_pool_doc_embeddings_large_factor(self):
        """Test document pooling with a large pool factor.
        
        With a large pool_factor=10, the document size should be dramatically reduced
        (from 50 tokens to ~5 tokens). This tests the behavior of the pooling
        algorithm with aggressive reduction factors, ensuring it still produces
        reasonable results without failing even when asked to perform significant
        dimensionality reduction.
        """
        pooled = pool_doc_embeddings(self.doc_embeddings, pool_factor=10)
        
        self.assertLess(pooled.shape[0], 10)
        self.assertAlmostEqual(pooled.shape[0], self.doc_embeddings.shape[0] / 10, delta=3)
    
    def test_pool_doc_embeddings_return_stats(self):
        """Test return_stats parameter returns pooling statistics.
        
        When return_stats=True, the function should return a PoolingResult namedtuple
        with detailed statistics about the pooling operation. This test verifies:
        
        1. For single document inputs:
           - Returns statistics on original and pooled token counts
           - Provides compression ratio
           - Indicates whether pooling was applied
        
        2. For list document inputs:
           - Returns per-document statistics
           - Includes document-specific compression ratios
           - Provides aggregate statistics across all documents
        """
        result = pool_doc_embeddings(self.doc_embeddings, pool_factor=2, return_stats=True)
        
        self.assertIsInstance(result, PoolingResult)
        self.assertIsInstance(result.embeddings, torch.Tensor)
        self.assertIsInstance(result.stats, dict)
        
        self.assertIn('original_tokens', result.stats)
        self.assertIn('pooled_tokens', result.stats)
        self.assertIn('compression_ratio', result.stats)
        self.assertIn('pooling_applied', result.stats)
        
        self.assertEqual(result.stats['original_tokens'], self.doc_embeddings.shape[0])
        self.assertEqual(result.stats['pooled_tokens'], result.embeddings.shape[0])
        self.assertTrue(result.stats['pooling_applied'])
        
        result_list = pool_doc_embeddings(self.doc_list, pool_factor=2, return_stats=True)
        
        self.assertIsInstance(result_list, PoolingResult)
        self.assertIsInstance(result_list.embeddings, list)
        self.assertIsInstance(result_list.stats, dict)
        
        self.assertIn('original_tokens', result_list.stats)
        self.assertIn('pooled_tokens', result_list.stats)
        self.assertIn('compression_ratios', result_list.stats)
        self.assertIn('pooling_applied', result_list.stats)
        self.assertIn('total_reduction', result_list.stats)
        
        self.assertEqual(len(result_list.stats['original_tokens']), len(self.doc_list))
        self.assertEqual(len(result_list.stats['pooled_tokens']), len(self.doc_list))
        self.assertEqual(len(result_list.stats['pooling_applied']), len(self.doc_list))
        
        self.assertTrue(all(result_list.stats['pooling_applied']))
    
    def test_pool_doc_embeddings_protected_tokens(self):
        """Test protected_tokens parameter prevents pooling of important tokens.
        
        The protected_tokens parameter allows preserving the first N tokens from
        pooling, which is useful for maintaining important tokens (like special
        tokens or key content) in their original form. This test verifies:
        
        1. When using protected_tokens, the resulting document has more tokens
           than when pooling without protection
        2. The compression ratio is lower (less aggressive) with protection
        3. The number of tokens in the result is at least equal to the number
           of protected tokens
        4. Protected tokens works correctly for both single documents and lists
        """
        num_protected = 15
        
        pooled_normal = pool_doc_embeddings(self.doc_with_important, pool_factor=4)
        
        pooled_protected = pool_doc_embeddings(
            self.doc_with_important, 
            pool_factor=4,
            protected_tokens=num_protected
        )
        
        self.assertGreater(pooled_protected.shape[0], pooled_normal.shape[0])
        
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
        
        self.assertTrue(result_normal.stats['pooling_applied'])
        self.assertTrue(result_protected.stats['pooling_applied'])
        
        self.assertGreater(
            result_normal.stats['compression_ratio'],
            result_protected.stats['compression_ratio']
        )
        
        pooled_list = pool_doc_embeddings(
            self.doc_list, 
            pool_factor=2,
            protected_tokens=10,
            return_stats=True
        )
        
        for i, doc in enumerate(pooled_list.embeddings):
            self.assertGreaterEqual(doc.shape[0], 10)
    
    def test_pool_doc_embeddings_min_tokens(self):
        """Test min_tokens parameter prevents over-pooling.
        
        The min_tokens parameter sets a lower bound on the number of tokens
        after pooling, preventing too aggressive pooling. This test verifies:
        
        1. When min_tokens is set, the resulting document contains at least
           that many tokens, even with an aggressive pool_factor
        2. If pooling would result in fewer tokens than min_tokens, either
           pooling is not applied or the result respects the minimum
        """
        pooled_normal = pool_doc_embeddings(self.doc_embeddings, pool_factor=25)
        
        high_min_tokens = 20
        pooled_min = pool_doc_embeddings(
            self.doc_embeddings, 
            pool_factor=25,
            min_tokens=high_min_tokens,
            return_stats=True
        )
        
        if pooled_min.stats['pooling_applied']:
            self.assertGreaterEqual(pooled_min.embeddings.shape[0], high_min_tokens)
        else:
            self.assertEqual(pooled_min.embeddings.shape[0], self.doc_embeddings.shape[0])
    
    def test_pool_doc_embeddings_invalid_inputs(self):
        """Test pool_doc_embeddings with invalid inputs.
        
        This test verifies that the function handles various invalid or
        problematic inputs gracefully:
        
        1. When pool_factor <= 1, no pooling should occur
        2. When document size is already smaller than min_tokens, no pooling should occur
        3. When document size is too small compared to protected_tokens and pool_factor,
           no pooling should occur
        
        In all cases, the function should return the original embeddings unchanged
        and include a reason in the stats explaining why pooling was not applied.
        """
        result = pool_doc_embeddings(
            self.doc_embeddings, 
            pool_factor=0.5,
            return_stats=True
        )
        
        self.assertFalse(result.stats['pooling_applied'])
        self.assertEqual(result.stats['reason'], 'pool_factor <= 1')
        torch.testing.assert_close(result.embeddings, self.doc_embeddings)
        
        small_doc = torch.randn(3, 4)
        small_doc = small_doc / torch.norm(small_doc, dim=1, keepdim=True)
        
        result2 = pool_doc_embeddings(
            small_doc, 
            pool_factor=2,
            min_tokens=4,
            return_stats=True
        )
        
        self.assertFalse(result2.stats['pooling_applied'])
        torch.testing.assert_close(result2.embeddings, small_doc)
        
        medium_doc = torch.randn(8, 4)
        medium_doc = medium_doc / torch.norm(medium_doc, dim=1, keepdim=True)
        
        result3 = pool_doc_embeddings(
            medium_doc, 
            pool_factor=5,
            protected_tokens=4,
            return_stats=True
        )
        
        self.assertFalse(result3.stats['pooling_applied'])
        self.assertIn('pool_factor', result3.stats['reason'])
        torch.testing.assert_close(result3.embeddings, medium_doc)


if __name__ == "__main__":
    unittest.main()
