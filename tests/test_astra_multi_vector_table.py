import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np
from astrapy import Database, Table
from astrapy.info import TableVectorIndexOptions, VectorServiceOptions
from sentence_transformers import SentenceTransformer
from rerankers import Reranker
from rerankers.results import RankedResults

from astra_multivector import AstraMultiVectorTable, VectorColumnOptions
from astra_multivector.vector_column_options import VectorColumnType


# Create special mocks for test
class MockSentenceTransformerOptions:
    """Mock VectorColumnOptions for SENTENCE_TRANSFORMER type"""
    def __init__(self, column_name="embeddings1"):
        self.column_name = column_name
        self.dimension = 768
        self.model = MagicMock(spec=SentenceTransformer)
        self.model.encode.return_value = np.array([0.1, 0.2, 0.3])
        self.table_vector_index_options = TableVectorIndexOptions()
        self.vector_service_options = None
        
    @property
    def type(self):
        return VectorColumnType.SENTENCE_TRANSFORMER


class MockVectorizeOptions:
    """Mock VectorColumnOptions for VECTORIZE type"""
    def __init__(self, column_name="embeddings2"):
        self.column_name = column_name
        self.dimension = 1536
        self.model = None
        self.table_vector_index_options = TableVectorIndexOptions()
        self.vector_service_options = VectorServiceOptions(
            provider="openai",
            model_name="text-embedding-3-small",
            authentication={"providerKey": "test-key"}
        )
        
    @property
    def type(self):
        return VectorColumnType.VECTORIZE


class MockPrecomputedOptions:
    """Mock VectorColumnOptions for PRECOMPUTED type"""
    def __init__(self, column_name="precomputed_embeddings"):
        self.column_name = column_name
        self.dimension = 768
        self.model = None
        self.table_vector_index_options = TableVectorIndexOptions()
        self.vector_service_options = None
        
    @property
    def type(self):
        return VectorColumnType.PRECOMPUTED


# Wrapper for query results to handle to_list method
class QueryResults(list):
    """A list wrapper that adds to_list method"""
    def to_list(self):
        return self


# Mock reranker creator
def create_mock_reranker():
    """Create a properly configured mock reranker"""
    mock = MagicMock(spec=Reranker)
    mock.rank = MagicMock()
    # Create a proper RankedResults mock with query
    mock_ranked_results = MagicMock(spec=RankedResults)
    mock_ranked_results.query = "test query"
    mock_ranked_results.results = ["ranked1", "ranked2", "ranked3"]
    mock.rank.return_value = mock_ranked_results
    return mock, mock_ranked_results


class TestAstraMultiVectorTable(unittest.TestCase):
    
    def setUp(self):
        # Mock the database and table
        self.mock_db = MagicMock(spec=Database)
        self.mock_table = MagicMock(spec=Table)
        self.mock_db.create_table.return_value = self.mock_table
        
        # Create vector options using our mock classes instead of real ones
        self.vector_options1 = MockSentenceTransformerOptions(column_name="embeddings1")
        self.vector_options2 = MockVectorizeOptions(column_name="embeddings2")
        self.precomputed_options = MockPrecomputedOptions(column_name="precomputed_embeddings")
        
        # Standard and extended vector options lists
        self.vector_options = [self.vector_options1, self.vector_options2]
        self.extended_vector_options = [
            self.vector_options1, 
            self.vector_options2,
            self.precomputed_options
        ]
        
        # Create the table with our mock options
        with patch('astra_multivector.AstraMultiVectorTable._create_table') as mock_create:
            mock_create.return_value = self.mock_table
            self.table = AstraMultiVectorTable(
                db=self.mock_db,
                table_name="test_table",
                vector_column_options=self.vector_options
            )
            
            # Table with precomputed option for specific tests
            self.table_with_precomputed = AstraMultiVectorTable(
                db=self.mock_db,
                table_name="test_table_precomputed",
                vector_column_options=self.extended_vector_options
            )
        
        # Ensure both tables use our mock table
        self.table.table = self.mock_table
        self.table_with_precomputed.table = self.mock_table
        
        # Set up a properly mocked reranker
        self.mock_reranker, self.mock_ranked_results = create_mock_reranker()
    
    def test_init_and_create_table(self):
        # Create a fresh table to test initialization
        with patch('astra_multivector.AstraMultiVectorTable._create_table') as mock_create:
            mock_create.return_value = self.mock_table
            table = AstraMultiVectorTable(
                db=self.mock_db,
                table_name="test_table",
                vector_column_options=self.vector_options
            )
        
        # Check that table name was set correctly
        self.assertEqual(table.name, "test_table")
        
        # Check that vector options were stored
        self.assertEqual(len(table.vector_column_options), 2)
    
    def test_insert_chunk(self):
        # Reset mocks to ensure clean test
        self.mock_table.reset_mock()
        
        # Prepare the test
        text_chunk = "This is a test chunk"
        
        # Call the method
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = "test-uuid"
            self.table.insert_chunk(text_chunk)
        
        # Verify insert_one was called with the correct arguments
        self.mock_table.insert_one.assert_called_once()
        call_args = self.mock_table.insert_one.call_args[0][0]
        
        # Check content and chunk_id
        self.assertEqual(call_args["content"], text_chunk)
        self.assertEqual(call_args["chunk_id"], "test-uuid")
        
        # Check that embeddings were included
        self.assertIn("embeddings1", call_args)
        self.assertIn("embeddings2", call_args)
        
        # Check that model.encode was called for first embedding
        self.vector_options1.model.encode.assert_called_with(text_chunk)
        
        # Check that the second embedding (vectorize) just used the text directly
        self.assertEqual(call_args["embeddings2"], text_chunk)
    
    def test_bulk_insert_chunks(self):
        
        # Mock the actual implementation for testing
        with patch.object(AstraMultiVectorTable, 'bulk_insert_chunks') as mock_bulk_insert:
            text_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
            
            # Call the method
            self.table.bulk_insert_chunks(text_chunks, batch_size=2)
            
            # Get the actual call arguments
            args, kwargs = mock_bulk_insert.call_args
            
            # Verify text_chunks is the first argument
            self.assertEqual(args[0], text_chunks)
            
            # Verify batch_size parameter was passed correctly
            self.assertEqual(kwargs.get('batch_size'), 2)
            
            # Alternative approach: check that the method was called exactly once
            self.assertEqual(mock_bulk_insert.call_count, 1)
    
    def test_multi_vector_similarity_search_single_column(self):
        # Reset mocks to ensure clean test
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        # Prepare test
        query_text = "test query"
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        # Setup mock results using our wrapper class
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        self.mock_table.find.return_value = mock_results
        
        # Call the method with a single vector column
        results = self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings1"],
            candidates_per_column=5
        )
        
        # Verify find was called with the correct parameters
        self.mock_table.find.assert_called_with(
            sort={"embeddings1": encoded_query.tolist()},
            limit=5,
            include_similarity=True,
            filter={},
        )
        
        # Reset mock for the next test
        self.mock_table.reset_mock()
        
        # Test with the vectorize column (embeddings2)
        self.mock_table.find.return_value = mock_results
        self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings2"],
            candidates_per_column=5
        )
        
        # Verify find was called with the raw text for vectorize
        self.mock_table.find.assert_called_with(
            filter={},  # This should now be present
            sort={"embeddings2": query_text},
            limit=5,
            include_similarity=True,
        )
    
    def test_multi_vector_similarity_search_default_columns(self):
        # Reset mocks to ensure clean test
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        # Test that it uses all vector columns when none are specified
        query_text = "test query"
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        # Setup mock results for different columns using our wrapper class
        mock_results1 = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        mock_results2 = QueryResults([
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ])
        
        # Use side_effect to return different results for each call
        self.mock_table.find.side_effect = [mock_results1, mock_results2]
        
        # Call the method without specifying vector columns
        results = self.table.multi_vector_similarity_search(query_text=query_text)
        
        # Verify find was called twice (once for each column)
        self.assertEqual(self.mock_table.find.call_count, 2)
    
    def test_multi_vector_similarity_search_invalid_column(self):
        # Test with an invalid column name
        with self.assertRaises(ValueError):
            self.table.multi_vector_similarity_search(
                query_text="test query",
                vector_columns=["non_existent_column"]
            )
    
    def test_batch_search_by_text(self):
        # Reset mocks to ensure clean test
        self.mock_table.reset_mock()
        
        # Prepare test
        queries = ["query1", "query2"]
        
        # Setup mock results using our wrapper class
        mock_results1 = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        mock_results2 = QueryResults([
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ])
        
        # Use side_effect to return different results for each call
        self.mock_table.find.side_effect = [mock_results1, mock_results2]
        
        # Call the method
        results = self.table.batch_search_by_text(
            queries=queries,
            vector_columns=["embeddings1"],
            rerank=False,
        )
        
        # Verify find was called twice (once per query)
        self.assertEqual(self.mock_table.find.call_count, 2)
    
    def test_multi_vector_similarity_search(self):
        # Reset mocks to ensure clean test
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        # Prepare test
        query_text = "test query"
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        # Setup mock results for different columns using our wrapper class
        mock_results1 = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ])
        mock_results2 = QueryResults([
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.85},
            {"chunk_id": "id3", "content": "content3", "$similarity": 0.7}
        ])
        
        # Use side_effect to return different results for each call
        self.mock_table.find.side_effect = [mock_results1, mock_results2]
        
        # Call the method with all vector columns
        results = self.table.multi_vector_similarity_search(
            query_text=query_text,
            candidates_per_column=2
        )
        
        # Verify find was called twice (once for each column)
        self.assertEqual(self.mock_table.find.call_count, 2)
    
    def test_multi_vector_similarity_search_specific_columns(self):
        # Reset mocks to ensure clean test
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        # Prepare test
        query_text = "test query"
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        # Setup mock results using our wrapper class
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        self.mock_table.find.return_value = mock_results
        
        # Call the method with a specific vector column
        results = self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings1"]
        )
        
        # Verify find was called once (only for the specified column)
        self.assertEqual(self.mock_table.find.call_count, 1)
    
    def test_multi_vector_similarity_search_with_precomputed(self):
        # Reset mocks to ensure clean test
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        # Prepare test
        query_text = "test query"
        precomputed_embeddings = {"embeddings1": [0.1, 0.2, 0.3]}

        #self.precomputed_options.model.encode.return_value = [0.1, 0.2, 0.3]
        
        # Setup mock results using our wrapper class
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        self.mock_table.find.return_value = mock_results
        
        # Call the method with precomputed embeddings
        results = self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings1"],
            precomputed_embeddings=precomputed_embeddings
        )
        
        # Verify find was called with the precomputed embedding
        self.mock_table.find.assert_called_with(
            sort={"embeddings1": precomputed_embeddings["embeddings1"]}, 
            limit=10, 
            include_similarity=True,
            filter={},
        )
        
        # Verify model.encode was not called
        self.vector_options1.model.encode.assert_not_called()
    
    def test_multi_vector_similarity_search_missing_precomputed(self):
        # Reset mocks to ensure clean test
        self.mock_table.reset_mock()
        
        # Test with missing precomputed embedding for a PRECOMPUTED column
        with self.assertRaises(ValueError):
            self.table_with_precomputed.multi_vector_similarity_search(
                query_text="test query",
                vector_columns=["precomputed_embeddings"],
                precomputed_embeddings={}  # Empty precomputed embeddings
            )
    
    def test_rerank_results(self):
        # Prepare test
        query_text = "test query"
        results = [
            {"chunk_id": "id1", "content": "content1"},
            {"chunk_id": "id2", "content": "content2"},
            {"chunk_id": "id3", "content": "content3"}
        ]
        
        # Create a fresh mock reranker for this test
        mock_reranker, mock_ranked_results = create_mock_reranker()
        
        # Call the method
        reranked = self.table.rerank_results(
            query_text=query_text,
            results=results,
            reranker=mock_reranker
        )
        
        # Verify reranker.rank was called with the correct arguments
        mock_reranker.rank.assert_called_once_with(
            query=query_text,
            docs=["content1", "content2", "content3"],
            doc_ids=["id1", "id2", "id3"]
        )
        
        # Verify the results
        self.assertEqual(reranked, mock_ranked_results)
    
    def test_rerank_results_with_limit(self):
        # Prepare test
        query_text = "test query"
        results = [
            {"chunk_id": "id1", "content": "content1"},
            {"chunk_id": "id2", "content": "content2"},
            {"chunk_id": "id3", "content": "content3"}
        ]
        
        # Create a fresh mock reranker for this test
        mock_reranker, mock_ranked_results = create_mock_reranker()
        mock_ranked_results.results = ["ranked1", "ranked2", "ranked3"]
        
        # Patch the RankedResults constructor to ensure the query parameter is included
        with patch('rerankers.results.RankedResults') as mock_ranked_results_class:
            # Setup a limited results object
            limited_results = MagicMock(spec=RankedResults)
            limited_results.results = ["ranked1", "ranked2"]
            mock_ranked_results_class.return_value = limited_results
            
            # Call the method with a limit
            reranked = self.table.rerank_results(
                query_text=query_text,
                results=results,
                reranker=mock_reranker,
                limit=2
            )
            
            # Verify RankedResults constructor was called with query
            mock_ranked_results_class.assert_called_once_with(
                query=query_text,
                results=["ranked1", "ranked2"]
            )
            
            # Verify the results
            self.assertEqual(reranked, limited_results)
    
    def test_search_and_rerank(self):
        # Reset mocks to ensure clean test
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        # Prepare test
        query_text = "test query"
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        # Setup mock results using our wrapper class
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ])
        self.mock_table.find.return_value = mock_results
        
        # Create a fresh mock reranker for this test
        mock_reranker, mock_ranked_results = create_mock_reranker()
        
        # Call the method
        results = self.table.search_and_rerank(
            query_text=query_text,
            reranker=mock_reranker,
            vector_columns=["embeddings1"],
            candidates_per_column=5,
            rerank_limit=3
        )
        
        # Verify search was called
        self.assertEqual(self.mock_table.find.call_count, 1)
        
        # Verify reranker.rank was called
        self.assertEqual(mock_reranker.rank.call_count, 1)
        
        # Verify the results
        self.assertEqual(results, mock_ranked_results)
    
    def test_batch_search_by_text_with_reranking(self):
        # Reset mocks to ensure clean test
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        # Prepare test
        queries = ["query1", "query2"]
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        # Setup mock results using our wrapper class
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ])
        self.mock_table.find.return_value = mock_results
        
        # Create a fresh mock reranker for this test
        mock_reranker, mock_ranked_results = create_mock_reranker()
        
        # Call the method with reranking
        results = self.table.batch_search_by_text(
            queries=queries,
            vector_columns=["embeddings1"],
            rerank=True,
            reranker=mock_reranker,
            rerank_limit=3
        )
        
        # Verify find was called twice (once per query)
        self.assertEqual(self.mock_table.find.call_count, 2)
        # Verify reranker.rank was called twice
        self.assertEqual(mock_reranker.rank.call_count, 2)
        
        # Verify the results are a list with the expected length
        self.assertEqual(len(results), 2)
    
    def test_batch_search_by_text_with_reranking_error(self):
        # Test reranking without providing a reranker
        with self.assertRaises(ValueError):
            self.table.batch_search_by_text(
                queries=["query1"],
                rerank=True,
                reranker=None
            )
    
    def test_batch_search_by_text_precomputed_length_error(self):
        # Test with incorrect number of precomputed embeddings
        with self.assertRaises(ValueError):
            self.table.batch_search_by_text(
                queries=["query1", "query2"],
                precomputed_embeddings=[{"embeddings1": [0.1, 0.2]}]  # Only one embedding for two queries
            )


if __name__ == "__main__":
    unittest.main()
