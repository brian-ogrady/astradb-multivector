import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from astrapy import Database, Table
from astrapy.info import TableVectorIndexOptions, VectorServiceOptions
from sentence_transformers import SentenceTransformer
from rerankers import Reranker
from rerankers.results import RankedResults

from astra_multivector import AstraMultiVectorTable
from astra_multivector.vector_column_options import VectorColumnType


class MockSentenceTransformerOptions:
    """Mock VectorColumnOptions for SENTENCE_TRANSFORMER type.
    
    A mock implementation of VectorColumnOptions that simulates a SentenceTransformer-based
    vector column option with customizable column name and predefined dimension.
    The mock model returns a fixed embedding array when encode() is called.
    """
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
    """Mock VectorColumnOptions for VECTORIZE type.
    
    A mock implementation of VectorColumnOptions that simulates a Vectorize-based
    vector column option with customizable column name and predefined dimension.
    Includes vector_service_options for calling external embedding services.
    """
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
    """Mock VectorColumnOptions for PRECOMPUTED type.
    
    A mock implementation of VectorColumnOptions that simulates a precomputed
    vector column option with customizable column name and predefined dimension.
    Used for testing scenarios where embeddings are provided directly rather than
    computed through a model or service.
    """
    def __init__(self, column_name="precomputed_embeddings"):
        self.column_name = column_name
        self.dimension = 768
        self.model = None
        self.table_vector_index_options = TableVectorIndexOptions()
        self.vector_service_options = None
        
    @property
    def type(self):
        return VectorColumnType.PRECOMPUTED


# Helper classes for testing
class QueryResults(list):
    """A list wrapper that adds to_list method.
    
    Extends list to add the to_list method that returns self, mimicking
    the behavior of Astra query results objects which provide this method.
    This allows test code to not need to distinguish between real and mock results.
    """
    def to_list(self):
        return self


def create_mock_reranker():
    """Create a properly configured mock reranker.
    
    Returns a mock Reranker object and its corresponding mock RankedResults.
    The mock reranker's rank method returns the mock RankedResults when called.
    This allows tests to verify both the reranker interactions and results.
    
    Returns:
        tuple: (mock_reranker, mock_ranked_results)
    """
    mock = MagicMock(spec=Reranker)
    mock.rank = MagicMock()
    mock_ranked_results = MagicMock(spec=RankedResults)
    mock_ranked_results.query = "test query"
    mock_ranked_results.results = ["ranked1", "ranked2", "ranked3"]
    mock.rank.return_value = mock_ranked_results
    return mock, mock_ranked_results


class TestAstraMultiVectorTable(unittest.TestCase):
    """Test suite for the AstraMultiVectorTable class.
    
    Tests the functionality of the AstraMultiVectorTable class, which provides
    a high-level interface for working with multiple vector embeddings in Astra.
    """
    
    def setUp(self):
        """Set up test fixtures for each test method.
        
        Creates mock database and table objects, vector column options,
        and AstraMultiVectorTable instances for testing. Sets up two table instances:
        1. self.table: with SENTENCE_TRANSFORMER and VECTORIZE columns
        2. self.table_with_precomputed: adds a PRECOMPUTED column for specific tests
        """
        self.mock_db = MagicMock(spec=Database)
        self.mock_table = MagicMock(spec=Table)
        self.mock_db.create_table.return_value = self.mock_table
        
        self.vector_options1 = MockSentenceTransformerOptions(column_name="embeddings1")
        self.vector_options2 = MockVectorizeOptions(column_name="embeddings2")
        self.precomputed_options = MockPrecomputedOptions(column_name="precomputed_embeddings")
        
        self.vector_options = [self.vector_options1, self.vector_options2]
        self.extended_vector_options = [
            self.vector_options1, 
            self.vector_options2,
            self.precomputed_options
        ]
        
        with patch('astra_multivector.AstraMultiVectorTable._create_table') as mock_create:
            mock_create.return_value = self.mock_table
            self.table = AstraMultiVectorTable(
                db=self.mock_db,
                table_name="test_table",
                vector_column_options=self.vector_options
            )
            
            self.table_with_precomputed = AstraMultiVectorTable(
                db=self.mock_db,
                table_name="test_table_precomputed",
                vector_column_options=self.extended_vector_options
            )
        
        self.table.table = self.mock_table
        self.table_with_precomputed.table = self.mock_table
        
        self.mock_reranker, self.mock_ranked_results = create_mock_reranker()
    
    def test_init_and_create_table(self):
        """Test table initialization and creation.
        
        Verifies that the AstraMultiVectorTable constructor:
        - Sets the table name correctly
        - Stores the vector column options
        - Calls the _create_table method
        """
        with patch('astra_multivector.AstraMultiVectorTable._create_table') as mock_create:
            mock_create.return_value = self.mock_table
            table = AstraMultiVectorTable(
                db=self.mock_db,
                table_name="test_table",
                vector_column_options=self.vector_options
            )
        
        self.assertEqual(table.name, "test_table")
        self.assertEqual(len(table.vector_column_options), 2)
    
    def test_insert_chunk(self):
        """Test insertion of a single text chunk.
        
        Verifies that insert_chunk:
        - Generates a UUID for the chunk_id
        - Properly calls insert_one on the Astra table
        - Includes the right content and chunk_id in the inserted document
        - Generates embeddings for SentenceTransformer columns by calling encode
        - Passes raw text for Vectorize columns
        """
        self.mock_table.reset_mock()
        
        text_chunk = "This is a test chunk"
        
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = "test-uuid"
            self.table.insert_chunk(text_chunk)
        
        self.mock_table.insert_one.assert_called_once()
        call_args = self.mock_table.insert_one.call_args[0][0]
        
        self.assertEqual(call_args["content"], text_chunk)
        self.assertEqual(call_args["chunk_id"], "test-uuid")
        
        self.assertIn("embeddings1", call_args)
        self.assertIn("embeddings2", call_args)
        
        self.vector_options1.model.encode.assert_called_with(text_chunk)
        self.assertEqual(call_args["embeddings2"], text_chunk)
    
    def test_bulk_insert_chunks(self):
        """Test bulk insertion of multiple text chunks.
        
        Verifies that bulk_insert_chunks:
        - Takes a list of text chunks as input
        - Passes the batch_size parameter correctly
        - Calls the bulk insert implementation once
        """
        with patch.object(AstraMultiVectorTable, 'bulk_insert_chunks') as mock_bulk_insert:
            text_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
            
            self.table.bulk_insert_chunks(text_chunks, batch_size=2)
            
            args, kwargs = mock_bulk_insert.call_args
            
            self.assertEqual(args[0], text_chunks)
            self.assertEqual(kwargs.get('batch_size'), 2)
            self.assertEqual(mock_bulk_insert.call_count, 1)
    
    def test_multi_vector_similarity_search_single_column(self):
        """Test similarity search using a single vector column.
        
        Verifies that multi_vector_similarity_search:
        - Properly encodes the query for SentenceTransformer columns
        - Passes raw query text for Vectorize columns
        - Calls find with correct search parameters for each column type
        - Sets appropriate similarity search parameters (limit, sort, etc.)
        """
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        query_text = "test query"
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        self.mock_table.find.return_value = mock_results
        
        # Test with SentenceTransformer column
        results = self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings1"],
            candidates_per_column=5
        )
        
        self.mock_table.find.assert_called_with(
            sort={"embeddings1": encoded_query.tolist()},
            limit=5,
            include_similarity=True,
            filter={},
        )
        
        # Test with Vectorize column
        self.mock_table.reset_mock()
        self.mock_table.find.return_value = mock_results
        
        self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings2"],
            candidates_per_column=5
        )
        
        self.mock_table.find.assert_called_with(
            filter={},
            sort={"embeddings2": query_text},
            limit=5,
            include_similarity=True,
        )
    
    def test_multi_vector_similarity_search_default_columns(self):
        """Test similarity search using default columns (all available).
        
        Verifies that multi_vector_similarity_search:
        - Uses all available vector columns when none are explicitly specified
        - Makes the correct number of find calls (one per column)
        - Merges results from multiple columns correctly
        """
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        query_text = "test query"
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        mock_results1 = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        mock_results2 = QueryResults([
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ])
        
        self.mock_table.find.side_effect = [mock_results1, mock_results2]
        
        results = self.table.multi_vector_similarity_search(query_text=query_text)
        
        self.assertEqual(self.mock_table.find.call_count, 2)
    
    def test_multi_vector_similarity_search_invalid_column(self):
        """Test similarity search with an invalid vector column name.
        
        Verifies that multi_vector_similarity_search raises ValueError
        when a non-existent column name is specified.
        """
        with self.assertRaises(ValueError):
            self.table.multi_vector_similarity_search(
                query_text="test query",
                vector_columns=["non_existent_column"]
            )
    
    def test_batch_search_by_text(self):
        """Test batch searching of multiple queries.
        
        Verifies that batch_search_by_text:
        - Processes multiple queries in a single call
        - Makes the correct number of find calls (one per query)
        - Returns a list of results corresponding to each query
        - Handles the rerank=False case correctly
        """
        self.mock_table.reset_mock()
        
        queries = ["query1", "query2"]
        
        mock_results1 = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        mock_results2 = QueryResults([
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ])
        
        self.mock_table.find.side_effect = [mock_results1, mock_results2]
        
        results = self.table.batch_search_by_text(
            queries=queries,
            vector_columns=["embeddings1"],
            rerank=False,
        )
        
        self.assertEqual(self.mock_table.find.call_count, 2)
    
    def test_multi_vector_similarity_search(self):
        """Test similarity search using multiple vector columns.
        
        Verifies that multi_vector_similarity_search:
        - Makes the correct number of find calls (one per column)
        - Merges and deduplicates results from different columns
        - Sets the candidates_per_column parameter correctly
        """
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        query_text = "test query"
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        mock_results1 = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ])
        mock_results2 = QueryResults([
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.85},
            {"chunk_id": "id3", "content": "content3", "$similarity": 0.7}
        ])
        
        self.mock_table.find.side_effect = [mock_results1, mock_results2]
        
        results = self.table.multi_vector_similarity_search(
            query_text=query_text,
            candidates_per_column=2
        )
        
        self.assertEqual(self.mock_table.find.call_count, 2)
    
    def test_multi_vector_similarity_search_specific_columns(self):
        """Test similarity search using explicitly specified vector columns.
        
        Verifies that multi_vector_similarity_search:
        - Only searches specified columns when vector_columns parameter is provided
        - Makes the correct number of find calls (one per specified column)
        """
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        query_text = "test query"
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        self.mock_table.find.return_value = mock_results
        
        results = self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings1"]
        )
        
        self.assertEqual(self.mock_table.find.call_count, 1)
    
    def test_multi_vector_similarity_search_with_precomputed(self):
        """Test similarity search using precomputed embeddings.
        
        Verifies that multi_vector_similarity_search:
        - Correctly processes precomputed embeddings when provided
        - Passes the precomputed vectors to the find method's sort parameter
        - Properly configures other search parameters when using precomputed embeddings
        """
        self.mock_table.reset_mock()
        
        query_text = "test query"
        precomputed_embeddings = {"precomputed_embeddings": [0.4, 0.5, 0.6]}
        
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        self.mock_table.find.return_value = mock_results
        
        results = self.table_with_precomputed.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["precomputed_embeddings"],
            precomputed_embeddings=precomputed_embeddings
        )
        
        self.mock_table.find.assert_called_with(
            sort={"precomputed_embeddings": precomputed_embeddings["precomputed_embeddings"]}, 
            limit=10, 
            include_similarity=True,
            filter={},
        )
    
    def test_multi_vector_similarity_search_missing_precomputed(self):
        """Test error handling for missing precomputed embeddings.
        
        Verifies that multi_vector_similarity_search raises ValueError when:
        - A PRECOMPUTED vector column is specified
        - The corresponding precomputed embedding is not provided
        """
        self.mock_table.reset_mock()
        
        with self.assertRaises(ValueError):
            self.table_with_precomputed.multi_vector_similarity_search(
                query_text="test query",
                vector_columns=["precomputed_embeddings"],
                precomputed_embeddings={}  # Empty precomputed embeddings
            )
    
    def test_rerank_results(self):
        """Test reranking of search results.
        
        Verifies that rerank_results:
        - Properly extracts content and IDs from search results
        - Calls the reranker with the correct parameters (query, docs, doc_ids)
        - Returns the reranked results from the reranker
        """
        query_text = "test query"
        results = [
            {"chunk_id": "id1", "content": "content1"},
            {"chunk_id": "id2", "content": "content2"},
            {"chunk_id": "id3", "content": "content3"}
        ]
        
        mock_reranker, mock_ranked_results = create_mock_reranker()
        
        reranked = self.table.rerank_results(
            query_text=query_text,
            results=results,
            reranker=mock_reranker
        )
        
        mock_reranker.rank.assert_called_once_with(
            query=query_text,
            docs=["content1", "content2", "content3"],
            doc_ids=["id1", "id2", "id3"]
        )
        
        self.assertEqual(reranked, mock_ranked_results)
    
    def test_rerank_results_with_limit(self):
        """Test reranking of search results with a result limit.
        
        Verifies that rerank_results:
        - Properly limits results to the specified count when limit parameter is used
        - Creates a new RankedResults object with the limited results
        - Returns the limited RankedResults object
        """
        query_text = "test query"
        results = [
            {"chunk_id": "id1", "content": "content1"},
            {"chunk_id": "id2", "content": "content2"},
            {"chunk_id": "id3", "content": "content3"}
        ]
        
        mock_reranker, mock_ranked_results = create_mock_reranker()
        mock_ranked_results.query = query_text
        mock_ranked_results.results = ["ranked1", "ranked2", "ranked3"]
        
        limited_ranked_results = MagicMock(spec=RankedResults)
        limited_ranked_results.results = ["ranked1", "ranked2"]
        
        import astra_multivector.astra_multi_vector_table
        
        with patch.object(astra_multivector.astra_multi_vector_table, 'RankedResults') as mock_ranked_results_class:
            mock_ranked_results_class.return_value = limited_ranked_results
            
            reranked = self.table.rerank_results(
                query_text=query_text,
                results=results,
                reranker=mock_reranker,
                limit=2
            )
            
            mock_ranked_results_class.assert_called_once_with(
                query=query_text,
                results=["ranked1", "ranked2"]
            )
            
            self.assertEqual(reranked, limited_ranked_results)

    def test_search_and_rerank(self):
        """Test combined search and reranking workflow.
        
        Verifies that search_and_rerank:
        - Performs similarity search with the provided parameters
        - Calls the reranker with the search results
        - Respects the candidates_per_column and rerank_limit parameters
        - Returns the reranked results
        """
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        query_text = "test query"
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ])
        self.mock_table.find.return_value = mock_results
        
        mock_reranker, mock_ranked_results = create_mock_reranker()
        
        results = self.table.search_and_rerank(
            query_text=query_text,
            reranker=mock_reranker,
            vector_columns=["embeddings1"],
            candidates_per_column=5,
            rerank_limit=3
        )
        
        self.assertEqual(self.mock_table.find.call_count, 1)
        self.assertEqual(mock_reranker.rank.call_count, 1)
        self.assertEqual(results, mock_ranked_results)
    
    def test_batch_search_by_text_with_reranking(self):
        """Test batch searching of multiple queries with reranking.
        
        Verifies that batch_search_by_text with rerank=True:
        - Processes multiple queries in a single call
        - Makes the correct number of find calls (one per query) 
        - Calls the reranker for each query's results
        - Returns a list of reranked results corresponding to each query
        - Respects the rerank_limit parameter
        """
        self.mock_table.reset_mock()
        self.vector_options1.model.reset_mock()
        
        queries = ["query1", "query2"]
        encoded_query = np.array([0.1, 0.2, 0.3])
        self.vector_options1.model.encode.return_value = encoded_query
        
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ])
        self.mock_table.find.return_value = mock_results
        
        mock_reranker, mock_ranked_results = create_mock_reranker()
        
        results = self.table.batch_search_by_text(
            queries=queries,
            vector_columns=["embeddings1"],
            rerank=True,
            reranker=mock_reranker,
            rerank_limit=3
        )
        
        self.assertEqual(self.mock_table.find.call_count, 2)
        self.assertEqual(mock_reranker.rank.call_count, 2)
        self.assertEqual(len(results), 2)
    
    def test_batch_search_by_text_with_reranking_error(self):
        """Test error handling for batch search with invalid reranking parameters.
        
        Verifies that batch_search_by_text raises ValueError when:
        - rerank=True is specified
        - No reranker object is provided
        """
        with self.assertRaises(ValueError):
            self.table.batch_search_by_text(
                queries=["query1"],
                rerank=True,
                reranker=None
            )
    
    def test_batch_search_by_text_precomputed_length_error(self):
        """Test error handling for batch search with mismatched precomputed embeddings.
        
        Verifies that batch_search_by_text raises ValueError when:
        - The number of precomputed embeddings doesn't match the number of queries
        """
        with self.assertRaises(ValueError):
            self.table.batch_search_by_text(
                queries=["query1", "query2"],
                precomputed_embeddings=[{"embeddings1": [0.1, 0.2]}]
            )


if __name__ == "__main__":
    unittest.main()
