import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np
from astrapy import Database, Table
from astrapy.info import TableVectorIndexOptions, VectorServiceOptions, CreateTableDefinition, ColumnType
from astrapy.results import TableInsertManyResult, TableInsertOneResult
from sentence_transformers import SentenceTransformer
from rerankers import Reranker
from rerankers.results import RankedResults

from astra_multivector import AstraMultiVectorTable
from astra_multivector.vector_column_options import VectorColumnType


class MockSentenceTransformerOptions:
    """Mock implementation of VectorColumnOptions for SENTENCE_TRANSFORMER type.
    
    Provides a test double for VectorColumnOptions that simulates a SentenceTransformer-based
    vector column option. Contains all necessary properties and behaviors needed to
    test AstraMultiVectorTable functionality without requiring a real model.
    
    Attributes:
        column_name: Name of the vector column in the database
        dimension: Vector dimension for the embeddings (fixed at 768)
        model: Mock SentenceTransformer with encode method returning fixed values
        table_vector_index_options: Configuration options for the vector index
        vector_service_options: None for SentenceTransformer type (client-side embeddings)
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
        """Returns the vector column type enumeration value."""
        return VectorColumnType.SENTENCE_TRANSFORMER


class MockVectorizeOptions:
    """Mock implementation of VectorColumnOptions for VECTORIZE type.
    
    Provides a test double for VectorColumnOptions that simulates a Vectorize-based
    vector column option. Contains all necessary properties to test server-side
    embedding generation through Astra Vectorize.
    
    Attributes:
        column_name: Name of the vector column in the database
        dimension: Vector dimension for the embeddings (fixed at 1536)
        model: None for Vectorize type (server-side embeddings)
        table_vector_index_options: Configuration options for the vector index
        vector_service_options: Configuration for the external embedding service
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
        """Returns the vector column type enumeration value."""
        return VectorColumnType.VECTORIZE


class MockPrecomputedOptions:
    """Mock implementation of VectorColumnOptions for PRECOMPUTED type.
    
    Provides a test double for VectorColumnOptions that simulates a precomputed
    vector column option. Used for testing scenarios where embeddings are provided
    directly by the caller rather than generated through a model or service.
    
    Attributes:
        column_name: Name of the vector column in the database
        dimension: Vector dimension for the embeddings (fixed at 768)
        model: None for precomputed type (no model used)
        table_vector_index_options: Configuration options for the vector index
        vector_service_options: None for precomputed type (no service used)
    """
    def __init__(self, column_name="precomputed_embeddings"):
        self.column_name = column_name
        self.dimension = 768
        self.model = None
        self.table_vector_index_options = TableVectorIndexOptions()
        self.vector_service_options = None
        
    @property
    def type(self):
        """Returns the vector column type enumeration value."""
        return VectorColumnType.PRECOMPUTED


class QueryResults(list):
    """List wrapper that simulates Astra query results behavior.
    
    Extends the built-in list type to add a to_list method that returns itself,
    mimicking the behavior of Astra DB query results. This allows test code to
    use the same patterns that would be used with real results objects.
    
    Methods:
        to_list: Returns self, simulating the Astra results object method.
    """
    def to_list(self):
        """Returns self to mimic Astra DB result objects."""
        return self


def create_mock_reranker():
    """Creates a configured mock reranker for testing.
    
    Constructs a mock Reranker and a mock RankedResults object with predefined
    behavior. The mock reranker's rank method returns the mock ranked results
    when called, allowing tests to verify the reranking behavior without
    requiring a real reranker implementation.
    
    Returns:
        tuple: A tuple containing:
            - mock_reranker: The mock Reranker object
            - mock_ranked_results: The mock RankedResults that the reranker will return
    """
    mock = MagicMock(spec=Reranker)
    mock.rank = MagicMock()
    mock_ranked_results = MagicMock(spec=RankedResults)
    mock_ranked_results.query = "test query"
    mock_ranked_results.results = ["ranked1", "ranked2", "ranked3"]
    mock.rank.return_value = mock_ranked_results
    return mock, mock_ranked_results


class TestAstraMultiVectorTable(unittest.TestCase):
    """Comprehensive test suite for the AstraMultiVectorTable class.
    
    This test suite verifies the functionality of the AstraMultiVectorTable class,
    which provides a unified interface for working with multiple vector embeddings
    in Astra DB. The tests cover all major features including:
    
    - Table creation and schema configuration
    - Various embedding types (client-side, server-side, and precomputed)
    - Single and batch document insertion
    - Vector similarity search across multiple columns
    - Result filtering and reranking
    - Error handling and edge cases
    
    Each test method focuses on a specific aspect of the functionality, using
    mock objects to simulate the Astra DB API without requiring an actual database.
    """
    
    def setUp(self):
        """Initializes test fixtures before each test method execution.
        
        Creates and configures all mock objects needed for testing:
        
        - Mock Database: Simulates the Astra DB connection
        - Mock Table: Represents the table in the database
        - Vector column options: Configuration for different embedding types
          - SENTENCE_TRANSFORMER: Client-side embeddings using models
          - VECTORIZE: Server-side embeddings using Astra Vectorize
          - PRECOMPUTED: Pre-calculated embeddings provided by the caller
        - Two table instances:
          - self.table: Standard setup with SENTENCE_TRANSFORMER and VECTORIZE
          - self.table_with_precomputed: Extended setup that adds PRECOMPUTED
        - Mock reranker: For testing result reranking functionality
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
        """Validates the proper initialization of the AstraMultiVectorTable class.
        
        This test ensures that the constructor correctly initializes the table instance
        by verifying that:
        
        1. The table name property is set to the provided name
        2. The vector_column_options property contains the provided options
        3. The _create_table method is called during initialization
        
        This test only checks the basic constructor behavior, not the actual table
        creation logic, which is tested separately in test_create_table_structure.
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
    
    def test_create_table_structure(self):
        """Examines the database table creation process in detail.
        
        This test verifies that the _create_table method properly constructs and configures
        an Astra DB table with vector search capabilities by confirming that:
        
        1. The table schema includes the required base columns (chunk_id, content)
        2. Each vector column is added with the correct dimensions based on its configuration
        3. Partition keys are properly configured for chunk_id
        4. Vector indexes are created for each vector column with appropriate options
        5. Vectorize service options are properly configured for server-side embedding columns
        
        This test mocks the CreateTableDefinition builder to track schema construction
        and verify that all components are properly added and configured.
        """
        mock_db = MagicMock(spec=Database)
        mock_table = MagicMock(spec=Table)
        mock_db.create_table.return_value = mock_table
        
        vector_options = [
            MockSentenceTransformerOptions(column_name="embeddings1"),
            MockVectorizeOptions(column_name="embeddings2")
        ]
        
        with patch.object(CreateTableDefinition, 'builder') as mock_builder:
            mock_schema = MagicMock()
            mock_builder.return_value = mock_schema
            mock_schema.add_column.return_value = mock_schema
            mock_schema.add_partition_by.return_value = mock_schema
            mock_schema.add_vector_column.return_value = mock_schema
            mock_schema.build.return_value = {"mock_schema": True}
            
            table = AstraMultiVectorTable(
                db=mock_db,
                table_name="test_table",
                vector_column_options=vector_options
            )
            
            # Verify schema construction
            mock_schema.add_column.assert_has_calls([
                call("chunk_id", ColumnType.UUID),
                call("content", ColumnType.TEXT)
            ])
            mock_schema.add_partition_by.assert_called_once_with(["chunk_id"])
            mock_schema.add_vector_column.assert_has_calls([
                call("embeddings1", dimension=768),
                call("embeddings2", dimension=1536)
            ])
            
            # Verify table creation
            mock_db.create_table.assert_called_once_with(
                "test_table",
                definition={"mock_schema": True},
                if_not_exists=True
            )
            
            # Verify vector index creation
            mock_table.create_vector_index.assert_has_calls([
                call("test_table_embeddings1_idx", column="embeddings1", 
                     options=vector_options[0].table_vector_index_options, if_not_exists=True),
                call("test_table_embeddings2_idx", column="embeddings2", 
                     options=vector_options[1].table_vector_index_options, if_not_exists=True)
            ])
            
            # Verify Vectorize configuration for appropriate columns
            mock_table.alter.assert_called_once()
            args, kwargs = mock_table.alter.call_args
            self.assertEqual(args[0].columns, vector_options[1].vector_service_options)
    
    def test_insert_chunk(self):
        """Validates the insertion of a single text chunk into the vector table.
        
        This test ensures that the insert_chunk method correctly processes and stores
        a text chunk with appropriate vector embeddings by verifying that:
        
        1. A UUID is generated for the chunk_id when not explicitly provided
        2. The insert_one method is called on the underlying Astra table
        3. The inserted document includes both the original text content and the chunk_id
        4. For SentenceTransformer columns, embeddings are generated by calling encode() on the model
        5. For Vectorize columns, the raw text is passed directly for server-side embedding
        
        The test uses a mock UUID generator to ensure consistent testing of the generated IDs.
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
    
    def test_insert_chunk_with_precomputed_embeddings(self):
        """Validates insertion of a text chunk with externally computed embeddings.
        
        This test ensures that the insert_chunk method correctly handles pre-computed
        vector embeddings provided by the caller, verifying that:
        
        1. The supplied precomputed embeddings are used for PRECOMPUTED column types
        2. The embeddings are included in the document passed to insert_one
        3. Additional keyword arguments (like timeout) are correctly passed to insert_one
        4. The method returns the result from the underlying insert_one operation
        
        This test is crucial for scenarios where embeddings are generated externally
        or through custom embedding processes not directly supported by the library.
        """
        self.mock_table.reset_mock()
        
        text_chunk = "This is a test chunk"
        precomputed_embeddings = {
            "precomputed_embeddings": [0.4, 0.5, 0.6]
        }
        
        # Create mock for insert_one that returns success result
        mock_result = MagicMock(spec=TableInsertOneResult)
        self.mock_table.insert_one.return_value = mock_result
        
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = "test-uuid-2"
            result = self.table_with_precomputed.insert_chunk(
                text_chunk,
                precomputed_embeddings=precomputed_embeddings,
                timeout=10
            )
        
        # Verify correct call to insert_one with precomputed embeddings and kwargs
        self.assertEqual(self.mock_table.insert_one.call_count, 1)
        call_args, call_kwargs = self.mock_table.insert_one.call_args
        
        self.assertEqual(call_args[0]["content"], text_chunk)
        self.assertEqual(call_args[0]["chunk_id"], "test-uuid-2")
        self.assertEqual(call_args[0]["precomputed_embeddings"], [0.4, 0.5, 0.6])
        self.assertEqual(call_kwargs["timeout"], 10)
        
        # Verify result is returned correctly
        self.assertEqual(result, mock_result)
    
    def test_insert_chunk_missing_precomputed_embeddings(self):
        """Tests error handling when required precomputed embeddings are missing.
        
        This test validates that the insert_chunk method properly detects and reports
        missing precomputed embeddings that are required by the table configuration.
        It verifies that:
        
        1. A ValueError is raised when attempting to insert with PRECOMPUTED columns
           but without providing the necessary embeddings
        2. The error message clearly indicates which column's embeddings are missing
        
        This test helps ensure that the library prevents incomplete or invalid data
        from being stored in the database, maintaining data integrity.
        """
        with self.assertRaises(ValueError) as context:
            self.table_with_precomputed.insert_chunk(
                "Test chunk", 
                precomputed_embeddings={}  # Empty precomputed embeddings
            )
        self.assertIn("Precomputed embeddings required", str(context.exception))
    
    def test_bulk_insert_chunks_implementation(self):
        """Validates the detailed implementation of batch text chunk insertion.
        
        This test thoroughly examines the bulk_insert_chunks method's internal logic,
        ensuring it correctly processes multiple text chunks in an efficient manner.
        It verifies that:
        
        1. Text chunks are properly divided into batches according to the batch_size parameter
        2. Each chunk in a batch is correctly processed with appropriate embedding generation
        3. UUID generation works correctly for each document
        4. The insert_many method is called with the correctly formed batch documents
        5. Additional keyword arguments (e.g., timeout) are properly passed to insert_many
        6. The method collects and returns all insert_many operation results
        
        This test is critical for validating the efficiency and correctness of bulk
        operations, which are essential for high-throughput embedding pipelines.
        """
        self.mock_table.reset_mock()
        
        # Create 5 test chunks to be inserted in batches of 2
        text_chunks = [
            "Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Chunk 5"
        ]
        
        # Create mock for insert_many with appropriate returns
        mock_results = [MagicMock(spec=TableInsertManyResult) for _ in range(3)]
        self.mock_table.insert_many.side_effect = mock_results
        
        # Create predictable UUIDs
        mock_uuids = [f"uuid-{i}" for i in range(5)]
        
        with patch('uuid.uuid4', side_effect=mock_uuids):
            results = self.table.bulk_insert_chunks(
                text_chunks, 
                batch_size=2,
                timeout=20
            )
        
        # Verify correct number of batches and insert_many calls
        self.assertEqual(self.mock_table.insert_many.call_count, 3)
        
        # Verify each batch had correct content
        expected_batches = [
            # First batch (2 items)
            [
                {"chunk_id": "uuid-0", "content": "Chunk 1", 
                 "embeddings1": [0.1, 0.2, 0.3], "embeddings2": "Chunk 1"},
                {"chunk_id": "uuid-1", "content": "Chunk 2", 
                 "embeddings1": [0.1, 0.2, 0.3], "embeddings2": "Chunk 2"}
            ],
            # Second batch (2 items)
            [
                {"chunk_id": "uuid-2", "content": "Chunk 3", 
                 "embeddings1": [0.1, 0.2, 0.3], "embeddings2": "Chunk 3"},
                {"chunk_id": "uuid-3", "content": "Chunk 4", 
                 "embeddings1": [0.1, 0.2, 0.3], "embeddings2": "Chunk 4"}
            ],
            # Third batch (1 item)
            [
                {"chunk_id": "uuid-4", "content": "Chunk 5", 
                 "embeddings1": [0.1, 0.2, 0.3], "embeddings2": "Chunk 5"}
            ]
        ]
        
        for i, expected_batch in enumerate(expected_batches):
            call_args, call_kwargs = self.mock_table.insert_many.call_args_list[i]
            self.assertEqual(call_args[0], expected_batch)
            self.assertEqual(call_kwargs["timeout"], 20)
        
        # Verify encode was called for each chunk
        self.assertEqual(self.vector_options1.model.encode.call_count, 5)
        
        # Verify results collection
        self.assertEqual(len(results), 3)
        self.assertEqual(results, mock_results)
    
    def test_bulk_insert_with_precomputed_embeddings(self):
        """Validates batch insertion with externally computed embeddings.
        
        This test ensures that the bulk_insert_chunks method correctly processes
        multiple text chunks with precomputed embeddings, verifying that:
        
        1. Precomputed embeddings are correctly matched to their corresponding chunks
        2. Each document in the batch includes its appropriate precomputed embedding
        3. The embeddings are properly included in the documents sent to insert_many
        
        This test is particularly important for workflows where embeddings are
        generated through external processes or batch embedding pipelines, and then
        need to be efficiently stored in the Astra database.
        """
        self.mock_table.reset_mock()
        
        text_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        precomputed_embeddings = {
            "precomputed_embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9]
            ]
        }
        
        mock_result = MagicMock(spec=TableInsertManyResult)
        self.mock_table.insert_many.return_value = mock_result
        
        with patch('uuid.uuid4', side_effect=["uuid-1", "uuid-2", "uuid-3"]):
            results = self.table_with_precomputed.bulk_insert_chunks(
                text_chunks, 
                precomputed_embeddings=precomputed_embeddings,
                batch_size=3
            )
        
        # Verify insert_many called with correct embeddings for each document
        call_args = self.mock_table.insert_many.call_args[0][0]
        
        # Check each document has the correct precomputed embedding
        for i, doc in enumerate(call_args):
            expected_embedding = precomputed_embeddings["precomputed_embeddings"][i]
            self.assertEqual(doc["precomputed_embeddings"], expected_embedding)
    
    def test_bulk_insert_not_enough_precomputed_embeddings(self):
        """Tests error handling when batch insertion has insufficient precomputed embeddings.
        
        This test validates that the bulk_insert_chunks method properly detects and
        reports situations where there aren't enough precomputed embeddings provided
        for all the text chunks being inserted. It verifies that:
        
        1. A ValueError is raised when the number of precomputed embeddings is less
           than the number of text chunks
        2. The error message clearly identifies the column with insufficient embeddings
        
        This validation is crucial for preventing partial or inconsistent data insertion
        that could compromise data integrity or cause unexpected behavior during retrieval.
        """
        text_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        precomputed_embeddings = {
            "precomputed_embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
                # Missing embedding for third chunk
            ]
        }
        
        with self.assertRaises(ValueError) as context:
            self.table_with_precomputed.bulk_insert_chunks(
                text_chunks, 
                precomputed_embeddings=precomputed_embeddings
            )
        self.assertIn("Not enough precomputed embeddings", str(context.exception))
    
    def test_bulk_insert_chunks(self):
        """Tests the public API for bulk insertion of multiple text chunks.
        
        This test focuses on the interface aspects of the bulk_insert_chunks method,
        ensuring that it correctly accepts and processes its parameters. It verifies that:
        
        1. The method accepts a list of text chunks as its primary input
        2. The batch_size parameter is correctly passed to the implementation
        3. The method is called exactly once with the expected parameters
        
        Unlike test_bulk_insert_chunks_implementation, which tests the internal logic,
        this test ensures the method's public interface works as expected.
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
    
    def test_multi_vector_similarity_search_result_structure(self):
        """Validates the detailed structure of multi-vector search results.
        
        This test examines the format and content of results returned by the
        multi_vector_similarity_search method, focusing on the rich metadata
        added to enhance result understanding and analysis. It verifies that:
        
        1. Results are correctly deduplicated by chunk_id across multiple vector columns
        2. The source_columns metadata is added to each result document
        3. For documents found in multiple columns, all source columns are recorded
        4. Each source column entry contains the column name, rank, and similarity score
        5. All original document data is preserved in the results
        
        This metadata is essential for applications that need to understand which
        embedding types contributed to a document's retrieval and how similar the
        document was to the query in each vector space.
        """
        self.mock_table.reset_mock()
        
        query_text = "test query"
        
        # Create results with some overlapping document IDs
        mock_results1 = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8},
            {"chunk_id": "id3", "content": "content3", "$similarity": 0.7}
        ])
        
        mock_results2 = QueryResults([
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.85},
            {"chunk_id": "id3", "content": "content3", "$similarity": 0.75},
            {"chunk_id": "id4", "content": "content4", "$similarity": 0.65}
        ])
        
        self.mock_table.find.side_effect = [mock_results1, mock_results2]
        
        results = self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings1", "embeddings2"]
        )
        
        # Should have 4 unique documents
        self.assertEqual(len(results), 4)
        
        # Check source_columns structure for a document found in both columns
        doc2 = next(doc for doc in results if doc["chunk_id"] == "id2")
        self.assertIn("source_columns", doc2)
        self.assertEqual(len(doc2["source_columns"]), 2)
        
        # Verify source column metadata structure
        source_cols = doc2["source_columns"]
        self.assertEqual(source_cols[0]["column"], "embeddings1")
        self.assertEqual(source_cols[0]["rank"], 2)
        self.assertEqual(source_cols[0]["similarity"], 0.8)
        
        self.assertEqual(source_cols[1]["column"], "embeddings2")
        self.assertEqual(source_cols[1]["rank"], 1)
        self.assertEqual(source_cols[1]["similarity"], 0.85)
        
        # Check document only found in one column
        doc1 = next(doc for doc in results if doc["chunk_id"] == "id1")
        self.assertEqual(len(doc1["source_columns"]), 1)
        self.assertEqual(doc1["source_columns"][0]["column"], "embeddings1")
    
    def test_multi_vector_similarity_search_with_filter(self):
        """Validates vector similarity search with filtering capabilities.
        
        This test ensures that the multi_vector_similarity_search method properly
        supports filtering and additional query parameters by verifying that:
        
        1. Filter conditions are correctly passed to the underlying find method
        2. The filter is preserved and applied consistently across all vector columns
        3. Additional parameters like projection are correctly passed to the find method
        
        Filtering is essential for real-world applications that need to combine
        vector similarity search with traditional query constraints (e.g., searching
        only within specific categories, date ranges, or other metadata conditions).
        """
        self.mock_table.reset_mock()
        
        query_text = "test query"
        test_filter = {"category": "science"}
        
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        self.mock_table.find.return_value = mock_results
        
        results = self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings1"],
            filter=test_filter,
            projection=["chunk_id", "content"]
        )
        
        # Verify filter was passed correctly
        call_kwargs = self.mock_table.find.call_args[1]
        self.assertEqual(call_kwargs["filter"], test_filter)
        
        # Verify additional parameters were passed correctly
        self.assertEqual(call_kwargs["projection"], ["chunk_id", "content"])
    
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
    
    def test_multi_vector_similarity_search_with_warning(self):
        """Tests graceful handling of malformed documents in search results.
        
        This test validates that the multi_vector_similarity_search method properly
        handles edge cases where documents in the search results are missing required
        fields. It verifies that:
        
        1. A warning is issued when a document without a chunk_id is encountered
        2. Malformed documents are excluded from the final results
        3. Properly formed documents are still processed and included
        
        This robust error handling is critical for production applications where
        data integrity issues might occur, ensuring the system can continue operating
        even when some documents have missing or invalid data.
        """
        self.mock_table.reset_mock()
        
        query_text = "test query"
        
        # Create results with a document missing chunk_id
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"content": "content_no_id", "$similarity": 0.8},  # Missing chunk_id
            {"chunk_id": "id3", "content": "content3", "$similarity": 0.7}
        ])
        
        self.mock_table.find.return_value = mock_results
        
        with self.assertWarns(Warning):
            results = self.table.multi_vector_similarity_search(
                query_text=query_text,
                vector_columns=["embeddings1"]
            )
        
        # Should only include documents with chunk_id
        self.assertEqual(len(results), 2)
        chunk_ids = [doc["chunk_id"] for doc in results]
        self.assertIn("id1", chunk_ids)
        self.assertIn("id3", chunk_ids)
    
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
    
    def test_rerank_results_empty_list(self):
        """Tests graceful handling of empty result sets in the reranking process.
        
        This test validates that the rerank_results method properly handles the
        edge case where no search results are available for reranking. It verifies that:
        
        1. An empty RankedResults object is returned when given an empty results list
        2. The reranker's rank method is not called when there are no results to rerank
        3. The returned object is a valid RankedResults instance with an empty results list
        
        This edge case handling is important for ensuring that applications can handle
        zero-result scenarios gracefully without errors or exceptions.
        """
        query_text = "test query"
        empty_results = []
        
        mock_reranker, _ = create_mock_reranker()
        
        result = self.table.rerank_results(
            query_text=query_text,
            results=empty_results,
            reranker=mock_reranker
        )
        
        # Reranker should not be called
        mock_reranker.rank.assert_not_called()
        
        # Should return empty RankedResults
        self.assertIsInstance(result, RankedResults)
        self.assertEqual(len(result.results), 0)
    
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
    
    def test_search_and_rerank_with_precomputed(self):
        """Validates the end-to-end search and rerank workflow with precomputed embeddings.
        
        This test ensures that the search_and_rerank method properly integrates with
        precomputed embeddings in a complete retrieval and reranking pipeline. It verifies that:
        
        1. Precomputed embeddings are correctly passed to the underlying search method
        2. The search is conducted using the specified vector columns and precomputed embeddings
        3. The search results are properly passed to the reranking step
        4. The rerank_limit parameter correctly controls the number of results after reranking
        
        This test validates the complete retrieval pipeline for scenarios where embeddings
        are computed through external means, ensuring all components work together seamlessly.
        """
        self.mock_table.reset_mock()
        
        query_text = "test query"
        precomputed_embeddings = {"precomputed_embeddings": [0.4, 0.5, 0.6]}
        
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ])
        self.mock_table.find.return_value = mock_results
        
        mock_reranker, mock_ranked_results = create_mock_reranker()
        
        with patch.object(self.table_with_precomputed, 'multi_vector_similarity_search') as mock_search, \
             patch.object(self.table_with_precomputed, 'rerank_results') as mock_rerank:
            
            mock_search.return_value = mock_results
            mock_rerank.return_value = mock_ranked_results
            
            results = self.table_with_precomputed.search_and_rerank(
                query_text=query_text,
                reranker=mock_reranker,
                vector_columns=["precomputed_embeddings"],
                precomputed_embeddings=precomputed_embeddings,
                candidates_per_column=5,
                rerank_limit=2
            )
            
            # Verify search called with precomputed embeddings
            mock_search.assert_called_once_with(
                query_text=query_text,
                vector_columns=["precomputed_embeddings"],
                precomputed_embeddings=precomputed_embeddings,
                candidates_per_column=5
            )
            
            # Verify rerank called with correct limit
            mock_rerank.assert_called_once_with(
                query_text=query_text,
                results=mock_results,
                reranker=mock_reranker,
                limit=2
            )
            
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
    
    def test_batch_search_with_precomputed_embeddings(self):
        """Validates batch search operations with precomputed embeddings for each query.
        
        This test ensures that the batch_search_by_text method correctly processes
        multiple queries with their corresponding precomputed embeddings. It verifies that:
        
        1. Each query is matched with its corresponding precomputed embeddings
        2. The correct embeddings are passed to the search method for each query
        3. All queries in the batch are processed correctly
        
        This functionality is essential for high-throughput applications that need to
        perform many searches with custom embedding processes, such as specialized
        embedding models or multi-stage retrieval pipelines.
        """
        self.mock_table.reset_mock()
        
        queries = ["query1", "query2"]
        precomputed_embeddings = [
            {"precomputed_embeddings": [0.1, 0.2, 0.3]},
            {"precomputed_embeddings": [0.4, 0.5, 0.6]}
        ]
        
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        self.mock_table.find.return_value = mock_results
        
        with patch.object(self.table_with_precomputed, 'multi_vector_similarity_search') as mock_search:
            mock_search.return_value = mock_results
            
            results = self.table_with_precomputed.batch_search_by_text(
                queries=queries,
                vector_columns=["precomputed_embeddings"],
                precomputed_embeddings=precomputed_embeddings,
                rerank=False
            )
            
            # Verify search called with correct precomputed embeddings for each query
            self.assertEqual(mock_search.call_count, 2)
            
            call_args_list = mock_search.call_args_list
            self.assertEqual(call_args_list[0][1]["precomputed_embeddings"], precomputed_embeddings[0])
            self.assertEqual(call_args_list[1][1]["precomputed_embeddings"], precomputed_embeddings[1])
    
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
    
    def test_batch_search_by_text_empty_queries(self):
        """Test batch search with empty query list.
        
        Verifies that batch_search_by_text:
        - Returns an empty list when given empty queries
        - Does not make any find calls
        """
        self.mock_table.reset_mock()
        
        results = self.table.batch_search_by_text(
            queries=[],
            vector_columns=["embeddings1"],
            rerank=False
        )
        
        self.assertEqual(results, [])
        self.mock_table.find.assert_not_called()
    
    def test_batch_search_custom_parameters(self):
        """Test batch search with additional custom parameters.
        
        Verifies that batch_search_by_text:
        - Passes additional parameters through to the search method
        - Handles complex filtering and configuration options
        """
        self.mock_table.reset_mock()
        
        queries = ["query1"]
        custom_filter = {"category": "science"}
        
        mock_results = QueryResults([
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ])
        self.mock_table.find.return_value = mock_results
        
        with patch.object(self.table, 'multi_vector_similarity_search') as mock_search:
            mock_search.return_value = mock_results
            
            self.table.batch_search_by_text(
                queries=queries,
                vector_columns=["embeddings1"],
                filter=custom_filter,
                projection=["chunk_id", "content"],
                timeout=1000,
                rerank=False
            )
            
            # Verify all parameters passed through
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args[1]
            self.assertEqual(call_kwargs["filter"], custom_filter)
            self.assertEqual(call_kwargs["projection"], ["chunk_id", "content"])
            self.assertEqual(call_kwargs["timeout"], 1000)


if __name__ == "__main__":
    unittest.main()
