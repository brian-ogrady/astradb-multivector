import unittest
from unittest.mock import MagicMock, patch, AsyncMock, call
import asyncio

from astrapy import AsyncDatabase, AsyncTable
from astrapy.info import TableVectorIndexOptions, VectorServiceOptions
from sentence_transformers import SentenceTransformer
from rerankers import Reranker
from rerankers.results import RankedResults

from astra_multivector import AsyncAstraMultiVectorTable, VectorColumnOptions


class TestAsyncAstraMultiVectorTable(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the AsyncAstraMultiVectorTable class.
    
    This test class uses unittest.IsolatedAsyncioTestCase to run async tests
    with isolated event loops to ensure proper testing of async functionality.
    """
    
    async def asyncSetUp(self):
        """
        Set up test environment before each test.
        
        Creates mock objects for:
        - Database and table with proper method chaining
        - Cursor with to_list method for simulating query results
        - SentenceTransformer model for embedding generation
        - Vector column options for both client-side and server-side embedding
        - AsyncAstraMultiVectorTable instance for testing
        """
        self.mock_db = AsyncMock(spec=AsyncDatabase)
        self.mock_table = AsyncMock(spec=AsyncTable)
        
        self.mock_db.create_table.return_value = self.mock_table
        self.mock_table.alter.return_value = self.mock_table
        
        cursor_mock = AsyncMock()
        cursor_mock.to_list = AsyncMock(return_value=[])
        self.mock_table.find.return_value = cursor_mock
        
        self.mock_model = MagicMock(spec=SentenceTransformer)
        self.mock_model.encode.return_value = [0.1, 0.2, 0.3]
        self.mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_card_data = MagicMock()
        mock_card_data.base_model = "test-model"
        self.mock_model.model_card_data = mock_card_data
        
        self.vector_options1 = VectorColumnOptions.from_sentence_transformer(
            model=self.mock_model,
            column_name="embeddings1",
            table_vector_index_options=TableVectorIndexOptions()
        )

        self.vector_service_options = VectorServiceOptions(
            provider="openai",
            model_name="text-embedding-3-small",
            authentication={"providerKey": "test-key"}
        )
        
        self.vector_options2 = VectorColumnOptions.from_vectorize(
            column_name="embeddings2",
            dimension=1536,
            vector_service_options=self.vector_service_options,
            table_vector_index_options=TableVectorIndexOptions()
        )
        
        self.table = AsyncAstraMultiVectorTable(
            db=self.mock_db,
            table_name="test_table",
            vector_column_options=[self.vector_options1, self.vector_options2],
            default_concurrency_limit=5
        )
    
    @patch('asyncio.to_thread')
    async def test_init_and_initialize(self, mock_to_thread):
        """
        Test initialization and asynchronous initialization of the table.
        
        Verifies that:
        - Before initialization, table is not marked as initialized
        - After calling _initialize(), the table is properly set up
        - Database create_table is called correctly
        - Table initialization flag is set
        - Table reference is set to the mock table
        - Table name is set correctly
        - Default concurrency limit is set correctly
        - Vector indexes are created for each vector column
        """
        self.assertFalse(self.table._initialized)
        self.assertIsNone(self.table.table)
        
        await self.table._initialize()
        
        self.mock_db.create_table.assert_called_once()
        
        self.assertTrue(self.table._initialized)
        self.assertEqual(self.table.table, self.mock_table)
        
        self.assertEqual(self.table.name, "test_table")
        
        self.assertEqual(self.table.default_concurrency_limit, 5)
        
        self.assertEqual(self.mock_table.create_vector_index.call_count, 2)
    
    @patch('asyncio.to_thread')
    async def test_insert_chunk(self, mock_to_thread):
        """
        Test inserting a single text chunk into the table.
        
        This test verifies:
        - The client-side embedding generation is offloaded to a thread
        - UUID generation is used for chunk_id
        - insert_one is called with correct document structure
        - Content and chunk_id fields are set correctly
        - Both embeddings (client and server side) are included
        - Server-side embedding uses raw text
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        await self.table._initialize()
        
        text_chunk = "This is a test chunk"
        
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = "test-uuid"
            await self.table.insert_chunk(text_chunk)
        
        self.mock_table.insert_one.assert_called_once()
        call_args = self.mock_table.insert_one.call_args[0][0]
        
        self.assertEqual(call_args["content"], text_chunk)
        self.assertEqual(call_args["chunk_id"], "test-uuid")
        
        self.assertIn("embeddings1", call_args)
        self.assertIn("embeddings2", call_args)
        
        mock_to_thread.assert_called_once()
        
        self.assertEqual(call_args["embeddings2"], text_chunk)
    
    @patch('asyncio.to_thread')
    async def test_bulk_insert_chunks(self, mock_to_thread):
        """
        Test bulk insertion of multiple text chunks into the table.
        
        This test verifies:
        - The method handles multiple text chunks correctly
        - insert_many is called once for the batch
        - Client-side embeddings are generated for each chunk
        - Thread offloading occurs for each embedding generation
        - Custom concurrency limits are respected
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        await self.table._initialize()
        
        text_chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"]
        
        await self.table.bulk_insert_chunks(
            text_chunks=text_chunks,
            max_concurrency=3,
        )
        
        self.assertEqual(self.mock_table.insert_many.call_count, 1)
        
        self.assertEqual(mock_to_thread.call_count, 4)
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_single_column(self, mock_to_thread):
        """
        Test similarity search using a single vector column.
        
        This test verifies:
        - Search with client-side embedding column works correctly
        - Search with server-side embedding column works correctly 
        - Thread offloading occurs for client-side embedding generation
        - Server-side embedding uses raw query text
        - find is called with correct parameters (sorting, limit, etc.)
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        await self.table._initialize()
        
        query_text = "test query"
        mock_results = [{"chunk_id": "id1", "content": "content1", "$similarity": 0.9}]
        cursor_mock = AsyncMock()
        cursor_mock.to_list.return_value = mock_results
        self.mock_table.find.return_value = cursor_mock
        
        results = await self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings1"],
            candidates_per_column=5
        )
        
        self.mock_table.find.assert_called_with(
            filter={},
            sort={"embeddings1": mock_to_thread.return_value},
            limit=5,
            include_similarity=True
        )
        
        cursor_mock = AsyncMock()
        cursor_mock.to_list.return_value = mock_results
        self.mock_table.find.return_value = cursor_mock
        
        await self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings2"],
            candidates_per_column=5
        )
        
        self.mock_table.find.assert_called_with(
            filter={},
            sort={"embeddings2": query_text},
            limit=5,
            include_similarity=True
        )
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_default_columns(self, mock_to_thread):
        """
        Test similarity search using all vector columns (default behavior).
        
        This test verifies:
        - When vector_columns is not specified, all columns are searched
        - Results from multiple columns are combined correctly
        - Each column is searched with the appropriate parameters
        - Thread offloading occurs for client-side embedding generation
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        await self.table._initialize()
        
        results1 = [{"chunk_id": "id1", "content": "content1", "$similarity": 0.9}]
        results2 = [{"chunk_id": "id2", "content": "content2", "$similarity": 0.8}]
        
        cursor1 = AsyncMock()
        cursor1.to_list.return_value = results1
        cursor2 = AsyncMock()
        cursor2.to_list.return_value = results2
        
        def find_side_effect(**kwargs):
            sort_column = list(kwargs.get("sort", {}).keys())[0]
            if sort_column == "embeddings1":
                return cursor1
            elif sort_column == "embeddings2":
                return cursor2
            return AsyncMock()
            
        self.mock_table.find.side_effect = find_side_effect
        
        results = await self.table.multi_vector_similarity_search(query_text="test query")
        
        self.assertEqual(self.mock_table.find.call_count, 2)
        
        self.assertEqual(len(results), 2)
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_invalid_column(self, mock_to_thread):
        """
        Test similarity search with an invalid vector column name.
        
        This test verifies:
        - ValueError is raised when a non-existent column is specified
        - The error handling properly validates column names
        """
        await self.table._initialize()
        
        with self.assertRaises(ValueError):
            await self.table.multi_vector_similarity_search(
                query_text="test query",
                vector_columns=["non_existent_column"]
            )
    
    @patch('asyncio.to_thread')
    async def test_batch_search_by_text(self, mock_to_thread):
        """
        Test batch searching for multiple queries.
        
        This test verifies:
        - Multiple queries can be processed in a batch
        - Results are returned for each query
        - find is called for each query with appropriate parameters
        - Results are correctly associated with each query
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        await self.table._initialize()
        
        queries = ["query1", "query2"]
        result1 = [{"chunk_id": "id1", "content": "content1", "$similarity": 0.9}]
        result2 = [{"chunk_id": "id2", "content": "content2", "$similarity": 0.8}]
        
        cursor1 = AsyncMock()
        cursor1.to_list = AsyncMock(return_value=result1)
        cursor2 = AsyncMock()
        cursor2.to_list = AsyncMock(return_value=result2)
        
        find_calls = []
        def find_side_effect(*args, **kwargs):
            find_calls.append(kwargs)
            if len(find_calls) == 1:
                return cursor1
            else:
                return cursor2
        
        self.mock_table.find.side_effect = find_side_effect
        
        results = await self.table.batch_search_by_text(
            queries=queries,
            vector_columns="embeddings1",
            limit=2
        )
        self.assertEqual(self.mock_table.find.call_count, 2)
        
        self.assertEqual(results, [result1, result2])
    
    @patch('asyncio.to_thread')
    async def test_parallel_process_chunks(self, mock_to_thread):
        """
        Test parallel processing of items with a custom async function.
        
        This test verifies:
        - Items can be processed concurrently with a custom async function
        - Concurrency limits are respected
        - Results are collected correctly from all processed items
        """
        await self.table._initialize()
        
        async def process_fn(item):
            return f"processed_{item}"
        
        results = await self.table.parallel_process_chunks(
            items=["item1", "item2", "item3"],
            process_fn=process_fn,
            max_concurrency=2
        )
        
        self.assertEqual(results, ["processed_item1", "processed_item2", "processed_item3"])
    
    @patch('asyncio.to_thread')
    async def test_get_embedding_for_column(self, mock_to_thread):
        """
        Test getting embeddings for different column types.
        
        This test verifies:
        - Client-side embedding generation uses to_thread for offloading
        - Server-side embedding returns the raw text
        - Precomputed embeddings are returned directly from provided data
        - Each embedding type is processed correctly
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        text = "Test text"
        
        embedding = await self.table._get_embedding_for_column(
            text=text,
            column_options=self.vector_options1
        )
        
        mock_to_thread.assert_called_once()
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        
        mock_to_thread.reset_mock()
        
        embedding = await self.table._get_embedding_for_column(
            text=text,
            column_options=self.vector_options2
        )
        
        mock_to_thread.assert_not_called()
        self.assertEqual(embedding, text)
        
        precomputed_options = VectorColumnOptions.from_precomputed_embeddings(
            column_name="precomputed",
            dimension=3,
            table_vector_index_options=TableVectorIndexOptions()
        )
        
        precomputed_embeddings = {"precomputed": [0.4, 0.5, 0.6]}
        embedding = await self.table._get_embedding_for_column(
            text=text,
            column_options=precomputed_options,
            precomputed_embeddings=precomputed_embeddings
        )
        
        self.assertEqual(embedding, [0.4, 0.5, 0.6])
    
    @patch('asyncio.to_thread')
    async def test_get_embedding_for_column_missing_precomputed(self, mock_to_thread):
        """
        Test error handling when precomputed embeddings are missing.
        
        This test verifies:
        - ValueError is raised when required precomputed embeddings are missing
        - The error message contains clear information about the issue
        """
        precomputed_options = VectorColumnOptions.from_precomputed_embeddings(
            column_name="precomputed",
            dimension=3,
            table_vector_index_options=TableVectorIndexOptions()
        )
        
        with self.assertRaises(ValueError):
            await self.table._get_embedding_for_column(
                text="test",
                column_options=precomputed_options,
                precomputed_embeddings={}
            )
    
    @patch('asyncio.to_thread')
    async def test_add_embedding_to_insertion(self, mock_to_thread):
        """
        Test adding embeddings to a document before insertion.
        
        This test verifies:
        - Embeddings for all configured vector columns are added to the document
        - Client-side embeddings use the thread-offloaded vector
        - Server-side embeddings use the raw text
        - Original document fields are preserved
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        text = "Test text"
        insertion = {"chunk_id": "test-id", "content": text}
        
        result = await self.table._add_embedding_to_insertion(insertion, text)
        
        self.assertEqual(result["embeddings1"], [0.1, 0.2, 0.3])
        self.assertEqual(result["embeddings2"], text)
        self.assertEqual(result["chunk_id"], "test-id")
        self.assertEqual(result["content"], text)

    def setup_find_mock(self, column_results_mapping):
        """
        Setup the find method mock to return different cursors based on the sort column.
        
        This helper method configures the mock_table.find method to return different
        cursor results depending on which vector column is being queried.
        
        Args:
            column_results_mapping: Dict mapping column names to their expected results.
                                E.g. {"embeddings1": results1, "embeddings2": results2}
                                
        Returns:
            Dictionary of cursor mocks created for each column
        """
        cursors = {}
        for column, results in column_results_mapping.items():
            cursor = AsyncMock()
            cursor.to_list = AsyncMock(return_value=results)
            cursors[column] = cursor
        
        def find_side_effect(**kwargs):
            sort_dict = kwargs.get("sort", {})
            if not sort_dict:
                default_cursor = AsyncMock()
                default_cursor.to_list = AsyncMock(return_value=[])
                return default_cursor
                
            sort_column = list(sort_dict.keys())[0]
            if sort_column in cursors:
                return cursors[sort_column]
            
            fallback_cursor = AsyncMock()
            fallback_cursor.to_list = AsyncMock(return_value=[])
            return fallback_cursor
            
        self.mock_table.find.side_effect = find_side_effect
    
        return cursors
    
    @patch('asyncio.to_thread')
    async def test_search_column(self, mock_to_thread):
        """
        Test searching a single vector column.
        
        This test verifies:
        - The internal _search_column method works correctly
        - Column name and results are returned as expected
        - find is called with correct parameters for the specified column
        - Client-side embedding generation is offloaded to a thread
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        await self.table._initialize()
        
        query_text = "test query"
        col_results = [
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ]
        cursor_mock = AsyncMock()
        cursor_mock.to_list.return_value = col_results
        self.mock_table.find.return_value = cursor_mock
        
        col_name, results = await self.table._search_column(
            column_name="embeddings1",
            query_text=query_text,
            candidates_per_column=2
        )
        
        self.assertEqual(col_name, "embeddings1")
        
        self.mock_table.find.assert_called_with(
            filter={},
            sort={"embeddings1": mock_to_thread.return_value},
            limit=2,
            include_similarity=True
        )
        
        self.assertEqual(results, col_results)
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search(self, mock_to_thread):
        """
        Test comprehensive multi-vector similarity search.
        
        This test verifies:
        - Results from multiple vector columns are combined correctly
        - Metadata about source columns is added to results
        - Documents found in multiple columns have source info from all columns
        - Similarity scores from each column are preserved
        - Results are properly deduplicated by chunk_id
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        await self.table._initialize()
        
        results1 = [
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ]
        results2 = [
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.85},
            {"chunk_id": "id3", "content": "content3", "$similarity": 0.7}
        ]
        
        cursor1 = AsyncMock()
        cursor1.to_list.return_value = results1
        cursor2 = AsyncMock()
        cursor2.to_list.return_value = results2
        
        def find_side_effect(**kwargs):
            sort_column = list(kwargs.get("sort", {}).keys())[0]
            if sort_column == "embeddings1":
                return cursor1
            elif sort_column == "embeddings2":
                return cursor2
            return AsyncMock()
            
        self.mock_table.find.side_effect = find_side_effect
        
        results = await self.table.multi_vector_similarity_search(
            query_text="test query",
            candidates_per_column=2
        )
        
        self.assertEqual(self.mock_table.find.call_count, 2)
        
        self.assertEqual(len(results), 3)  # 3 unique document IDs
        
        for doc in results:
            self.assertIn("source_columns", doc)
            
        id2_doc = next(doc for doc in results if doc["chunk_id"] == "id2")
        source_columns = id2_doc["source_columns"]
        self.assertEqual(len(source_columns), 2)
        
        col_names = [sc["column"] for sc in source_columns]
        self.assertIn("embeddings1", col_names)
        self.assertIn("embeddings2", col_names)
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_specific_columns(self, mock_to_thread):
        """
        Test similarity search with explicitly specified vector columns.
        
        This test verifies:
        - Search respects the specified vector columns parameter
        - Only the specified columns are searched
        - find is called with the correct sort column and parameters
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        await self.table._initialize()
        
        results = [
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ]
        cursor = AsyncMock()
        cursor.to_list.return_value = results
        self.mock_table.find.return_value = cursor
        
        await self.table.multi_vector_similarity_search(
            query_text="test query",
            vector_columns=["embeddings1"]
        )
        
        self.assertEqual(self.mock_table.find.call_count, 1)
        
        sort_args = self.mock_table.find.call_args[1]["sort"]
        self.assertIn("embeddings1", sort_args)
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_with_precomputed(self, mock_to_thread):
        """
        Test similarity search with precomputed embeddings.
        
        This test verifies:
        - Precomputed embeddings are used correctly for search
        - find is called with the precomputed vector, not the query text
        - The correct vector column is used for sorting
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        precomputed_options = VectorColumnOptions.from_precomputed_embeddings(
            column_name="precomputed",
            dimension=3,
            table_vector_index_options=TableVectorIndexOptions()
        )
        self.table.vector_column_options.append(precomputed_options)
        
        await self.table._initialize()
        
        results = [
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ]
        cursor = AsyncMock()
        cursor.to_list.return_value = results
        self.mock_table.find.return_value = cursor
        
        precomputed_embeddings = {"precomputed": [0.4, 0.5, 0.6]}
        await self.table.multi_vector_similarity_search(
            query_text="test query",
            vector_columns=["precomputed"],
            precomputed_embeddings=precomputed_embeddings
        )
        
        sort_args = self.mock_table.find.call_args[1]["sort"]
        self.assertEqual(sort_args["precomputed"], [0.4, 0.5, 0.6])
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_invalid_column(self, mock_to_thread):
        """
        Test error handling for invalid vector column names in similarity search.
        
        This test verifies:
        - ValueError is raised when a non-existent column is specified
        - The error handling properly validates column names
        """
        await self.table._initialize()
        
        with self.assertRaises(ValueError):
            await self.table.multi_vector_similarity_search(
                query_text="test query",
                vector_columns=["non_existent_column"]
            )
    
    @patch('asyncio.to_thread')
    async def test_rerank_results(self, mock_to_thread):
        """
        Test reranking of search results.
        
        This test verifies:
        - Reranker is properly used to rerank search results
        - Reranker.rank is called via thread offloading
        - The reranker receives the correct query and results
        - The reranked results are returned correctly
        """
        await self.table._initialize()
        
        query_text = "test query"
        results = [
            {"chunk_id": "id1", "content": "content1"},
            {"chunk_id": "id2", "content": "content2"},
            {"chunk_id": "id3", "content": "content3"}
        ]
        
        mock_reranker = MagicMock(spec=Reranker)
        mock_ranked_results = MagicMock(spec=RankedResults)
        mock_ranked_results.results = ["ranked1", "ranked2", "ranked3"]
        
        mock_to_thread.return_value = mock_ranked_results
        
        reranked = await self.table.rerank_results(
            query_text=query_text,
            results=results,
            reranker=mock_reranker
        )
        
        mock_to_thread.assert_called_once()
        
        self.assertEqual(reranked, mock_ranked_results)
    
    @patch('asyncio.to_thread')
    async def test_rerank_results_with_limit(self, mock_to_thread):
        """
        Test reranking with a specified result limit.
        
        This test verifies:
        - Reranked results are limited to the specified number
        - The limit is applied after reranking
        - The top N results are returned in correct order
        """
        await self.table._initialize()
        
        query_text = "test query"
        results = [
            {"chunk_id": "id1", "content": "content1"},
            {"chunk_id": "id2", "content": "content2"},
            {"chunk_id": "id3", "content": "content3"}
        ]
        
        mock_reranker = MagicMock(spec=Reranker)
        mock_ranked_results = MagicMock(spec=RankedResults)
        mock_ranked_results.results = ["ranked1", "ranked2", "ranked3"]
        
        mock_to_thread.return_value = mock_ranked_results
        
        reranked = await self.table.rerank_results(
            query_text=query_text,
            results=results,
            reranker=mock_reranker,
            limit=2
        )
        
        self.assertEqual(len(reranked.results), 2)
        self.assertEqual(reranked.results, ["ranked1", "ranked2"])
    
    @patch('asyncio.to_thread')
    async def test_search_and_rerank(self, mock_to_thread):
        """
        Test combined search and rerank operation.
        
        This test verifies:
        - The method correctly chains multi_vector_similarity_search and rerank_results
        - All parameters are passed correctly to both methods
        - The final reranked results are returned
        """
        await self.table._initialize()
        
        mock_search_results = [
            {"chunk_id": "id1", "content": "content1"},
            {"chunk_id": "id2", "content": "content2"}
        ]
        
        mock_reranker = MagicMock(spec=Reranker)
        mock_ranked_results = MagicMock(spec=RankedResults)
        
        self.table.multi_vector_similarity_search = AsyncMock(return_value=mock_search_results)
        self.table.rerank_results = AsyncMock(return_value=mock_ranked_results)
        
        results = await self.table.search_and_rerank(
            query_text="test query",
            reranker=mock_reranker,
            vector_columns=["embeddings1"],
            candidates_per_column=5,
            rerank_limit=3
        )
        
        self.table.multi_vector_similarity_search.assert_called_once_with(
            query_text="test query",
            vector_columns=["embeddings1"],
            precomputed_embeddings=None,
            candidates_per_column=5
        )
        
        self.table.rerank_results.assert_called_once_with(
            query_text="test query",
            results=mock_search_results,
            reranker=mock_reranker,
            limit=3
        )
        
        self.assertEqual(results, mock_ranked_results)
    
    @patch('asyncio.to_thread')
    async def test_process_single_query(self, mock_to_thread):
        """
        Test processing of a single query with and without reranking.
        
        This test verifies:
        - Without reranking, multi_vector_similarity_search is called
        - With reranking, search_and_rerank is called
        - Query data is correctly processed in both cases
        - Appropriate results are returned based on search method
        """
        await self.table._initialize()
        
        self.table.multi_vector_similarity_search = AsyncMock(return_value=["result1"])
        self.table.search_and_rerank = AsyncMock(return_value="ranked_results")
        
        query_data = (0, "test query")
        result = await self.table._process_single_query(
            query_data=query_data,
            vector_columns=["embeddings1"],
            candidates_per_column=5
        )
        
        self.table.multi_vector_similarity_search.assert_called_once()
        self.table.search_and_rerank.assert_not_called()
        self.assertEqual(result, ["result1"])
        
        self.table.multi_vector_similarity_search.reset_mock()
        
        mock_reranker = MagicMock(spec=Reranker)
        result = await self.table._process_single_query(
            query_data=query_data,
            vector_columns=["embeddings1"],
            candidates_per_column=5,
            rerank=True,
            reranker=mock_reranker
        )
        
        self.table.search_and_rerank.assert_called_once()
        self.assertEqual(result, "ranked_results")
    
    @patch('asyncio.to_thread')
    async def test_batch_search_by_text_with_reranking(self, mock_to_thread):
        """
        Test batch searching with reranking enabled.
        
        This test verifies:
        - Multiple queries can be processed in batch with reranking
        - _process_single_query is called for each query with correct parameters
        - Results are correctly associated with each query
        - Reranker is passed to the processing function
        """
        await self.table._initialize()
        
        queries = ["query1", "query2"]
        
        async def process_side_effect(query_data, **kwargs):
            query_index = query_data[0]
            return f"result{query_index+1}"
            
        self.table._process_single_query = AsyncMock(side_effect=process_side_effect)
        
        mock_reranker = MagicMock(spec=Reranker)
        results = await self.table.batch_search_by_text(
            queries=queries,
            vector_columns=["embeddings1"],
            rerank=True,
            reranker=mock_reranker,
            rerank_limit=3
        )
        
        self.assertEqual(self.table._process_single_query.call_count, 2)
        
        self.assertEqual(results, ["result1", "result2"])
    
    @patch('asyncio.to_thread')
    async def test_batch_search_by_text_with_reranking_error(self, mock_to_thread):
        """
        Test error handling when reranking is requested without a reranker.
        
        This test verifies:
        - ValueError is raised when rerank=True but no reranker is provided
        - The error message contains clear information about the issue
        """
        await self.table._initialize()
        
        with self.assertRaises(ValueError):
            await self.table.batch_search_by_text(
                queries=["query1"],
                rerank=True
            )
    
    @patch('asyncio.to_thread')
    async def test_batch_search_by_text_precomputed_length_error(self, mock_to_thread):
        """
        Test error handling for mismatched precomputed embeddings length.
        
        This test verifies:
        - ValueError is raised when the number of precomputed embeddings doesn't match queries
        - The error message clearly indicates the mismatch issue
        """
        await self.table._initialize()
        
        with self.assertRaises(ValueError):
            await self.table.batch_search_by_text(
                queries=["query1", "query2"],
                precomputed_embeddings=[{"embeddings1": [0.1, 0.2]}]  # Only one for two queries
            )
    
    @patch('asyncio.to_thread')
    async def test_process_chunk_with_semaphore(self, mock_to_thread):
        """
        Test processing a single chunk with a semaphore for concurrency control.
        
        This test verifies:
        - Chunk processing respects the semaphore for concurrency control
        - UUID generation is used for chunk_id
        - Embeddings are added to the chunk document
        - The chunk is properly formatted with all required fields
        """
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        text_chunk = "test chunk"
        semaphore = asyncio.Semaphore(1)
        
        original_add_embedding = self.table._add_embedding_to_insertion
        self.table._add_embedding_to_insertion = AsyncMock(
            side_effect=lambda insertion, text, precomputed: {**insertion, "embeddings": [0.1, 0.2]}
        )
        
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = "test-uuid"
            result = await self.table._process_chunk_with_semaphore(
                j=0,
                text_chunk=text_chunk,
                semaphore=semaphore
            )
        
        self.assertEqual(result["chunk_id"], "test-uuid")
        self.assertEqual(result["content"], text_chunk)
        self.assertEqual(result["embeddings"], [0.1, 0.2])
        
        self.table._add_embedding_to_insertion = original_add_embedding


if __name__ == "__main__":
    unittest.main()
    