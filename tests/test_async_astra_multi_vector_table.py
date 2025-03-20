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
    
    async def asyncSetUp(self):
        # Mock the database and table
        self.mock_db = AsyncMock(spec=AsyncDatabase)
        self.mock_table = AsyncMock(spec=AsyncTable)
        self.mock_db.create_table.return_value = self.mock_table

        self.mock_table.find = AsyncMock(return_value=[])
        
        # Mock the vector column options
        self.mock_model = MagicMock(spec=SentenceTransformer)
        self.mock_model.encode.return_value = [0.1, 0.2, 0.3]
        self.mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_card_data = MagicMock()
        mock_card_data.base_model = "test-model"
        self.mock_model.model_card_data = mock_card_data
        
        # Create vector options
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
        
        # Create the table
        self.table = AsyncAstraMultiVectorTable(
            db=self.mock_db,
            table_name="test_table",
            vector_column_options=[self.vector_options1, self.vector_options2],
            default_concurrency_limit=5
        )
    
    @patch('asyncio.to_thread')
    async def test_init_and_initialize(self, mock_to_thread):
        # At this point, table should not be initialized
        self.assertFalse(self.table._initialized)
        self.assertIsNone(self.table.table)
        
        # Trigger initialization
        await self.table._initialize()
        
        # Verify the table was created with the correct schema
        self.mock_db.create_table.assert_called_once()
        
        # Check that the table is now initialized
        self.assertTrue(self.table._initialized)
        self.assertEqual(self.table.table, self.mock_table)
        
        # Check that table name was set correctly
        self.assertEqual(self.table.name, "test_table")
        
        # Check default concurrency limit
        self.assertEqual(self.table.default_concurrency_limit, 5)
        
        # Verify vector indexes were created
        self.assertEqual(self.mock_table.create_vector_index.call_count, 2)
    
    @patch('asyncio.to_thread')
    async def test_insert_chunk(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare the test
        text_chunk = "This is a test chunk"
        
        # Call the method
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = "test-uuid"
            await self.table.insert_chunk(text_chunk)
        
        # Verify insert_one was called with the correct arguments
        self.mock_table.insert_one.assert_called_once()
        call_args = self.mock_table.insert_one.call_args[0][0]
        
        # Check content and chunk_id
        self.assertEqual(call_args["content"], text_chunk)
        self.assertEqual(call_args["chunk_id"], "test-uuid")
        
        # Check that embeddings were included
        self.assertIn("embeddings1", call_args)
        self.assertIn("embeddings2", call_args)
        
        # Check that to_thread was called for client-side embedding
        mock_to_thread.assert_called_once()
        
        # Check that the second embedding (vectorize) just used the text directly
        self.assertEqual(call_args["embeddings2"], text_chunk)
    
    @patch('asyncio.to_thread')
    async def test_bulk_insert_chunks(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare the test
        text_chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"]
        
        # Call the method with custom concurrency and batch size
        await self.table.bulk_insert_chunks(
            text_chunks=text_chunks,
            max_concurrency=3,
            batch_size=2
        )
        
        # Verify insert_many was called twice (for 2 batches of 2 chunks)
        self.assertEqual(self.mock_table.insert_many.call_count, 2)
        
        # Verify to_thread was called for each chunk (4 times)
        self.assertEqual(mock_to_thread.call_count, 4)
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_single_column(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare test
        query_text = "test query"
        mock_results = [{"chunk_id": "id1", "content": "content1", "$similarity": 0.9}]
        cursor_mock = AsyncMock()
        cursor_mock.to_list.return_value = mock_results
        self.mock_table.find.return_value = cursor_mock
        
        # Call the method with a single vector column
        results = await self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings1"],
            candidates_per_column=5
        )
        
        # Verify find was called with the correct parameters
        self.mock_table.find.assert_called_with(
            filter={},
            sort={"embeddings1": mock_to_thread.return_value},
            limit=5,
            include_similarity=True
        )
        
        # Reset mock and test with vectorize column
        cursor_mock = AsyncMock()
        cursor_mock.to_list.return_value = mock_results
        self.mock_table.find.return_value = cursor_mock
        
        # Test with the vectorize column (embeddings2)
        await self.table.multi_vector_similarity_search(
            query_text=query_text,
            vector_columns=["embeddings2"],
            candidates_per_column=5
        )
        
        # Verify find was called with the raw text for vectorize
        self.mock_table.find.assert_called_with(
            filter={},
            sort={"embeddings2": query_text},
            limit=5,
            include_similarity=True
        )
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_default_columns(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Configure mock cursor results for different columns
        results1 = [{"chunk_id": "id1", "content": "content1", "$similarity": 0.9}]
        results2 = [{"chunk_id": "id2", "content": "content2", "$similarity": 0.8}]
        
        cursor1 = AsyncMock()
        cursor1.to_list.return_value = results1
        cursor2 = AsyncMock()
        cursor2.to_list.return_value = results2
        
        # Setup mock to return different cursors based on column
        def find_side_effect(**kwargs):
            sort_column = list(kwargs.get("sort", {}).keys())[0]
            if sort_column == "embeddings1":
                return cursor1
            elif sort_column == "embeddings2":
                return cursor2
            return AsyncMock()
            
        self.mock_table.find.side_effect = find_side_effect
        
        # Call the method without specifying vector columns
        results = await self.table.multi_vector_similarity_search(query_text="test query")
        
        # Verify it searched all columns with default candidates_per_column
        self.assertEqual(self.mock_table.find.call_count, 2)
        
        # Verify results were combined
        self.assertEqual(len(results), 2)
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_invalid_column(self, mock_to_thread):
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Test with an invalid column name
        with self.assertRaises(ValueError):
            await self.table.multi_vector_similarity_search(
                query_text="test query",
                vector_columns=["non_existent_column"]
            )
    
    @patch('asyncio.to_thread')
    async def test_batch_search_by_text(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare test
        queries = ["query1", "query2"]
        self.mock_table.find.side_effect = [["result1"], ["result2"]]
        
        # Call the method
        results = await self.table.batch_search_by_text(
            queries=queries,
            vector_column="embeddings1",
            limit=2
        )
        
        # Verify find was called twice
        self.assertEqual(self.mock_table.find.call_count, 2)
        
        # Verify the results
        self.assertEqual(results, [["result1"], ["result2"]])
    
    @patch('asyncio.to_thread')
    async def test_parallel_process_chunks(self, mock_to_thread):
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Define a test async processing function
        async def process_fn(item):
            return f"processed_{item}"
        
        # Call the method
        results = await self.table.parallel_process_chunks(
            items=["item1", "item2", "item3"],
            process_fn=process_fn,
            max_concurrency=2
        )
        
        # Verify results
        self.assertEqual(results, ["processed_item1", "processed_item2", "processed_item3"])
    
    @patch('asyncio.to_thread')
    async def test_get_embedding_for_column(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Prepare test
        text = "Test text"
        
        # Test with SENTENCE_TRANSFORMER column
        embedding = await self.table._get_embedding_for_column(
            text=text,
            column_options=self.vector_options1
        )
        
        # Verify to_thread was called
        mock_to_thread.assert_called_once()
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        
        # Reset mock
        mock_to_thread.reset_mock()
        
        # Test with VECTORIZE column
        embedding = await self.table._get_embedding_for_column(
            text=text,
            column_options=self.vector_options2
        )
        
        # Verify to_thread was not called
        mock_to_thread.assert_not_called()
        self.assertEqual(embedding, text)
        
        # Test with PRECOMPUTED column
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
        # Test with PRECOMPUTED column but missing embedding
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
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Prepare test
        text = "Test text"
        insertion = {"chunk_id": "test-id", "content": text}
        
        # Test adding embeddings to insertion
        result = await self.table._add_embedding_to_insertion(insertion, text)
        
        # Verify the insertion was updated correctly
        self.assertEqual(result["embeddings1"], [0.1, 0.2, 0.3])
        self.assertEqual(result["embeddings2"], text)
        self.assertEqual(result["chunk_id"], "test-id")
        self.assertEqual(result["content"], text)
    
    @patch('asyncio.to_thread')
    async def test_search_column(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare test
        query_text = "test query"
        col_results = [
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9},
            {"chunk_id": "id2", "content": "content2", "$similarity": 0.8}
        ]
        cursor_mock = AsyncMock()
        cursor_mock.to_list.return_value = col_results
        self.mock_table.find.return_value = cursor_mock
        
        # Call the method
        col_name, results = await self.table._search_column(
            column_name="embeddings1",
            query_text=query_text,
            candidates_per_column=2
        )
        
        # Verify correct column name is returned
        self.assertEqual(col_name, "embeddings1")
        
        # Verify find was called with correct parameters
        self.mock_table.find.assert_called_with(
            sort={"embeddings1": mock_to_thread.return_value},
            limit=2,
            include_similarity=True
        )
        
        # Verify results
        self.assertEqual(results, col_results)
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare test results for different columns
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
        
        # Set up mock to return different cursors for different columns
        def find_side_effect(**kwargs):
            sort_column = list(kwargs.get("sort", {}).keys())[0]
            if sort_column == "embeddings1":
                return cursor1
            elif sort_column == "embeddings2":
                return cursor2
            return AsyncMock()
            
        self.mock_table.find.side_effect = find_side_effect
        
        # Call the method
        results = await self.table.multi_vector_similarity_search(
            query_text="test query",
            candidates_per_column=2
        )
        
        # Verify find was called twice (once per column)
        self.assertEqual(self.mock_table.find.call_count, 2)
        
        # Verify results were combined correctly
        self.assertEqual(len(results), 3)  # 3 unique document IDs
        
        # Verify source_columns metadata
        for doc in results:
            self.assertIn("source_columns", doc)
            
        # Check that id2 appears in both source columns
        id2_doc = next(doc for doc in results if doc["chunk_id"] == "id2")
        source_columns = id2_doc["source_columns"]
        self.assertEqual(len(source_columns), 2)
        
        # Verify columns
        col_names = [sc["column"] for sc in source_columns]
        self.assertIn("embeddings1", col_names)
        self.assertIn("embeddings2", col_names)
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_specific_columns(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare test results
        results = [
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ]
        cursor = AsyncMock()
        cursor.to_list.return_value = results
        self.mock_table.find.return_value = cursor
        
        # Call the method with specific column
        await self.table.multi_vector_similarity_search(
            query_text="test query",
            vector_columns=["embeddings1"]
        )
        
        # Verify find was called once with the correct column
        self.assertEqual(self.mock_table.find.call_count, 1)
        
        # Check correct sort column was used
        sort_args = self.mock_table.find.call_args[1]["sort"]
        self.assertIn("embeddings1", sort_args)
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_with_precomputed(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Add a precomputed column
        precomputed_options = VectorColumnOptions.from_precomputed_embeddings(
            column_name="precomputed",
            dimension=3,
            table_vector_index_options=TableVectorIndexOptions()
        )
        self.table.vector_column_options.append(precomputed_options)
        
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare test results
        results = [
            {"chunk_id": "id1", "content": "content1", "$similarity": 0.9}
        ]
        cursor = AsyncMock()
        cursor.to_list.return_value = results
        self.mock_table.find.return_value = cursor
        
        # Call the method with precomputed embedding
        precomputed_embeddings = {"precomputed": [0.4, 0.5, 0.6]}
        await self.table.multi_vector_similarity_search(
            query_text="test query",
            vector_columns=["precomputed"],
            precomputed_embeddings=precomputed_embeddings
        )
        
        # Verify find was called with the precomputed embedding
        sort_args = self.mock_table.find.call_args[1]["sort"]
        self.assertEqual(sort_args["precomputed"], [0.4, 0.5, 0.6])
    
    @patch('asyncio.to_thread')
    async def test_multi_vector_similarity_search_invalid_column(self, mock_to_thread):
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Test with invalid column name
        with self.assertRaises(ValueError):
            await self.table.multi_vector_similarity_search(
                query_text="test query",
                vector_columns=["non_existent_column"]
            )
    
    @patch('asyncio.to_thread')
    async def test_rerank_results(self, mock_to_thread):
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare test
        query_text = "test query"
        results = [
            {"chunk_id": "id1", "content": "content1"},
            {"chunk_id": "id2", "content": "content2"},
            {"chunk_id": "id3", "content": "content3"}
        ]
        
        # Create mock reranker and ranked results
        mock_reranker = MagicMock(spec=Reranker)
        mock_ranked_results = MagicMock(spec=RankedResults)
        mock_ranked_results.results = ["ranked1", "ranked2", "ranked3"]
        
        # Mock the reranker.rank call via to_thread
        mock_to_thread.return_value = mock_ranked_results
        
        # Call the method
        reranked = await self.table.rerank_results(
            query_text=query_text,
            results=results,
            reranker=mock_reranker
        )
        
        # Verify to_thread was called with lambda that calls reranker.rank
        mock_to_thread.assert_called_once()
        
        # Verify the results
        self.assertEqual(reranked, mock_ranked_results)
    
    @patch('asyncio.to_thread')
    async def test_rerank_results_with_limit(self, mock_to_thread):
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare test
        query_text = "test query"
        results = [
            {"chunk_id": "id1", "content": "content1"},
            {"chunk_id": "id2", "content": "content2"},
            {"chunk_id": "id3", "content": "content3"}
        ]
        
        # Create mock reranker and ranked results
        mock_reranker = MagicMock(spec=Reranker)
        mock_ranked_results = MagicMock(spec=RankedResults)
        mock_ranked_results.results = ["ranked1", "ranked2", "ranked3"]
        
        # Mock the reranker.rank call via to_thread
        mock_to_thread.return_value = mock_ranked_results
        
        # Call the method with limit
        reranked = await self.table.rerank_results(
            query_text=query_text,
            results=results,
            reranker=mock_reranker,
            limit=2
        )
        
        # Verify results were limited
        self.assertEqual(len(reranked.results), 2)
        self.assertEqual(reranked.results, ["ranked1", "ranked2"])
    
    @patch('asyncio.to_thread')
    async def test_search_and_rerank(self, mock_to_thread):
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Mock multi_vector_similarity_search results
        mock_search_results = [
            {"chunk_id": "id1", "content": "content1"},
            {"chunk_id": "id2", "content": "content2"}
        ]
        
        # Mock reranker and rerank_results
        mock_reranker = MagicMock(spec=Reranker)
        mock_ranked_results = MagicMock(spec=RankedResults)
        
        # Mock the methods
        self.table.multi_vector_similarity_search = AsyncMock(return_value=mock_search_results)
        self.table.rerank_results = AsyncMock(return_value=mock_ranked_results)
        
        # Call the method
        results = await self.table.search_and_rerank(
            query_text="test query",
            reranker=mock_reranker,
            vector_columns=["embeddings1"],
            candidates_per_column=5,
            rerank_limit=3
        )
        
        # Verify multi_vector_similarity_search was called
        self.table.multi_vector_similarity_search.assert_called_once_with(
            query_text="test query",
            vector_columns=["embeddings1"],
            precomputed_embeddings=None,
            candidates_per_column=5
        )
        
        # Verify rerank_results was called
        self.table.rerank_results.assert_called_once_with(
            query_text="test query",
            results=mock_search_results,
            reranker=mock_reranker,
            limit=3
        )
        
        # Verify results
        self.assertEqual(results, mock_ranked_results)
    
    @patch('asyncio.to_thread')
    async def test_process_single_query(self, mock_to_thread):
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Mock the required methods
        self.table.multi_vector_similarity_search = AsyncMock(return_value=["result1"])
        self.table.search_and_rerank = AsyncMock(return_value="ranked_results")
        
        # Test without reranking
        query_data = (0, "test query")
        result = await self.table._process_single_query(
            query_data=query_data,
            vector_columns=["embeddings1"],
            candidates_per_column=5
        )
        
        # Verify multi_vector_similarity_search was called
        self.table.multi_vector_similarity_search.assert_called_once()
        self.table.search_and_rerank.assert_not_called()
        self.assertEqual(result, ["result1"])
        
        # Reset mocks
        self.table.multi_vector_similarity_search.reset_mock()
        
        # Test with reranking
        mock_reranker = MagicMock(spec=Reranker)
        result = await self.table._process_single_query(
            query_data=query_data,
            vector_columns=["embeddings1"],
            candidates_per_column=5,
            rerank=True,
            reranker=mock_reranker
        )
        
        # Verify search_and_rerank was called
        self.table.search_and_rerank.assert_called_once()
        self.assertEqual(result, "ranked_results")
    
    @patch('asyncio.to_thread')
    async def test_batch_search_by_text_with_reranking(self, mock_to_thread):
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare test
        queries = ["query1", "query2"]
        
        # Mock _process_single_query to return different results for each query
        async def process_side_effect(query_data, **kwargs):
            query_index = query_data[0]
            return f"result{query_index+1}"
            
        self.table._process_single_query = AsyncMock(side_effect=process_side_effect)
        
        # Call the method with reranking
        mock_reranker = MagicMock(spec=Reranker)
        results = await self.table.batch_search_by_text(
            queries=queries,
            vector_columns=["embeddings1"],
            rerank=True,
            reranker=mock_reranker,
            rerank_limit=3
        )
        
        # Verify _process_single_query was called for each query
        self.assertEqual(self.table._process_single_query.call_count, 2)
        
        # Verify the results
        self.assertEqual(results, ["result1", "result2"])
    
    @patch('asyncio.to_thread')
    async def test_batch_search_by_text_with_reranking_error(self, mock_to_thread):
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Test reranking without providing a reranker
        with self.assertRaises(ValueError):
            await self.table.batch_search_by_text(
                queries=["query1"],
                rerank=True
            )
    
    @patch('asyncio.to_thread')
    async def test_batch_search_by_text_precomputed_length_error(self, mock_to_thread):
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Test with incorrect number of precomputed embeddings
        with self.assertRaises(ValueError):
            await self.table.batch_search_by_text(
                queries=["query1", "query2"],
                precomputed_embeddings=[{"embeddings1": [0.1, 0.2]}]  # Only one for two queries
            )
    
    @patch('asyncio.to_thread')
    async def test_process_chunk_with_semaphore(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Prepare test
        text_chunk = "test chunk"
        semaphore = asyncio.Semaphore(1)
        
        # Mock _add_embedding_to_insertion
        original_add_embedding = self.table._add_embedding_to_insertion
        self.table._add_embedding_to_insertion = AsyncMock(
            side_effect=lambda insertion, text, precomputed: {**insertion, "embeddings": [0.1, 0.2]}
        )
        
        # Call the method
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = "test-uuid"
            result = await self.table._process_chunk_with_semaphore(
                j=0,
                text_chunk=text_chunk,
                semaphore=semaphore
            )
        
        # Verify the result
        self.assertEqual(result["chunk_id"], "test-uuid")
        self.assertEqual(result["content"], text_chunk)
        self.assertEqual(result["embeddings"], [0.1, 0.2])
        
        # Restore original method
        self.table._add_embedding_to_insertion = original_add_embedding


if __name__ == "__main__":
    unittest.main()