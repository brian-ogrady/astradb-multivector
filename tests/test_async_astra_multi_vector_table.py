import unittest
from unittest.mock import MagicMock, patch, AsyncMock, call
import asyncio
import uuid

from astrapy.database import AsyncDatabase, AsyncTable
from astrapy.info import CreateTableDefinition, ColumnType, TableVectorIndexOptions
from sentence_transformers import SentenceTransformer

from astra_multivector import AsyncAstraMultiVectorTable, VectorColumnOptions


class TestAsyncAstraMultiVectorTable(unittest.IsolatedAsyncioTestCase):
    
    async def asyncSetUp(self):
        # Mock the database and table
        self.mock_db = AsyncMock(spec=AsyncDatabase)
        self.mock_table = AsyncMock(spec=AsyncTable)
        self.mock_db.create_table.return_value = self.mock_table
        
        # Mock the vector column options
        self.mock_model = MagicMock(spec=SentenceTransformer)
        self.mock_model.encode.return_value = [0.1, 0.2, 0.3]
        
        # Create vector options
        self.vector_options1 = VectorColumnOptions(
            column_name="embeddings1",
            dimension=768,
            model=self.mock_model,
            table_vector_index_options=TableVectorIndexOptions()
        )
        
        self.vector_options2 = VectorColumnOptions(
            column_name="embeddings2",
            dimension=1536,
            vector_service_options=MagicMock(),
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
    async def test_search_by_text(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Prepare test
        query_text = "test query"
        self.mock_table.find.return_value = ["result1", "result2"]
        
        # Call the method with the first embedding column
        results = await self.table.search_by_text(
            query_text=query_text,
            vector_column="embeddings1",
            limit=5
        )
        
        # Verify find was called with the correct parameters
        self.mock_table.find.assert_called_with(
            filter={},
            sort={"embeddings1": mock_to_thread.return_value},
            limit=5
        )
        
        # Verify the results
        self.assertEqual(results, ["result1", "result2"])
        
        # Test with the vectorize column (embeddings2)
        await self.table.search_by_text(
            query_text=query_text,
            vector_column="embeddings2",
            limit=5
        )
        
        # Verify find was called with the raw text for vectorize
        self.mock_table.find.assert_called_with(
            filter={},
            sort={"embeddings2": query_text},
            limit=5
        )
    
    @patch('asyncio.to_thread')
    async def test_search_by_text_default_column(self, mock_to_thread):
        # Mock the to_thread function to return the encoded vector
        mock_to_thread.return_value = [0.1, 0.2, 0.3]
        
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Test that it uses the first vector column when none is specified
        query_text = "test query"
        await self.table.search_by_text(query_text=query_text)
        
        # Verify it used the first column
        self.mock_table.find.assert_called_with(
            filter={},
            sort={"embeddings1": mock_to_thread.return_value},
            limit=10  # default limit
        )
    
    @patch('asyncio.to_thread')
    async def test_search_by_text_invalid_column(self, mock_to_thread):
        # Ensure the table is initialized
        await self.table._initialize()
        
        # Test with an invalid column name
        with self.assertRaises(ValueError):
            await self.table.search_by_text(
                query_text="test query",
                vector_column="non_existent_column"
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


if __name__ == "__main__":
    unittest.main()