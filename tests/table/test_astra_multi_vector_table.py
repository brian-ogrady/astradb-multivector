import unittest
from unittest.mock import MagicMock, patch, call
import uuid

from astrapy.database import Database, Table
from astrapy.info import CreateTableDefinition, ColumnType, TableVectorIndexOptions
from sentence_transformers import SentenceTransformer

from astra_multivector import AstraMultiVectorTable, VectorColumnOptions


class TestAstraMultiVectorTable(unittest.TestCase):
    
    def setUp(self):
        # Mock the database and table
        self.mock_db = MagicMock(spec=Database)
        self.mock_table = MagicMock(spec=Table)
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
        self.table = AstraMultiVectorTable(
            db=self.mock_db,
            table_name="test_table",
            vector_column_options=[self.vector_options1, self.vector_options2]
        )
    
    def test_init_and_create_table(self):
        # Verify the table was created with the correct schema
        self.mock_db.create_table.assert_called_once()
        
        # Check that table name was set correctly
        self.assertEqual(self.table.name, "test_table")
        
        # Check that vector options were stored
        self.assertEqual(len(self.table.vector_column_options), 2)
        
        # Verify vector indexes were created
        self.assertEqual(self.mock_table.create_vector_index.call_count, 2)
    
    def test_insert_chunk(self):
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
        self.mock_model.encode.assert_called_once_with(text_chunk)
        
        # Check that the second embedding (vectorize) just used the text directly
        self.assertEqual(call_args["embeddings2"], text_chunk)
    
    def test_bulk_insert_chunks(self):
        # Prepare the test
        text_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        
        # Call the method with a small batch size to trigger multiple batches
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.side_effect = ["uuid1", "uuid2", "uuid3"]
            self.table.bulk_insert_chunks(text_chunks, batch_size=2)
        
        # Verify insert_many was called twice (for batch sizes 2 and 1)
        self.assertEqual(self.mock_table.insert_many.call_count, 2)
        
        # Verify encode was called for each chunk
        self.assertEqual(self.mock_model.encode.call_count, 3)
    
    def test_search_by_text(self):
        # Prepare test
        query_text = "test query"
        encoded_query = [0.1, 0.2, 0.3]
        self.mock_model.encode.return_value = encoded_query
        self.mock_table.find.return_value = ["result1", "result2"]
        
        # Call the method with the first embedding column
        results = self.table.search_by_text(
            query_text=query_text,
            vector_column="embeddings1",
            limit=5
        )
        
        # Verify find was called with the correct parameters
        self.mock_table.find.assert_called_with(
            filter={},
            sort={"embeddings1": encoded_query.tolist()},
            limit=5
        )
        
        # Verify the results
        self.assertEqual(results, ["result1", "result2"])
        
        # Test with the vectorize column (embeddings2)
        self.table.search_by_text(
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
    
    def test_search_by_text_default_column(self):
        # Test that it uses the first vector column when none is specified
        query_text = "test query"
        self.table.search_by_text(query_text=query_text)
        
        # Verify it used the first column
        self.mock_table.find.assert_called_with(
            filter={},
            sort={"embeddings1": self.mock_model.encode.return_value.tolist()},
            limit=10  # default limit
        )
    
    def test_search_by_text_invalid_column(self):
        # Test with an invalid column name
        with self.assertRaises(ValueError):
            self.table.search_by_text(
                query_text="test query",
                vector_column="non_existent_column"
            )
    
    def test_batch_search_by_text(self):
        # Prepare test
        queries = ["query1", "query2"]
        self.mock_table.find.side_effect = [["result1"], ["result2"]]
        
        # Call the method
        results = self.table.batch_search_by_text(
            queries=queries,
            vector_column="embeddings1",
            limit=2
        )
        
        # Verify find was called twice
        self.assertEqual(self.mock_table.find.call_count, 2)
        
        # Verify the results
        self.assertEqual(results, [["result1"], ["result2"]])


if __name__ == "__main__":
    unittest.main()