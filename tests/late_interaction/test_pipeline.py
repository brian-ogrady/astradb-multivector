#!/usr/bin/env python3
"""
Unit tests for the LateInteractionPipeline class.
"""

import unittest
import uuid
import json
import asyncio
import torch
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

from astra_multivector.late_interaction import LateInteractionPipeline


class MockLateInteractionModel:
    """Mock model for testing the pipeline."""
    
    def __init__(self, dim=128):
        self._dim = dim
        self._model_name = "mock_model"
        
    async def encode_query(self, q):
        """Return mock query embeddings."""
        # Return a simple tensor for predictable results
        tensor = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        return tensor[:, :self._dim]
    
    async def encode_doc(self, chunks):
        """Return mock document embeddings."""
        # Return a list of simple tensors
        if len(chunks) == 1:
            tensor = torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=torch.float32)
            return [tensor[:, :self._dim]]
        else:
            return [torch.randn(3, self._dim) for _ in range(len(chunks))]
    
    def score(self, Q, D):
        """Calculate mock scores."""
        # Simple scoring for testing: use the sum of cosine similarities
        scores = []
        for doc in D:
            # Calculate similarity between each query token and each doc token
            similarities = torch.matmul(Q, doc.transpose(0, 1))
            # Take max similarity for each query token
            max_sims = torch.max(similarities, dim=1)[0]
            # Sum across query tokens
            score = torch.sum(max_sims)
            scores.append(score)
        return torch.tensor(scores)
    
    def _embeddings_to_numpy(self, embeddings):
        """Convert embeddings to numpy."""
        return embeddings.numpy()
    
    def _numpy_to_embeddings(self, array):
        """Convert numpy to embeddings."""
        return torch.from_numpy(array).float()
    
    @property
    def dim(self):
        """Return model dimension."""
        return self._dim


class MockAsyncCursor:
    """Mock cursor for testing database operations."""
    
    def __init__(self, items=None):
        self.items = items or []
    
    async def to_list(self):
        """Return a list of items."""
        return self.items


class MockAsyncTable:
    """Mock table for testing database operations."""
    
    def __init__(self, name="mock_table"):
        self.name = name
        self.items = []
        self.find = AsyncMock()
        self.find.return_value = MockAsyncCursor()
        self.insert_one = AsyncMock()
        self.delete = AsyncMock()
        self.delete.return_value = MagicMock(deleted_count=1)
        
    async def create_if_not_exists(self):
        """Mock table creation."""
        return self


class MockAsyncDatabase:
    """Mock database for testing."""
    
    def __init__(self):
        self.tables = {}
        
    async def create_table(self, table_name, definition=None, if_not_exists=False):
        """Create a mock table."""
        if table_name not in self.tables:
            self.tables[table_name] = MockAsyncTable(table_name)
        return self.tables[table_name]


class TestLateInteractionPipeline(unittest.TestCase):
    """Tests for the LateInteractionPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.db = MockAsyncDatabase()
        self.model = MockLateInteractionModel(dim=3)
        
        # Create pipeline
        self.pipeline = LateInteractionPipeline(
            db=self.db,
            model=self.model,
            base_table_name="test_pipeline",
            doc_pool_factor=2,
            query_pool_distance=0.05,
            sim_metric="cosine",
            default_concurrency_limit=5
        )
    
    async def async_setup(self):
        """Async setup for tests that need initialization."""
        await self.pipeline.initialize()
    
    @patch('astra_multivector.late_interaction.late_interaction_pipeline.CreateTableDefinition')
    @patch('astra_multivector.late_interaction.late_interaction_pipeline.AsyncAstraMultiVectorTable')
    async def test_initialize(self, mock_multi_vector_table, mock_create_table_def):
        """Test pipeline initialization."""
        # Set up mocks
        mock_builder = MagicMock()
        mock_create_table_def.builder.return_value = mock_builder
        mock_builder.build.return_value = {}
        
        mock_token_table = MagicMock()
        mock_multi_vector_table.return_value = mock_token_table
        
        # Call the method
        await self.pipeline.initialize()
        
        # Check that tables were created
        self.assertTrue(self.pipeline._initialized)
        self.assertIsNotNone(self.pipeline._doc_table)
        self.assertIsNotNone(self.pipeline._token_table)
        
        # Verify that create_table was called for doc table
        self.assertEqual(mock_create_table_def.builder.call_count, 1)
        
        # Verify that AsyncAstraMultiVectorTable was created for token table
        mock_multi_vector_table.assert_called_once()
    
    async def test_index_document(self):
        """Test document indexing."""
        # Create a test document
        content = "Test document"
        metadata = {"key": "value"}
        doc_id = uuid.uuid4()
        
        # Initialize the pipeline
        await self.async_setup()
        
        # Mock the insert methods
        self.pipeline._doc_table.insert_one = AsyncMock()
        self.pipeline._index_token_embeddings = AsyncMock(return_value=[uuid.uuid4()])
        
        # Call the method
        result_id = await self.pipeline.index_document(content, metadata, doc_id)
        
        # Verify the results
        self.assertEqual(result_id, doc_id)
        self.pipeline._doc_table.insert_one.assert_called_once()
        self.pipeline._index_token_embeddings.assert_called_once()
    
    async def test_bulk_index_documents(self):
        """Test bulk document indexing."""
        # Create test documents
        contents = ["Doc 1", "Doc 2", "Doc 3"]
        metadata_list = [{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]
        doc_ids = [uuid.uuid4() for _ in range(3)]
        
        # Initialize the pipeline
        await self.async_setup()
        
        # Mock the index_document method
        self.pipeline.index_document = AsyncMock(side_effect=doc_ids)
        
        # Call the method
        result_ids = await self.pipeline.bulk_index_documents(
            documents=contents,
            metadata_list=metadata_list,
            doc_ids=doc_ids,
            max_concurrency=2,
            batch_size=2
        )
        
        # Verify the results
        self.assertEqual(result_ids, doc_ids)
        self.assertEqual(self.pipeline.index_document.call_count, 3)
    
    async def test_search(self):
        """Test search functionality."""
        # Initialize the pipeline
        await self.async_setup()
        
        # Create mock document embeddings
        doc_embeddings = [
            torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ], dtype=torch.float32)
        ]
        
        # Mock methods
        self.pipeline._search_with_embeddings = AsyncMock(
            return_value=[
                (uuid.uuid4(), 0.95, {"title": "Doc 1"}),
                (uuid.uuid4(), 0.85, {"title": "Doc 2"})
            ]
        )
        
        # Call the method
        results = await self.pipeline.search("test query", k=2)
        
        # Verify the results
        self.assertEqual(len(results), 2)
        self.pipeline._search_with_embeddings.assert_called_once()
    
    async def test_get_document(self):
        """Test document retrieval."""
        # Initialize the pipeline
        await self.async_setup()
        
        # Create a test document ID
        doc_id = uuid.uuid4()
        
        # Mock the find method
        mock_doc = {
            "doc_id": str(doc_id),
            "content": "Test document",
            "metadata": json.dumps({"key": "value"})
        }
        self.pipeline._doc_table.find.return_value = MockAsyncCursor([mock_doc])
        
        # Call the method
        result = await self.pipeline.get_document(doc_id)
        
        # Verify the results
        self.assertIsNotNone(result)
        self.assertEqual(result["doc_id"], doc_id)
        self.assertEqual(result["content"], "Test document")
        self.assertEqual(result["metadata"], {"key": "value"})
    
    async def test_delete_document(self):
        """Test document deletion."""
        # Initialize the pipeline
        await self.async_setup()
        
        # Create a test document ID
        doc_id = uuid.uuid4()
        
        # Mock the find method for token table
        mock_tokens = [{"chunk_id": str(uuid.uuid4())}]
        self.pipeline._token_table.table.find.return_value = MockAsyncCursor(mock_tokens)
        
        # Call the method
        result = await self.pipeline.delete_document(doc_id)
        
        # Verify the results
        self.assertTrue(result)
        self.pipeline._doc_table.delete.assert_called_once()
        self.pipeline._token_table.table.delete.assert_called_once()
    
    async def test_load_doc_token_embeddings(self):
        """Test loading document token embeddings."""
        # Initialize the pipeline
        await self.async_setup()
        
        # Create test document IDs
        doc_ids = [uuid.uuid4()]
        
        # Mock token embeddings
        mock_tokens = [
            {"content": f"{doc_ids[0]}:0", "token_embedding": [1.0, 0.0, 0.0]},
            {"content": f"{doc_ids[0]}:1", "token_embedding": [0.0, 1.0, 0.0]},
            {"content": f"{doc_ids[0]}:2", "token_embedding": [0.0, 0.0, 1.0]}
        ]
        self.pipeline._token_table.table.find.return_value = MockAsyncCursor(mock_tokens)
        
        # Call the method
        result = await self.pipeline._load_doc_token_embeddings(doc_ids)
        
        # Verify the results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, torch.Size([3, 3]))  # 3 tokens x 3 dimensions
    
    async def test_encode_query(self):
        """Test query encoding with pooling."""
        # Create a pipeline with query pooling disabled
        pipeline_no_pooling = LateInteractionPipeline(
            db=self.db,
            model=self.model,
            base_table_name="test_no_pooling",
            query_pool_distance=0  # Disable pooling
        )
        
        # Call the methods
        result_no_pooling = await pipeline_no_pooling.encode_query("test query")
        result_with_pooling = await self.pipeline.encode_query("test query")
        
        # Verify that without pooling we get the original embeddings
        self.assertEqual(result_no_pooling.shape, torch.Size([3, 3]))
        
        # With pooling, should be the same or fewer tokens
        self.assertLessEqual(result_with_pooling.shape[0], result_no_pooling.shape[0])


def async_test(coro):
    """Decorator for running async tests."""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


# Apply async_test decorator to all test methods
for attr_name in dir(TestLateInteractionPipeline):
    if attr_name.startswith('test_'):
        attr = getattr(TestLateInteractionPipeline, attr_name)
        if asyncio.iscoroutinefunction(attr):
            setattr(TestLateInteractionPipeline, attr_name, async_test(attr))


if __name__ == "__main__":
    unittest.main()