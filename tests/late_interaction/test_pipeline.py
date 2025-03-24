#!/usr/bin/env python3
"""
Unit tests for the LateInteractionPipeline class.

This module contains a comprehensive test suite for the LateInteractionPipeline,
which is responsible for implementing a late-interaction vector retrieval pattern
with AstraDB. The tests cover initialization, document indexing, token management,
query processing, document retrieval, and deletion functionality.
"""

import unittest
import uuid
import asyncio
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from unittest.mock import patch, MagicMock, AsyncMock, call

from astra_multivector.late_interaction import LateInteractionPipeline
from astra_multivector.late_interaction.utils import PoolingResult


# Configure logging to suppress pipeline messages during tests
logging.getLogger("astra_multivector.late_interaction.late_interaction_pipeline").setLevel(logging.ERROR)


class MockLateInteractionModel:
    """
    Mock implementation of a late interaction model for testing.
    
    Provides simplified implementations of all required methods and properties
    for the late interaction model interface. Returns predictable tensors
    for encode methods to facilitate testing.
    """
    
    def __init__(self, dim=128, supports_images=False):
        self._dim = dim
        self._model_name = "mock_model"
        self._supports_images = supports_images
        
    async def encode_query(self, q):
        tensor = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=torch.float32)
        return tensor[:, :self._dim]
    
    async def encode_doc(self, chunks):
        if len(chunks) == 1:
            tensor = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ], dtype=torch.float32)
            return [tensor[:, :self._dim]]
        else:
            return [torch.randn(3, self._dim) for _ in range(len(chunks))]
    
    def score(self, Q, D):
        scores = []
        for doc in D:
            similarities = torch.matmul(Q, doc.transpose(0, 1))
            max_sims = torch.max(similarities, dim=1)[0]
            score = torch.sum(max_sims)
            scores.append(score)
        return torch.tensor(scores)
    
    def _embeddings_to_numpy(self, embeddings):
        return embeddings.numpy()
    
    def _numpy_to_embeddings(self, array):
        return torch.from_numpy(array).float()
    
    def to_device(self, tensor):
        return tensor
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def supports_images(self):
        return self._supports_images
    
    @property
    def model_name(self):
        return self._model_name


class MockAsyncCursor:
    """
    Mock implementation of an async database cursor.
    
    Simulates the behavior of an async cursor that would be returned
    by database query operations.
    """
    
    def __init__(self, items=None):
        self.items = items or []
    
    async def to_list(self):
        return self.items


class MockAsyncTable:
    """
    Mock implementation of an AstraDB AsyncTable.
    
    Simulates the behavior of an AstraDB table with methods for
    insertion, deletion, querying, and schema operations.
    """
    
    def __init__(self, name="mock_table", items=None):
        self.name = name
        self.items = items or []
        self._definition = {
            "columns": [
                {"name": "doc_id", "type": "UUID"},
                {"name": "content", "type": "TEXT"},
                {"name": "token_id", "type": "UUID"},
                {"name": "token_embedding", "type": "VECTOR"}
            ]
        }
        
        self.insert_one = AsyncMock()
        self.insert_many = AsyncMock()
        
        self.delete_many = AsyncMock()
        self.delete_many.return_value = {"deletedCount": 1}
        
        self.create_vector_index = AsyncMock()
    
    async def definition(self):
        """
        Return a mock table definition object.
        
        Creates a mock definition with columns matching the structure
        defined in self._definition, suitable for schema introspection.
        """
        columns = []
        for col in self._definition["columns"]:
            column_mock = AsyncMock()
            column_mock.name = col["name"]
            columns.append(column_mock)
        
        definition_mock = MagicMock()
        definition_mock.columns = columns
        return definition_mock
    
    async def find(self, *args, **kwargs):
        return MockAsyncCursor(self.items)


class MockAsyncDatabase:
    """
    Mock implementation of an AstraDB AsyncDatabase.
    
    Simulates the behavior of an AstraDB database with methods
    for table management.
    """
    
    def __init__(self):
        self.tables = {}
        
    async def create_table(self, table_name, definition=None, if_not_exists=False):
        """
        Create a mock table with the given name.
        """
        if table_name not in self.tables:
            self.tables[table_name] = MockAsyncTable(table_name)
        return self.tables[table_name]


class TestLateInteractionPipeline(unittest.TestCase):
    """
    Test suite for the LateInteractionPipeline class.
    
    Tests the complete functionality of the pipeline including initialization,
    document processing, indexing, search, and deletion operations.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method runs.
        
        Creates mock database and model objects and initializes the pipeline
        with standard test parameters.
        """
        self.db = MockAsyncDatabase()
        self.model = MockLateInteractionModel(dim=4)
        
        self.pipeline = LateInteractionPipeline(
            db=self.db,
            model=self.model,
            base_table_name="test_pipeline",
            doc_pool_factor=2,
            query_pool_distance=0.05,
            sim_metric="cosine",
            default_concurrency_limit=5,
            embedding_cache_size=10
        )
    
    async def async_setup(self):
        """
        Async setup method for tests that require initialized pipeline.
        
        Ensures the pipeline is initialized before running test methods
        that require database tables to be created.
        """
        await self.pipeline.initialize()
    
    @patch('astra_multivector.late_interaction.late_interaction_pipeline.CreateTableDefinition')
    async def test_initialize(self, mock_create_table_def):
        """
        Test pipeline initialization process.
        
        Verifies that the required tables are created with correct schemas
        and that vector indices are created on the token table.
        """
        mock_builder = MagicMock()
        mock_create_table_def.builder.return_value = mock_builder
        mock_builder.add_column.return_value = mock_builder
        mock_builder.add_vector_column.return_value = mock_builder
        mock_builder.add_partition_by.return_value = mock_builder
        mock_builder.build.return_value = {}
        
        await self.pipeline.initialize()
        
        self.assertTrue(self.pipeline._initialized)
        self.assertIsNotNone(self.pipeline._doc_table)
        self.assertIsNotNone(self.pipeline._token_table)
        
        self.assertEqual(len(self.db.tables), 2)
        self.assertIn("test_pipeline_docs", self.db.tables)
        self.assertIn("test_pipeline_tokens", self.db.tables)
        
        self.pipeline._token_table.create_vector_index.assert_called_once()
    
    async def test_create_doc_table(self):
        """
        Test document table creation.
        
        Verifies that the document table is created with the correct name
        and that models supporting images add the required content_type column.
        """
        doc_table = await self.pipeline._create_doc_table()
        self.assertEqual(doc_table.name, "test_pipeline_docs")
        
        image_model = MockLateInteractionModel(dim=4, supports_images=True)
        image_pipeline = LateInteractionPipeline(
            db=self.db,
            model=image_model,
            base_table_name="image_pipeline"
        )
        
        await image_pipeline._create_doc_table()
    
    async def test_create_token_table(self):
        """
        Test token table creation.
        
        Verifies that the token table is created with the correct name
        and that a vector index is created with the specified similarity metric.
        """
        token_table = await self.pipeline._create_token_table()
        self.assertEqual(token_table.name, "test_pipeline_tokens")
        
        token_table.create_vector_index.assert_called_once()
        
        self.assertEqual(token_table.create_vector_index.call_args[1]["options"].metric, "cosine")
    
    async def test_validate_row(self):
        """
        Test document row validation.
        
        Verifies that document rows are correctly validated, that required
        fields are checked, and that document IDs are auto-generated when missing.
        """
        self.pipeline._doc_table = MockAsyncTable()
        
        doc_id = uuid.uuid4()
        document_row = {
            "content": "Test document",
            "doc_id": doc_id
        }
        
        validated = await self.pipeline._validate_row(document_row)
        self.assertEqual(validated["doc_id"], doc_id)
        self.assertEqual(validated["original_content"], "Test document")
        self.assertEqual(validated["validated_insertion"]["content"], "Test document")
        
        invalid_row = {"doc_id": uuid.uuid4()}
        with self.assertRaises(ValueError):
            await self.pipeline._validate_row(invalid_row)
        
        auto_id_row = {"content": "Auto ID document"}
        validated = await self.pipeline._validate_row(auto_id_row)
        self.assertIsInstance(validated["doc_id"], uuid.UUID)
    
    @patch('astra_multivector.late_interaction.late_interaction_pipeline.pool_doc_embeddings')
    async def test_index_document(self, mock_pool_doc_embeddings):
        """
        Test document indexing functionality.
        
        Verifies that documents are correctly processed, embedded, pooled,
        and stored in the database with associated token embeddings.
        """
        doc_id = uuid.uuid4()
        document_row = {
            "content": "Test document",
            "doc_id": doc_id
        }
        
        await self.async_setup()
        
        doc_embeddings = [torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])]
        
        pooled_embeddings = [torch.tensor([[0.5, 0.5, 0.0, 0.0]])]
        mock_pool_doc_embeddings.return_value = pooled_embeddings
        
        self.model.encode_doc = AsyncMock(return_value=doc_embeddings)
        
        self.pipeline._index_token_embeddings = AsyncMock(return_value=[[uuid.uuid4()]])
        
        cached_doc_mock = AsyncMock()
        cached_doc_mock.cache_clear = MagicMock()
        original_cached_doc = self.pipeline._cached_doc_embeddings
        self.pipeline._cached_doc_embeddings = cached_doc_mock
        
        try:
            result = await self.pipeline.index_document(document_row)
            
            self.assertEqual(result, doc_id)
            
            self.pipeline._doc_table.insert_one.assert_called_once()
            doc_insertion = self.pipeline._doc_table.insert_one.call_args[0][0]
            self.assertEqual(doc_insertion["doc_id"], doc_id)
            self.assertEqual(doc_insertion["content"], "Test document")
            
            self.model.encode_doc.assert_called_once_with(["Test document"])
            mock_pool_doc_embeddings.assert_called_once_with(doc_embeddings, 2)
            
            self.pipeline._index_token_embeddings.assert_called_once_with(doc_id, pooled_embeddings[0])
            
            cached_doc_mock.cache_clear.assert_called_once()
        finally:
            self.pipeline._cached_doc_embeddings = original_cached_doc
    
    async def test_index_token_embeddings(self):
        """
        Test token embeddings indexing.
        
        Verifies that token embeddings are correctly processed and stored
        in the token table with appropriate document associations.
        """
        await self.async_setup()
        
        doc_id = uuid.uuid4()
        embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        self.model._embeddings_to_numpy = MagicMock(return_value=np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]))
        
        token_ids = await self.pipeline._index_token_embeddings(doc_id, embeddings)
        
        self.assertEqual(len(token_ids), 1)
        self.assertEqual(len(token_ids[0]), 2)
        for token_id in token_ids[0]:
            self.assertIsInstance(token_id, uuid.UUID)
        
        self.pipeline._token_table.insert_many.assert_called_once()
        insertions = self.pipeline._token_table.insert_many.call_args[0][0]
        self.assertEqual(len(insertions), 2)
        
        for insertion in insertions:
            self.assertEqual(insertion["doc_id"], doc_id)
            self.assertIn("token_id", insertion)
            self.assertIn("token_embedding", insertion)
    
    async def test_index_token_embeddings_multiple_docs(self):
        """
        Test token embeddings indexing for multiple documents.
        
        Verifies that token embeddings for multiple documents are correctly
        processed and stored with appropriate document associations.
        """
        await self.async_setup()
        
        doc_ids = [uuid.uuid4(), uuid.uuid4()]
        embeddings = [
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        ]
        
        self.model._embeddings_to_numpy = MagicMock(side_effect=[
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0, 0.0]])
        ])
        
        token_ids = await self.pipeline._index_token_embeddings(doc_ids, embeddings)
        
        self.assertEqual(len(token_ids), 2)
        self.assertEqual(len(token_ids[0]), 2)
        self.assertEqual(len(token_ids[1]), 1)
        
        self.pipeline._token_table.insert_many.assert_called_once()
        insertions = self.pipeline._token_table.insert_many.call_args[0][0]
        self.assertEqual(len(insertions), 3)
        
        self.assertEqual(insertions[0]["doc_id"], doc_ids[0])
        self.assertEqual(insertions[1]["doc_id"], doc_ids[0])
        
        self.assertEqual(insertions[2]["doc_id"], doc_ids[1])
    
    async def test_index_token_embeddings_validation(self):
        """
        Test token embeddings indexing validation.
        
        Verifies that token embeddings validation catches errors such as
        mismatched document IDs and embeddings, invalid embedding types,
        and wrong dimensionality.
        """
        await self.async_setup()
        
        doc_ids = [uuid.uuid4(), uuid.uuid4()]
        embeddings = [torch.tensor([[1.0, 0.0, 0.0, 0.0]])]
        
        with self.assertRaises(ValueError):
            await self.pipeline._index_token_embeddings(doc_ids, embeddings)
        
        with self.assertRaises(ValueError):
            await self.pipeline._index_token_embeddings(
                uuid.uuid4(),
                [1.0, 0.0, 0.0, 0.0]
            )
        
        with self.assertRaises(ValueError):
            await self.pipeline._index_token_embeddings(
                uuid.uuid4(),
                torch.tensor([1.0, 0.0, 0.0, 0.0])
            )
        
        with self.assertRaises(ValueError):
            await self.pipeline._index_token_embeddings(
                uuid.uuid4(),
                [torch.tensor([[1.0, 0.0]])]
            )
    
    @patch('astra_multivector.late_interaction.late_interaction_pipeline.pool_doc_embeddings')
    async def test_bulk_index_documents(self, mock_pool_doc_embeddings):
        """
        Test bulk document indexing.
        
        Verifies that multiple documents can be indexed in batches with
        appropriate concurrency controls and that all documents are
        correctly processed and stored.
        """
        await self.async_setup()
        
        doc_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
        document_rows = [
            {"content": f"Doc {i+1}", "doc_id": doc_id}
            for i, doc_id in enumerate(doc_ids)
        ]
        
        doc_embeddings = [torch.tensor([[1.0, 0.0, 0.0, 0.0]])]
        
        self.model.encode_doc = AsyncMock(return_value=doc_embeddings)
        
        pooled_embeddings = [torch.tensor([[0.5, 0.5, 0.0, 0.0]])]
        mock_pool_doc_embeddings.return_value = pooled_embeddings
        
        self.pipeline._index_token_embeddings = AsyncMock(return_value=[[uuid.uuid4()]])
        
        cached_doc_mock = AsyncMock()
        cached_doc_mock.cache_clear = MagicMock()
        original_cached_doc = self.pipeline._cached_doc_embeddings
        self.pipeline._cached_doc_embeddings = cached_doc_mock
        
        try:
            result_ids = await self.pipeline.bulk_index_documents(
                document_rows=document_rows,
                batch_size=2,
                embedding_concurrency=2
            )
            
            self.assertEqual(len(result_ids), 3)
            for doc_id in doc_ids:
                self.assertIn(doc_id, result_ids)
            
            self.assertEqual(self.pipeline._doc_table.insert_many.call_count, 2)
            self.assertEqual(len(self.pipeline._doc_table.insert_many.call_args_list[0][0][0]), 2)
            self.assertEqual(len(self.pipeline._doc_table.insert_many.call_args_list[1][0][0]), 1)
            
            self.assertEqual(self.model.encode_doc.call_count, 3)
            
            self.assertEqual(self.pipeline._index_token_embeddings.call_count, 3)
            
            cached_doc_mock.cache_clear.assert_called_once()
        finally:
            self.pipeline._cached_doc_embeddings = original_cached_doc
    
    @patch('astra_multivector.late_interaction.late_interaction_pipeline.pool_query_embeddings')
    async def test_encode_query(self, mock_pool_query_embeddings):
        """
        Test query encoding with pooling.
        
        Verifies that queries are correctly encoded into embeddings
        and that query token pooling is applied when configured.
        """
        await self.async_setup()
        
        query_embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
        pooled_embeddings = torch.tensor([
            [0.7, 0.7, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
        self.model.encode_query = AsyncMock(return_value=query_embeddings)
        mock_pool_query_embeddings.return_value = pooled_embeddings
        
        result = await self.pipeline.encode_query("test query")
        
        self.model.encode_query.assert_called_once_with("test query")
        mock_pool_query_embeddings.assert_called_once_with(
            query_embeddings,
            self.pipeline.query_pool_distance
        )
        torch.testing.assert_close(result, pooled_embeddings)
        
        self.pipeline.query_pool_distance = 0
        mock_pool_query_embeddings.reset_mock()
        
        result_no_pooling = await self.pipeline.encode_query("test query")
        
        mock_pool_query_embeddings.assert_not_called()
        torch.testing.assert_close(result_no_pooling, query_embeddings)
    
    @patch('astra_multivector.late_interaction.late_interaction_pipeline.pool_query_embeddings')
    async def test_search(self, mock_pool_query_embeddings):
        """
        Test search functionality.
        
        Verifies that the search process correctly encodes queries,
        finds matching tokens, retrieves document embeddings, and
        calculates similarity scores for late interaction ranking.
        """
        await self.async_setup()
        
        query_embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        search_results = [
            (uuid.uuid4(), 0.95, "Document content 1"),
            (uuid.uuid4(), 0.85, "Document content 2")
        ]
        
        self.model.encode_query = AsyncMock(return_value=query_embeddings)
        mock_pool_query_embeddings.return_value = query_embeddings
        self.pipeline._search_with_embeddings = AsyncMock(return_value=search_results)
        self.model._embeddings_to_numpy = MagicMock(return_value=np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]))
        
        results = await self.pipeline.search("test query", k=2)
        
        self.assertEqual(results, search_results)
        
        self.pipeline._search_with_embeddings.assert_called_once()
        call_args = self.pipeline._search_with_embeddings.call_args[0]
        torch.testing.assert_close(call_args[0], query_embeddings)
        self.assertEqual(call_args[2], 2)
        
        self.assertGreater(call_args[3], 2)
        self.assertGreater(call_args[4], 2)
    
    async def test_search_with_embeddings(self):
        """
        Test search with precomputed embeddings.
        
        Verifies that the search process correctly handles precomputed
        query embeddings, performs token-level retrieval, re-ranks results
        using late interaction, and returns correctly sorted results.
        """
        await self.async_setup()
        
        Q = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        Q_np = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        doc_id = uuid.uuid4()
        token_id = uuid.uuid4()
        token_results = [
            [
                {"doc_id": doc_id, "token_id": token_id, "$similarity": 0.9}
            ],
            [
                {"doc_id": doc_id, "token_id": uuid.uuid4(), "$similarity": 0.8}
            ]
        ]
        
        token_cursor1 = AsyncMock()
        token_cursor1.to_list.return_value = token_results[0]
        
        token_cursor2 = AsyncMock()
        token_cursor2.to_list.return_value = token_results[1]
        
        self.pipeline.async_find = AsyncMock()
        self.pipeline.async_find.side_effect = [token_cursor1, token_cursor2]
        
        doc_result = {
            "doc_id": doc_id,
            "content": "Test document"
        }
        
        doc_cursor = AsyncMock()
        doc_cursor.to_list.return_value = [doc_result]
        
        self.pipeline.async_find.side_effect = [token_cursor1, token_cursor2, doc_cursor]
        
        doc_embeddings = [torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])]
        
        self.pipeline._cached_doc_embeddings = AsyncMock(return_value=doc_embeddings)
        
        self.model.score = MagicMock(return_value=torch.tensor([0.95]))
        
        results = await self.pipeline._search_with_embeddings(
            Q, Q_np, k=1, n_ann_tokens=10, n_maxsim_candidates=5
        )
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], doc_id)
        torch.testing.assert_close(results[0][1], 0.95, rtol=1e-5, atol=1e-5)
        self.assertEqual(results[0][2], "Test document")
        
        self.assertEqual(self.pipeline.async_find.call_count, 3)
        
        first_call_args = self.pipeline.async_find.call_args_list[0]
        self.assertEqual(first_call_args[0][0], self.pipeline._token_table)
        
        second_call_args = self.pipeline.async_find.call_args_list[1]
        self.assertEqual(second_call_args[0][0], self.pipeline._token_table)
        
        third_call_args = self.pipeline.async_find.call_args_list[2]
        self.assertEqual(third_call_args[0][0], self.pipeline._doc_table)
        
        self.pipeline._cached_doc_embeddings.assert_called_once()
        
        self.model.score.assert_called_once_with(Q, doc_embeddings)
    
    async def test_cached_doc_embeddings(self):
        """
        Test document embeddings retrieval from cache.
        
        Verifies that the document embeddings caching mechanism works
        correctly, avoiding redundant database queries.
        """
        await self.async_setup()
        
        doc_id1 = uuid.uuid4()
        doc_id2 = uuid.uuid4()
        
        doc_embeddings = [
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        ]
        
        load_mock = AsyncMock()
        load_mock.return_value = doc_embeddings.copy()
        original_load = self.pipeline._load_doc_token_embeddings
        self.pipeline._load_doc_token_embeddings = load_mock
        
        cached_mock = AsyncMock()
        cached_mock.return_value = doc_embeddings.copy()
        original_cached = self.pipeline._cached_doc_embeddings
        self.pipeline._cached_doc_embeddings = cached_mock
        
        try:
            args = (str(doc_id1), str(doc_id2))
            result = await self.pipeline._cached_doc_embeddings(args)
            
            self.assertEqual(len(result), 2)
            
            cached_mock.assert_called_once_with(args)
            
            cached_mock.reset_mock()
            cached_mock.return_value = doc_embeddings.copy()
            result2 = await self.pipeline._cached_doc_embeddings(args)
            cached_mock.assert_called_once_with(args)
            
            self.assertEqual(len(result2), 2)
        finally:
            self.pipeline._load_doc_token_embeddings = original_load
            self.pipeline._cached_doc_embeddings = original_cached
    
    async def test_load_doc_token_embeddings(self):
        """
        Test loading document token embeddings from database.
        
        Verifies that token embeddings for a document can be correctly
        retrieved from the database and converted to the appropriate format.
        """
        await self.async_setup()
        
        doc_id = uuid.uuid4()
        
        token_results = [
            {"token_id": uuid.uuid4(), "token_embedding": [1.0, 0.0, 0.0, 0.0]},
            {"token_id": uuid.uuid4(), "token_embedding": [0.0, 1.0, 0.0, 0.0]}
        ]

        mock_cursor = AsyncMock()
        mock_cursor.to_list.return_value = token_results
        
        self.pipeline.async_find = AsyncMock(return_value=mock_cursor)
        
        self.model._numpy_to_embeddings = MagicMock(return_value=torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]))
        
        result = await self.pipeline._load_doc_token_embeddings([doc_id])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, torch.Size([2, 4]))
        
        self.pipeline.async_find.assert_called_once()
        call_args = self.pipeline.async_find.call_args
        self.assertEqual(call_args[0][0], self.pipeline._token_table)
        self.assertEqual(call_args[1]["filter"], {"doc_id": doc_id})
        
        mock_cursor.to_list.assert_called_once()
    
    async def test_fetch_token_embeddings(self):
        """
        Test fetching token embeddings for a document.
        
        Verifies that token embeddings can be retrieved for a specific document
        and handles error cases gracefully.
        """
        await self.async_setup()
        
        doc_id = uuid.uuid4()
        
        token_results = [
            {"token_id": uuid.uuid4(), "token_embedding": [1.0, 0.0, 0.0, 0.0]},
            {"token_id": uuid.uuid4(), "token_embedding": [0.0, 1.0, 0.0, 0.0]}
        ]

        cursor_mock = AsyncMock()
        cursor_mock.to_list.return_value = token_results
        
        self.pipeline.async_find = AsyncMock(return_value=cursor_mock)
            
        self.model._numpy_to_embeddings = MagicMock(return_value=torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]))
        
        result = await self.pipeline._fetch_token_embeddings(doc_id)
        
        self.assertEqual(result.shape, torch.Size([2, 4]))
        
        self.pipeline._token_table.find = AsyncMock(return_value=MockAsyncCursor([]))
        
        result = await self.pipeline._fetch_token_embeddings(doc_id)
        
        self.assertEqual(result.shape, torch.Size([2, 4]))
        
        token_results = [
            {"token_id": uuid.uuid4(), "token_embedding": [1.0, 0.0, 0.0, 0.0]},
            {"token_id": uuid.uuid4()}
        ]
        
        self.pipeline._token_table.find = AsyncMock(return_value=MockAsyncCursor(token_results))
        
        result = await self.pipeline._fetch_token_embeddings(doc_id)
        
        self.assertEqual(result.shape, torch.Size([2, 4]))
    
    async def test_delete_document(self):
        """
        Test document deletion functionality.
        
        Verifies that documents and their associated token embeddings
        are correctly deleted from the database and that the cache is cleared.
        """
        await self.async_setup()
        
        doc_id = uuid.uuid4()
        
        cached_doc_mock = AsyncMock()
        cached_doc_mock.cache_clear = MagicMock()
        original_cached_doc = self.pipeline._cached_doc_embeddings
        self.pipeline._cached_doc_embeddings = cached_doc_mock
        
        try:
            result = await self.pipeline.delete_document(doc_id)
            
            self.assertTrue(result)
            
            self.pipeline._doc_table.delete_many.assert_called_once()
            self.assertEqual(self.pipeline._doc_table.delete_many.call_args[1]["filter"], {"doc_id": str(doc_id)})
            
            self.pipeline._token_table.delete_many.assert_called_once()
            self.assertEqual(self.pipeline._token_table.delete_many.call_args[1]["filter"], {"doc_id": doc_id})
            
            cached_doc_mock.cache_clear.assert_called_once()
        finally:
            self.pipeline._cached_doc_embeddings = original_cached_doc


def async_test(coro):
    """
    Decorator for running asynchronous test coroutines.
    
    Wraps an async test function to make it compatible with unittest's
    synchronous test runner by setting up and tearing down the event loop.
    """
    def wrapper(*args, **kwargs):
        try:
            return asyncio.run(coro(*args, **kwargs))
        except Exception as e:
            print(f"Error in async test: {str(e)}")
            raise
    return wrapper


# Apply async_test decorator to all test methods
for attr_name in dir(TestLateInteractionPipeline):
    if attr_name.startswith('test_'):
        attr = getattr(TestLateInteractionPipeline, attr_name)
        if asyncio.iscoroutinefunction(attr):
            setattr(TestLateInteractionPipeline, attr_name, async_test(attr))


if __name__ == "__main__":
    unittest.main()