import asyncio
import warnings
from typing import List, Dict, Any, Optional, Union

import torch
from colbert import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.infra import ColBERTConfig
from PIL.Image import Image

from astra_multivector.late_interaction import LateInteractionModel


class ColBERTModel(LateInteractionModel):
    """
    ColBERT implementation of the LateInteractionModel interface.
    
    Uses the ColBERT neural IR model for token-level late interaction retrieval.
    ColBERT provides dense retrieval with fine-grained token-level interactions
    between query and document tokens at search time, offering high precision
    while maintaining efficiency.
    """
    
    def __init__(
        self, 
        model_name: str = 'answerdotai/answerai-colbert-small-v1',
        tokens_per_query: int = 32,
        max_doc_tokens: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize a ColBERT model.
        
        Args:
            model_name: HuggingFace model name or path to local checkpoint.
                       Default is 'answerdotai/answerai-colbert-small-v1'.
            tokens_per_query: Maximum number of tokens per query.
                             Default is 32 tokens.
            max_doc_tokens: Maximum number of tokens per document.
                          Default is 512 tokens.
            device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.)
                   If None, will use CUDA if available, otherwise CPU.
        """
        super().__init__(device=device)

        self._model_name = model_name
        self._tokens_per_query = tokens_per_query
        self._max_doc_tokens = max_doc_tokens
                
        self.config = ColBERTConfig(
            checkpoint=model_name,
            query_maxlen=tokens_per_query,
            doc_maxlen=max_doc_tokens,
        )
        
        self.checkpoint = Checkpoint(
            self.config.checkpoint, 
            colbert_config=self.config,
        )
        
        try:
            self.checkpoint.model = self.checkpoint.model.to(self._device)
        except RuntimeError as e:
            warnings.warn(f"Could not move model to {self._device}: {e}."
                          f"Use {self._get_actual_device(self.checkpoint)} instead.")
                
        self.encoder = CollectionEncoder(self.config, self.checkpoint)
        
        self._embedding_dim = self.config.dim
    
    async def encode_query(self, q: str) -> torch.Tensor:
        """
        Encode a query string into token embeddings asynchronously.
        
        Offloads the synchronous encoding work to a separate thread to avoid
        blocking the event loop.
        
        Args:
            q: The query string to encode
            
        Returns:
            Query token embeddings tensor of shape (num_tokens, embedding_dim)
        """
        
        return await asyncio.to_thread(self.encode_query_sync, q)
    
    def encode_query_sync(self, q: str) -> torch.Tensor:
        """
        Encode a query string into token embeddings synchronously.
        
        For empty queries, returns an empty tensor with the correct embedding dimension.
        For non-empty queries, uses the ColBERT queryFromText method to generate embeddings.
        
        Args:
            q: The query string to encode
            
        Returns:
            Query token embeddings tensor of shape (num_tokens, embedding_dim)
        """
        if not q.strip():
            return torch.zeros((0, self.dim), device=self._device)
            
        return self.checkpoint.queryFromText([q])[0]
    
    async def encode_doc(self, chunks: List[Union[str, Image]]) -> List[torch.Tensor]:
        """
        Encode document chunks into token embeddings asynchronously.
        
        Offloads the synchronous encoding work to a separate thread to avoid
        blocking the event loop. Validates that all inputs are text chunks
        since ColBERT doesn't support image inputs.
        
        Args:
            chunks: List of text chunks to encode
            
        Returns:
            List of token embedding tensors, one per chunk
            
        Raises:
            TypeError: If any chunk is not a string
        """
        if not chunks:
            return []
            
        if not all(isinstance(chunk, str) for chunk in chunks):
            raise TypeError("ColBERT only supports text chunks")
        
        return await asyncio.to_thread(self.encode_doc_sync, chunks)
    
    def encode_doc_sync(self, chunks: List[str]) -> List[torch.Tensor]:
        """
        Encode document chunks into token embeddings synchronously.
        
        Handles various edge cases including:
        - Empty chunk list
        - Individual empty chunks
        - Documents exceeding token limits
        
        For empty chunks, returns empty tensors with the correct embedding dimension.
        Applies attention masking to only include relevant tokens in the output.
        
        Args:
            chunks: List of text chunks to encode
            
        Returns:
            List of token embedding tensors, one per chunk
            
        Warnings:
            - When chunks are empty
            - When documents exceed token limits and are truncated
        """
        if not chunks:
            return []
            
        valid_chunks = []
        valid_indices = []
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                valid_chunks.append(chunk)
                valid_indices.append(i)
            else:
                warnings.warn(f"Chunk at index {i} was empty and will be skipped during encoding.")
        
        if not valid_chunks:
            warnings.warn("All chunks were empty. Returning empty embeddings.")
            return [torch.zeros((0, self.dim), device=self._device) for _ in range(len(chunks))]
        
        input_ids, attention_mask = self.checkpoint.doc_tokenizer.tensorize(valid_chunks)
        
        if input_ids.shape[1] > self._max_doc_tokens:
            warnings.warn(f"Document tokens exceed {self._max_doc_tokens}. Truncating.")
            input_ids = input_ids[:, :self._max_doc_tokens]
            attention_mask = attention_mask[:, :self._max_doc_tokens]
        
        D, mask = self.checkpoint.doc(input_ids, attention_mask, keep_dims='return_mask')
        
        valid_embeddings = []
        for i in range(len(valid_chunks)):
            Di = D[i]
            maski = mask[i].squeeze(-1).bool()
            Di = Di[maski]
            valid_embeddings.append(Di)
        
        result_embeddings = []
        valid_idx = 0
        
        for i in range(len(chunks)):
            if i in valid_indices:
                result_embeddings.append(valid_embeddings[valid_idx])
                valid_idx += 1
            else:
                result_embeddings.append(torch.zeros((0, self.dim), device=self._device))
        
        return result_embeddings
    
    def to_device(self, T: torch.Tensor) -> torch.Tensor:
        """
        Move tensor to the device used by this model.
        
        Args:
            T: Tensor to move to the model's device
            
        Returns:
            Tensor on the model's device
        """
        return T.to(self._get_actual_device(self.checkpoint))
    
    @property
    def dim(self) -> int:
        """
        Get the embedding dimension of the model.
        
        Returns:
            Embedding dimension as an integer
        """
        return self._embedding_dim
    
    @property
    def model_name(self) -> str:
        """
        Get the name of the model.
        
        Returns:
            Model name as a string
        """
        return self._model_name
    
    def __str__(self):
        """
        Get a string representation of the model.
        
        Returns:
            String describing the model configuration
        """
        return (
            f"ColBERTModel(model={self.model_name}, "
            f"dim={self.dim}, "
            f"tokens_per_query={self._tokens_per_query}, "
            f"max_doc_tokens={self._max_doc_tokens}, "
            f"device={self._device})"
        )