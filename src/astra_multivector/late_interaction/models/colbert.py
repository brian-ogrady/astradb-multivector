import asyncio
from typing import List, Dict, Any, Optional, Union

import torch
from colbert import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.infra import ColBERTConfig
from PIL.Image import Image

from astra_multivector.late_interaction import LateInteractionModel


def _get_module_device(module):
    """Helper function to get the device of a PyTorch module"""
    return next(module.parameters()).device


class ColBERTModel(LateInteractionModel):
    """
    ColBERT implementation of the LateInteractionModel interface.
    
    Uses the ColBERT neural IR model for token-level late interaction retrieval.
    """
    
    def __init__(
        self, 
        model_name: str = 'answerdotai/answerai-colbert-small-v1',
        tokens_per_query: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize a ColBERT model.
        
        Args:
            model_name: HuggingFace model name or path to local checkpoint
            tokens_per_query: Maximum number of tokens per query
            device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.)
                   If None, will use CUDA if available, otherwise CPU.
        """
        self._model_name = model_name
        self._tokens_per_query = tokens_per_query
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = device
        
        # Initialize ColBERT components
        self.config = ColBERTConfig(
            checkpoint=model_name,
            query_maxlen=tokens_per_query,
        )
        
        # Load model
        self.checkpoint = Checkpoint(
            self.config.checkpoint, 
            colbert_config=self.config,
            device=device
        )
        
        # Initialize encoder
        self.encoder = CollectionEncoder(self.config, self.checkpoint)
        
        # Get embedding dimension
        self._embedding_dim = len(self.encode_query_sync("test query")[0])
    
    async def encode_query(self, q: str) -> torch.Tensor:
        """
        Encode a query string into token embeddings.
        
        Args:
            q: The query string to encode
            
        Returns:
            Query token embeddings tensor of shape (num_tokens, embedding_dim)
        """
        # Offload to thread pool since ColBERT encoding is not async
        return await asyncio.to_thread(self.encode_query_sync, q)
    
    def encode_query_sync(self, q: str) -> torch.Tensor:
        """Synchronous version of encode_query"""
        return self.checkpoint.queryFromText([q])[0]
    
    async def encode_doc(self, chunks: List[Union[str, Image]]) -> List[torch.Tensor]:
        """
        Encode document chunks into token embeddings.
        
        Args:
            chunks: List of text chunks to encode
            
        Returns:
            List of token embedding tensors, one per chunk
        """
        # Validate input types
        if not all(isinstance(chunk, str) for chunk in chunks):
            raise TypeError("ColBERT only supports text chunks")
        
        # Offload to thread pool since ColBERT encoding is not async
        return await asyncio.to_thread(self.encode_doc_sync, chunks)
    
    def encode_doc_sync(self, chunks: List[str]) -> List[torch.Tensor]:
        """Synchronous version of encode_doc"""
        # Tokenize text chunks
        input_ids, attention_mask = self.checkpoint.doc_tokenizer.tensorize(chunks)
        
        # Encode tokens to embeddings
        D, mask = self.checkpoint.doc(input_ids, attention_mask, keep_dims='return_mask')
        
        # Extract non-padding token embeddings for each chunk
        embeddings_list = []
        for i in range(len(chunks)):
            Di = D[i]
            maski = mask[i].squeeze(-1).bool()
            Di = Di[maski]  # Keep only non-padded embeddings
            embeddings_list.append(Di)
            
        return embeddings_list
    
    def to_device(self, T: torch.Tensor) -> torch.Tensor:
        """Move tensor to the device used by this model"""
        return T.to(_get_module_device(self.checkpoint))
    
    @property
    def dim(self) -> int:
        """Return the embedding dimension"""
        return self._embedding_dim
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return self._model_name
    
    def __str__(self):
        return f"ColBERTModel(model={self.model_name}, dim={self.dim}, device={self._device})"
