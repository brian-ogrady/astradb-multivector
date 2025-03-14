from abc import ABC, abstractmethod
from typing import List, Union

import torch
import numpy as np
from PIL.Image import Image


class LateInteractionModel(ABC):
    """Abstract base class for late interaction retrieval models like ColBERT and ColPali.
    
    Late interaction models encode queries and documents into token-level embeddings,
    then perform similarity calculations between each token pair during retrieval.
    """
    
    @abstractmethod
    async def encode_query(self, q: str) -> torch.Tensor:
        """
        Encode a query string into a tensor of token embeddings.

        Args:
            q: The query string to encode.

        Returns:
            A 2D tensor of query embeddings with shape (num_tokens, embedding_dim),
            where num_tokens is variable (one embedding per token).
        """
        pass

    @abstractmethod
    async def encode_doc(self, chunks: List[Union[str, Image]]) -> List[torch.Tensor]:
        """
        Encode a batch of document chunks into tensors of token embeddings.

        Args:
            chunks: A list of content strings or images to encode.

        Returns:
            A list of 2D tensors of embeddings, one for each input chunk.
            Each tensor has shape (num_tokens, embedding_dim).
        """
        pass
    
    def score(self, Q: torch.Tensor, D: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculate MaxSim scores between query and document embeddings.
        
        For each query token, finds the maximum similarity with any document token,
        then sums these maximum similarities to get the final score.

        Args:
            Q: 2D query embeddings tensor with shape (num_query_tokens, embedding_dim)
            D: List of 2D document embeddings tensors, each with shape (num_doc_tokens, embedding_dim)

        Returns:
            A tensor of similarity scores for each document.
        """
        Q = self.to_device(Q.unsqueeze(0))  # Add batch dimension
        D = [self.to_device(d.to(Q.dtype)) for d in D]  # Match dtype
        
        # Pad document embeddings to same length for batch processing
        D_padded = torch.nn.utils.rnn.pad_sequence(D, batch_first=True, padding_value=0)
        
        # Compute MaxSim scores using einsum
        # - bcns: batch, chunk, query_token, doc_token
        # - Take max similarity for each query token (dim=3)
        # - Sum across all query tokens (dim=2)
        scores = torch.einsum("bnd,csd->bcns", Q, D_padded).max(dim=3)[0].sum(dim=2)
        
        return scores.squeeze(0).to(torch.float32)  # Remove batch dimension and convert to float32
    
    def _embeddings_to_numpy(self, embeddings: torch.Tensor) -> np.ndarray:
        """Convert torch embeddings to numpy arrays for database storage"""
        return embeddings.cpu().detach().numpy()
    
    def _numpy_to_embeddings(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy arrays from database back to torch tensors"""
        return torch.from_numpy(array).to(torch.float32)
    
    @abstractmethod
    def to_device(self, T: torch.Tensor) -> torch.Tensor:
        """
        Copy a tensor to the device used by this model.
        
        Used when loading tensors from the database or during scoring.
        """
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the embedding dimension for this model"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of this model"""
        pass
    
    @property
    def supports_images(self) -> bool:
        """Whether this model supports image inputs"""
        return False
