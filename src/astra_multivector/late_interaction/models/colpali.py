import asyncio
from typing import List, Dict, Any, Optional, Union

import torch
from PIL.Image import Image
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor

from astra_multivector.late_interaction import LateInteractionModel


def _get_module_device(module):
    """Helper function to get the device of a PyTorch module"""
    return next(module.parameters()).device


class ColPaliModel(LateInteractionModel):
    """
    ColPali implementation of the LateInteractionModel interface.
    
    Supports multimodal late interaction between text queries and image documents.
    """
    
    def __init__(
        self, 
        model_name: str = 'vidore/colqwen2-v0.1',
        device: Optional[str] = None
    ):
        """
        Initialize a ColPali model.
        
        Args:
            model_name: HuggingFace model name or path to local checkpoint
            device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.)
                   If None, will use automatic device mapping.
        """
        self._model_name = model_name
        
        # Determine model type and processor classes
        if 'qwen' in model_name:
            model_cls = ColQwen2
            processor_cls = ColQwen2Processor
        else:
            model_cls = ColPali
            processor_cls = ColPaliProcessor
            
        # Set device mapping
        device_map = "auto" if device is None else device
        
        # Load model and processor
        self.colpali = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        ).eval()
        
        self.processor = processor_cls.from_pretrained(model_name)
    
    async def encode_query(self, q: str) -> torch.Tensor:
        """
        Encode a query string into token embeddings.
        
        Args:
            q: The query string to encode
            
        Returns:
            Query token embeddings tensor
        """
        # Offload to thread pool since ColPali encoding is not async
        return await asyncio.to_thread(self.encode_query_sync, q)
    
    def encode_query_sync(self, q: str) -> torch.Tensor:
        """Synchronous version of encode_query"""
        with torch.no_grad():
            batch = self.processor.process_queries([q])
            batch = {k: self.to_device(v) for k, v in batch.items()}
            embeddings = self.colpali(**batch)
            
        return embeddings[0].float()  # Convert to float32
    
    async def encode_doc(self, images: List[Union[str, Image]]) -> List[torch.Tensor]:
        """
        Encode images into token embeddings.
        
        Args:
            images: List of PIL images to encode
            
        Returns:
            List of token embedding tensors, one per image
        """
        # Validate input types
        if not all(isinstance(img, Image) for img in images):
            raise TypeError("ColPali only supports image inputs")
            
        # Offload to thread pool since ColPali encoding is not async
        return await asyncio.to_thread(self.encode_doc_sync, images)
    
    def encode_doc_sync(self, images: List[Image]) -> List[torch.Tensor]:
        """Synchronous version of encode_doc"""
        with torch.no_grad():
            batch = self.processor.process_images(images)
            batch = {k: self.to_device(v) for k, v in batch.items()}
            raw_embeddings = self.colpali(**batch)
        
        # Discard zero vectors and convert to float32
        return [emb[emb.norm(dim=-1) > 0].float() for emb in raw_embeddings]
    
    def to_device(self, T: torch.Tensor) -> torch.Tensor:
        """Move tensor to the device used by this model"""
        return T.to(_get_module_device(self.colpali))
    
    @property
    def dim(self) -> int:
        """Return the embedding dimension"""
        return self.colpali.dim
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return self._model_name
    
    @property
    def supports_images(self) -> bool:
        """ColPali supports image inputs"""
        return True
    
    def __str__(self):
        return f"ColPaliModel(model={self.model_name}, dim={self.dim}, device={_get_module_device(self.colpali)})"
