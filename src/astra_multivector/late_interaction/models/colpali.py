import asyncio
import logging
from typing import List, Optional, Union

import torch
from PIL.Image import Image
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor

from astra_multivector.late_interaction import LateInteractionModel

logger = logging.getLogger(__name__)


class ColPaliModel(LateInteractionModel):
    """
    ColPali implementation of the LateInteractionModel interface.
    
    Supports multimodal late interaction between text queries and image documents.
    ColPali is designed for cross-modal retrieval, allowing efficient similarity
    search between text queries and image documents, which is particularly useful
    for image search and multimodal retrieval applications.
    """
    
    def __init__(
        self, 
        model_name: str = 'vidore/colqwen2-v0.1',
        device: Optional[str] = None
    ):
        """
        Initialize a ColPali model.
        
        Loads a ColPali or ColQwen2 model based on the model name and sets up
        the appropriate processor. Handles device placement with fallback to
        automatic device mapping if the requested device is unavailable.
        
        Args:
            model_name: HuggingFace model name or path to local checkpoint.
                       Default is 'vidore/colqwen2-v0.1'.
            device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.)
                   If None, will use automatic device mapping.
        """
        super().__init__(device=device)

        self._model_name = model_name
        
        if 'qwen' in model_name:
            model_cls = ColQwen2
            processor_cls = ColQwen2Processor
        else:
            model_cls = ColPali
            processor_cls = ColPaliProcessor

        try:
            self.colpali = model_cls.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=self._device,
            ).eval()
        except RuntimeError as e:
            logger.warning(f"Could not load model on {self._device}: {e}")
            self.colpali = model_cls.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()
        
        self.processor = processor_cls.from_pretrained(model_name)
    
    async def encode_query(self, q: str) -> torch.Tensor:
        """
        Encode a query string into token embeddings asynchronously.
        
        Offloads the synchronous encoding work to a separate thread to avoid
        blocking the event loop.
        
        Args:
            q: The query string to encode
            
        Returns:
            Query token embeddings tensor that can be used for similarity search
            against document embeddings
        """
        return await asyncio.to_thread(self.encode_query_sync, q)
    
    def encode_query_sync(self, q: str) -> torch.Tensor:
        """
        Encode a query string into token embeddings synchronously.
        
        Processes the query text through the ColPali processor and model to generate
        embeddings suitable for similarity search against image embeddings.
        Handles empty queries by returning an empty tensor with the correct dimension.
        
        Args:
            q: The query string to encode
            
        Returns:
            Query token embeddings tensor with shape (sequence_length, embedding_dim)
        """
        if not q.strip():
            return torch.zeros((0, self.dim), device=self._get_actual_device(self.colpali))

        with torch.no_grad():
            batch = self.processor.process_queries([q])
            batch = {k: self.to_device(v) for k, v in batch.items()}
            embeddings = self.colpali(**batch)
            
        return embeddings[0].float()
    
    async def encode_doc(self, images: List[Union[str, Image]]) -> List[torch.Tensor]:
        """
        Encode images into token embeddings asynchronously.
        
        Offloads the synchronous encoding work to a separate thread to avoid
        blocking the event loop. Validates that all inputs are images since
        ColPali only supports image inputs.
        
        Args:
            images: List of PIL images to encode
            
        Returns:
            List of token embedding tensors, one per image
            
        Raises:
            TypeError: If any input is not a PIL.Image.Image
        """
        if not images:
            return []
            
        if not all(isinstance(img, Image) for img in images):
            raise TypeError("ColPali only supports image inputs")
            
        return await asyncio.to_thread(self.encode_doc_sync, images)
    
    def encode_doc_sync(self, images: List[Image]) -> List[torch.Tensor]:
        """
        Encode images into token embeddings synchronously.
        
        Processes images through the ColPali processor and model to generate
        embeddings suitable for similarity search against query embeddings.
        Handles various edge cases including:
        - Empty image list
        - Invalid images (zero dimensions)
        - Non-image inputs
        
        Args:
            images: List of PIL images to encode
            
        Returns:
            List of token embedding tensors, one per image
            
        Raises:
            TypeError: If any input is not a PIL.Image.Image
        """
        if not images:
            return []
            
        valid_images = []
        valid_indices = []
        
        for i, img in enumerate(images):
            if not isinstance(img, Image):
                raise TypeError(f"ColPali only supports image chunks, got {type(img).__name__}")
            if img.width > 0 and img.height > 0:
                valid_images.append(img)
                valid_indices.append(i)
            else:
                logger.warning(f"Image at index {i} is invalid (zero dimensions) and will be skipped")
        
        if not valid_images:
            logger.warning("All images are invalid. Returning empty embeddings.")
            return [torch.zeros((0, self.dim), device=self._get_actual_device(self.colpali)) 
                    for _ in range(len(images))]

        with torch.no_grad():
            batch = self.processor.process_images(valid_images)
            batch = {k: self.to_device(v) for k, v in batch.items()}
            raw_embeddings = self.colpali(**batch)
        
        valid_embeddings = [emb[emb.norm(dim=-1) > 0].float() for emb in raw_embeddings]
        
        result_embeddings = []
        valid_idx = 0
        
        for i in range(len(images)):
            if i in valid_indices:
                result_embeddings.append(valid_embeddings[valid_idx])
                valid_idx += 1
            else:
                result_embeddings.append(torch.zeros((0, self.dim), 
                                                    device=self._get_actual_device(self.colpali)))
        
        return result_embeddings
    
    def to_device(self, T: Union[torch.Tensor, dict, None]) -> Union[torch.Tensor, dict, None]:
        """
        Move tensor or dictionary of tensors to the device used by this model.
        
        Recursively handles complex data structures containing tensors, such as
        dictionaries with nested dictionaries. Returns None for None input and
        raises TypeError for unsupported input types.
        
        Args:
            T: Tensor, dictionary of tensors, or None to move to device
            
        Returns:
            Input moved to the correct device, with the same structure
            
        Raises:
            TypeError: If T is not a tensor, dictionary, or None
        """
        if T is None:
            return None
            
        if isinstance(T, torch.Tensor):
            return T.to(self._get_actual_device(self.colpali))
            
        if isinstance(T, dict):
            return {k: self.to_device(v) for k, v in T.items()}
            
        raise TypeError(f"Expected torch.Tensor, dict, or None, got {type(T)}")
    
    @property
    def dim(self) -> int:
        """
        Get the embedding dimension of the model.
        
        Returns:
            Embedding dimension as an integer
        """
        return self.colpali.dim
    
    @property
    def model_name(self) -> str:
        """
        Get the name of the model.
        
        Returns:
            Model name as a string
        """
        return self._model_name
    
    @property
    def supports_images(self) -> bool:
        """
        Check if the model supports image inputs.
        
        ColPali is specifically designed for image inputs, so this
        always returns True.
        
        Returns:
            Always True for ColPali models
        """
        return True
    
    def __str__(self):
        """
        Get a string representation of the model.
        
        Returns:
            String describing the model configuration
        """
        return (
            f"ColPaliModel(model={self.model_name}, "
            f"dim={self.dim}, "
            f"device={self._device}, "
            f"supports_images={self.supports_images})"
        )
