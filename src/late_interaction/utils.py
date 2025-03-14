import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from colbert.modeling.checkpoint import pool_embeddings_hierarchical
from sklearn.cluster import AgglomerativeClustering


def expand_parameter(x: int, a: float, b: float, c: float) -> int:
    """
    Increases x by a factor that decays as x increases.
    
    Used to adaptively scale search parameters based on the requested number of results.
    
    Args:
        x: Base value to expand
        a, b, c: Coefficients controlling the expansion rate
        
    Returns:
        Expanded parameter value
    """
    if x < 1:
        return 0
    return max(x, int(a + b*x + c*x*math.log(x)))


def pool_query_embeddings(
    query_embeddings: torch.Tensor, 
    max_distance: float
) -> torch.Tensor:
    """
    Pool query embeddings using agglomerative clustering.
    
    Groups similar token embeddings together to reduce total count.
    
    Args:
        query_embeddings: Query token embeddings tensor
        max_distance: Maximum cosine distance for clustering tokens
        
    Returns:
        Pooled query embeddings tensor with fewer tokens
    """
    if max_distance <= 0:
        return query_embeddings
        
    # Convert embeddings to numpy for clustering
    embeddings_np = query_embeddings.cpu().numpy()
    
    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(
        metric='cosine',
        linkage='average',
        distance_threshold=max_distance,
        n_clusters=None
    )
    labels = clustering.fit_predict(embeddings_np)

    # Pool embeddings based on cluster assignments
    pooled_embeddings = []
    for label in set(labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_embeddings = query_embeddings[cluster_indices]
        
        if len(cluster_embeddings) > 1:
            # Average the embeddings in the cluster
            pooled_embedding = cluster_embeddings.mean(dim=0)
            # Re-normalize the pooled embedding
            pooled_embedding = pooled_embedding / torch.norm(pooled_embedding, p=2)
            pooled_embeddings.append(pooled_embedding)
        else:
            # Only one embedding in the cluster, no need for extra computation
            pooled_embeddings.append(cluster_embeddings[0])

    return torch.stack(pooled_embeddings)


def pool_doc_embeddings(
    doc_embeddings: Union[torch.Tensor, List[torch.Tensor]], 
    pool_factor: int
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Pool document embeddings using hierarchical pooling.
    
    Reduces the number of token embeddings per document by the specified factor.
    
    Args:
        doc_embeddings: Document token embeddings tensor or list of tensors
        pool_factor: Target reduction factor for number of embeddings
        
    Returns:
        Pooled document embeddings with reduced token count
    """
    if pool_factor <= 1:
        return doc_embeddings
        
    if isinstance(doc_embeddings, list):
        # Apply pooling to each document separately
        pooled_embeddings = []
        for Di in doc_embeddings:
            # Convert to float32 before pooling
            Di_float = Di.float()
            Di_pooled, _ = pool_embeddings_hierarchical(
                Di_float,
                [Di_float.shape[0]],  # Single document length
                pool_factor=pool_factor,
                protected_tokens=0
            )
            pooled_embeddings.append(Di_pooled)
        return pooled_embeddings
    else:
        # Pool a single document embedding tensor
        doc_float = doc_embeddings.float()
        pooled, _ = pool_embeddings_hierarchical(
            doc_float,
            [doc_float.shape[0]],
            pool_factor=pool_factor,
            protected_tokens=0
        )
        return pooled