import math
import logging
from collections import namedtuple
from typing import List, Union, Optional

import numpy as np
import torch
from colbert.modeling.checkpoint import pool_embeddings_hierarchical
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)

PoolingResult = namedtuple('PoolingResult', ['embeddings', 'stats'])


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
    max_distance: float, 
    min_clusters: int = 3,
    return_cluster_info: bool = False
) -> Union[torch.Tensor, PoolingResult]:
    """
    Pool query embeddings using agglomerative clustering.
    
    Groups similar token embeddings together to reduce total count.
    
    Args:
        query_embeddings: Query token embeddings tensor
        max_distance: Maximum cosine distance for clustering tokens
        min_clusters: Minimum number of clusters to maintain (prevents over-pooling)
        return_cluster_info: Whether to return additional clustering information
        
    Returns:
        - If return_cluster_info=False: Pooled query embeddings tensor with fewer tokens
        - If return_cluster_info=True: PoolingResult with .embeddings and .stats fields
          where stats contains 'original_count', 'pooled_count', 'compression_ratio'
    """
    original_count = query_embeddings.shape[0]
    
    if max_distance <= 0 or original_count <= min_clusters:
        if return_cluster_info:
            cluster_info = {
                'original_count': original_count,
                'pooled_count': original_count,
                'compression_ratio': 1.0,
                'pooling_applied': False,
                'reason': 'max_distance <= 0 or not enough tokens to pool'
            }
            return PoolingResult(query_embeddings, cluster_info)
        return query_embeddings
        
    embeddings_np = query_embeddings.cpu().numpy()
    
    clustering = AgglomerativeClustering(
        metric='cosine',
        linkage='average',
        distance_threshold=max_distance,
        n_clusters=None
    )
    labels = clustering.fit_predict(embeddings_np)
    unique_labels = set(labels)
    
    if len(unique_labels) < min_clusters:
        if return_cluster_info:
            cluster_info = {
                'original_count': original_count,
                'pooled_count': original_count,
                'compression_ratio': 1.0,
                'pooling_applied': False,
                'reason': f'too few clusters ({len(unique_labels)} < {min_clusters})'
            }
            return PoolingResult(query_embeddings, cluster_info)
        return query_embeddings
    
    pooled_embeddings = []
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_embeddings = query_embeddings[cluster_indices]
        
        if len(cluster_embeddings) > 1:
            pooled_embedding = cluster_embeddings.mean(dim=0)
            pooled_embedding = pooled_embedding / torch.norm(pooled_embedding, p=2)
            pooled_embeddings.append(pooled_embedding)
        else:
            pooled_embeddings.append(cluster_embeddings[0])

    result = torch.stack(pooled_embeddings)
    
    if return_cluster_info:
        pooled_count = result.shape[0]
        cluster_info = {
            'original_count': original_count,
            'pooled_count': pooled_count,
            'compression_ratio': original_count / pooled_count,
            'pooling_applied': True,
            'clusters': {label: len(np.where(labels == label)[0]) for label in unique_labels}
        }
        return PoolingResult(result, cluster_info)
        
    return result


def pool_doc_embeddings(
    doc_embeddings: Union[torch.Tensor, List[torch.Tensor]], 
    pool_factor: int,
    min_tokens: int = 4,
    protected_tokens: int = 0,
    return_stats: bool = False
) -> Union[torch.Tensor, List[torch.Tensor], PoolingResult]:
    """
    Pool document embeddings using hierarchical pooling.
    
    Reduces the number of token embeddings per document by the specified factor.
    
    Args:
        doc_embeddings: Document token embeddings tensor or list of tensors
        pool_factor: Target reduction factor for number of embeddings
        min_tokens: Minimum number of tokens to keep after pooling
        protected_tokens: Number of tokens at the start to never pool
        return_stats: Whether to return pooling statistics
        
    Returns:
        - If return_stats=False: Pooled document embeddings with reduced token count
        - If return_stats=True: PoolingResult with .embeddings and .stats fields
          containing pooling statistics
    """
    if pool_factor <= 1:
        if return_stats:
            if isinstance(doc_embeddings, list):
                token_counts = [d.shape[0] for d in doc_embeddings]
                stats = {
                    'original_tokens': token_counts,
                    'pooled_tokens': token_counts,
                    'pooling_applied': False,
                    'reason': 'pool_factor <= 1'
                }
            else:
                stats = {
                    'original_tokens': doc_embeddings.shape[0],
                    'pooled_tokens': doc_embeddings.shape[0],
                    'pooling_applied': False,
                    'reason': 'pool_factor <= 1'
                }
            return PoolingResult(doc_embeddings, stats)
        return doc_embeddings
    
    if isinstance(doc_embeddings, list):
        pooled_embeddings = []
        original_tokens = []
        pooled_tokens = []
        pooling_applied = []
        
        for i, Di in enumerate(doc_embeddings):
            original_count = Di.shape[0]
            original_tokens.append(original_count)
            
            if original_count <= min_tokens:
                pooled_embeddings.append(Di)
                pooled_tokens.append(original_count)
                pooling_applied.append(False)
                logger.warning(f"Document {i}: token count ({original_count}) is already <= min_tokens ({min_tokens}). Skipping pooling.")
                continue
                
            if original_count < protected_tokens + pool_factor:
                pooled_embeddings.append(Di)
                pooled_tokens.append(original_count)
                pooling_applied.append(False)
                logger.warning(f"Document {i}: pool_factor ({pool_factor}) too large for document with {original_count} tokens and {protected_tokens} protected tokens. Skipping pooling.")
                continue
            
            Di_float = Di.float()
            Di_pooled, _ = pool_embeddings_hierarchical(
                Di_float,
                [Di_float.shape[0]],
                pool_factor=pool_factor,
                protected_tokens=protected_tokens
            )
            
            if Di_pooled.shape[0] < min_tokens:
                pooled_embeddings.append(Di)
                pooled_tokens.append(original_count)
                pooling_applied.append(False)
                logger.warning(f"Document {i}: pooling would reduce tokens below min_tokens ({min_tokens}). Skipping pooling.")
                continue
                
            pooled_embeddings.append(Di_pooled)
            pooled_tokens.append(Di_pooled.shape[0])
            pooling_applied.append(True)
            
            if Di_pooled.shape[0] < original_count / 2:
                logger.info(f"Document {i}: Significant compression applied. Tokens reduced from {original_count} to {Di_pooled.shape[0]} (factor: {original_count/Di_pooled.shape[0]:.1f}x)")
                
        if return_stats:
            stats = {
                'original_tokens': original_tokens,
                'pooled_tokens': pooled_tokens,
                'compression_ratios': [o/p if b else 1.0 for o, p, b in zip(original_tokens, pooled_tokens, pooling_applied)],
                'pooling_applied': pooling_applied,
                'total_reduction': sum(original_tokens) / sum(pooled_tokens)
            }
            return PoolingResult(pooled_embeddings, stats)
            
        return pooled_embeddings
    else:
        original_count = doc_embeddings.shape[0]
        
        if original_count <= min_tokens:
            if return_stats:
                stats = {
                    'original_tokens': original_count,
                    'pooled_tokens': original_count,
                    'pooling_applied': False,
                    'reason': f'token count ({original_count}) is already <= min_tokens ({min_tokens})'
                }
                return PoolingResult(doc_embeddings, stats)
            return doc_embeddings
            
        if original_count < protected_tokens + pool_factor:
            if return_stats:
                stats = {
                    'original_tokens': original_count,
                    'pooled_tokens': original_count,
                    'pooling_applied': False,
                    'reason': f'pool_factor ({pool_factor}) too large for document with {original_count} tokens'
                }
                return PoolingResult(doc_embeddings, stats)
            logger.warning(f"pool_factor ({pool_factor}) too large for document with {original_count} tokens. Skipping pooling.")
            return doc_embeddings
        
        doc_float = doc_embeddings.float()
        pooled, _ = pool_embeddings_hierarchical(
            doc_float,
            [doc_float.shape[0]],
            pool_factor=pool_factor,
            protected_tokens=protected_tokens
        )
        
        if pooled.shape[0] < min_tokens:
            logger.warning(f"Pooling would reduce tokens below min_tokens ({min_tokens}). Skipping pooling.")
            if return_stats:
                stats = {
                    'original_tokens': original_count,
                    'pooled_tokens': original_count,
                    'pooling_applied': False,
                    'reason': f'pooling would reduce below min_tokens'
                }
                return PoolingResult(doc_embeddings, stats)
            return doc_embeddings
        
        if pooled.shape[0] < original_count / 2:
            logger.info(f"Significant compression applied. Tokens reduced from {original_count} to {pooled.shape[0]} (factor: {original_count/pooled.shape[0]:.1f}x)")
        
        if return_stats:
            stats = {
                'original_tokens': original_count,
                'pooled_tokens': pooled.shape[0],
                'compression_ratio': original_count / pooled.shape[0],
                'pooling_applied': True
            }
            return PoolingResult(pooled, stats)
            
        return pooled