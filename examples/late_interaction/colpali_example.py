#!/usr/bin/env python3
"""
ColPali Multimodal Late Interaction Example

This example demonstrates how to use the ColPali model for multimodal late interaction:
1. Sets up a ColPali model for multimodal (text+image) processing
2. Indexes a collection of images with metadata
3. Performs text-to-image search with late interaction
4. Shows how to customize search parameters for optimal results

Requirements:
- An AstraDB instance with vector search capabilities
- ColPali library installed (pip install colpali-engine)
- PIL for image processing
- Sample images in the 'images' directory
"""

import asyncio
import json
import os
import uuid
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image

from astrapy import DataAPIClient

from astra_multivector.late_interaction import LateInteractionPipeline, ColPaliModel


# Create a directory for sample images
def create_image_dir():
    img_dir = Path("./images")
    img_dir.mkdir(exist_ok=True)
    return img_dir


# Sample image metadata
SAMPLE_IMAGE_METADATA = [
    {"title": "Mountain landscape", "tags": ["nature", "mountains", "outdoors"], "category": "landscape"},
    {"title": "City skyline", "tags": ["urban", "buildings", "architecture"], "category": "cityscape"},
    {"title": "Beach sunset", "tags": ["nature", "ocean", "sunset"], "category": "seascape"},
    {"title": "Modern kitchen", "tags": ["interior", "design", "kitchen"], "category": "interior"},
    {"title": "Dog portrait", "tags": ["animal", "pet", "dog"], "category": "pet"}
]


async def setup_pipeline() -> LateInteractionPipeline:
    """Set up the ColPali pipeline with AstraDB."""
    # Load environment variables
    load_dotenv()
    
    # Required environment variables for AstraDB
    astra_token = os.getenv("ASTRA_TOKEN")
    astra_api_endpoint = os.getenv("ASTRA_API_ENDPOINT")
    
    if not astra_token or not astra_api_endpoint:
        raise ValueError(
            "Please set ASTRA_TOKEN and ASTRA_API_ENDPOINT environment variables"
        )
    
    # Create async database client
    db_client = DataAPIClient(token=astra_token)
    db = db_client.get_async_database(api_endpoint=astra_api_endpoint)
    
    # Create a ColPali model
    # Options include 'vidore/colpali-v0.1' or 'vidore/colqwen2-v0.1'
    model = ColPaliModel(
        model_name="vidore/colqwen2-v0.1",
        # Use specific device or 'auto' for automatic mapping
        device="cpu"  
    )
    
    # Create pipeline with customized parameters
    pipeline = LateInteractionPipeline(
        db=db,
        model=model,
        base_table_name="colpali_example",
        # Image pooling factor (reduces token count)
        doc_pool_factor=4,
        # Query token pooling threshold
        query_pool_distance=0.05,
        # Default concurrency for async operations
        default_concurrency_limit=5,
    )
    
    # Initialize the tables
    await pipeline.initialize()
    
    return pipeline


def create_sample_images(img_dir: Path) -> List[Tuple[str, Image.Image]]:
    """
    Create simple solid-colored images for demonstration.
    
    In a real application, you would use actual photos or images.
    """
    image_files = []
    colors = [
        (67, 142, 219),  # Blue for mountains
        (189, 189, 189),  # Gray for city
        (255, 183, 77),   # Orange for sunset
        (244, 244, 244),  # White for kitchen
        (161, 107, 72)    # Brown for dog
    ]
    
    for i, color in enumerate(colors):
        img_path = img_dir / f"sample_image_{i+1}.jpg"
        img = Image.new('RGB', (256, 256), color)
        img.save(img_path)
        image_files.append((str(img_path), img))
    
    return image_files


async def index_sample_images(
    pipeline: LateInteractionPipeline, 
    image_files: List[Tuple[str, Image.Image]]
) -> List[uuid.UUID]:
    """Index sample images into the ColPali pipeline."""
    print("Indexing sample images...")
    doc_ids = []
    
    # Index first image individually
    first_img_path, first_img = image_files[0]
    doc_id = await pipeline.index_document(
        content=first_img
    )
    doc_ids.append(doc_id)
    print(f"Indexed image 1/5 with ID: {doc_id}")
    
    # Index remaining images in bulk
    remaining_images = [img for _, img in image_files[1:]]
    
    bulk_ids = await pipeline.bulk_index_documents(
        documents=remaining_images,
        max_concurrency=3,
        batch_size=2
    )
        
    doc_ids.extend(bulk_ids)
    print(f"Bulk indexed {len(bulk_ids)} additional images")
    
    return doc_ids


async def perform_searches(pipeline: LateInteractionPipeline):
    """Perform different searches to demonstrate ColPali capabilities."""
    print("\nPerforming searches...")
    
    # Basic text-to-image search
    query = "mountains"
    results = await pipeline.search(query=query, k=3)
    
    print(f"\nSearch results for '{query}':")
    for i, (doc_id, score, content) in enumerate(results):
        print(f"{i+1}. Score: {score:.4f}")
        print(f"   Content: {content}")
        print(f"   Image Title: {SAMPLE_IMAGE_METADATA[i]['title'] if i < len(SAMPLE_IMAGE_METADATA) else 'Unknown'}")
    
    # Search with custom parameters
    query = "urban setting"
    results = await pipeline.search(
        query=query, 
        k=2
    )
    
    print(f"\nSearch results for '{query}':")
    for i, (doc_id, score, content) in enumerate(results):
        print(f"{i+1}. Score: {score:.4f}")
        print(f"   Content: {content}")
        print(f"   Image Title: {SAMPLE_IMAGE_METADATA[i]['title'] if i < len(SAMPLE_IMAGE_METADATA) else 'Unknown'}")
    
    # Detailed search with customized parameters
    query = "beach with sunset"
    results = await pipeline.search(
        query=query,
        k=3,
        # Customize search parameters for multimodal search
        n_ann_tokens=300,
        n_maxsim_candidates=10
    )
    
    print(f"\nDetailed search results for '{query}':")
    for i, (doc_id, score, content) in enumerate(results):
        print(f"{i+1}. Score: {score:.4f}")
        print(f"   Content: {content}")
        print(f"   Image Title: {SAMPLE_IMAGE_METADATA[i]['title'] if i < len(SAMPLE_IMAGE_METADATA) else 'Unknown'}")
        print(f"   Category: {SAMPLE_IMAGE_METADATA[i]['category'] if i < len(SAMPLE_IMAGE_METADATA) else 'Unknown'}")


async def main():
    # Create image directory
    img_dir = create_image_dir()
    
    # Create sample images (in a real app, you would use actual images)
    image_files = create_sample_images(img_dir)
    
    # Set up the pipeline
    pipeline = await setup_pipeline()
    
    try:
        # Index images
        doc_ids = await index_sample_images(pipeline, image_files)
        
        # Perform searches
        await perform_searches(pipeline)
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())