#!/usr/bin/env python3
"""
ColPali Multimodal Late Interaction Example

This example demonstrates how to use the ColPali model for multimodal late interaction:
1. Sets up a ColPali model for multimodal (text+image) processing
2. Indexes a collection of images with metadata
3. Performs text-to-image search with late interaction
4. Shows how to customize search parameters for optimal results
5. Visualizes search results with matched images

Requirements:
- An AstraDB instance with vector search capabilities
- ColPali library installed (pip install colpali-engine)
- PIL for image processing
- Optional: requests library for downloading sample images (pip install requests)
"""

import asyncio
import json
import os
import uuid
import random
import logging
from io import BytesIO
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from astrapy import DataAPIClient

from astra_multivector.late_interaction import LateInteractionPipeline, ColPaliModel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Create a directory for sample images
def create_image_dir():
    img_dir = Path("./images")
    img_dir.mkdir(exist_ok=True)
    return img_dir


# Sample image data
SAMPLE_IMAGES = [
    {
        "title": "Mountain landscape", 
        "tags": ["nature", "mountains", "outdoors"], 
        "category": "landscape",
        "description": "Mountain landscape with snow-capped peaks"
    },
    {
        "title": "City skyline", 
        "tags": ["urban", "buildings", "architecture"], 
        "category": "cityscape",
        "description": "Urban cityscape with modern skyscrapers"
    },
    {
        "title": "Beach sunset", 
        "tags": ["nature", "ocean", "sunset"], 
        "category": "seascape",
        "description": "Sunset view over calm ocean waters"
    },
    {
        "title": "Modern kitchen", 
        "tags": ["interior", "design", "kitchen"], 
        "category": "interior",
        "description": "Contemporary kitchen with minimalist design"
    },
    {
        "title": "Dog portrait", 
        "tags": ["animal", "pet", "dog"], 
        "category": "pet",
        "description": "Portrait of a happy golden retriever"
    }
]


# Public domain image URLs (from Unsplash)
SAMPLE_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600",  # Mountain
    "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=600",  # City
    "https://images.unsplash.com/photo-1493558103817-58b2924bce98?w=600",  # Beach sunset
    "https://images.unsplash.com/photo-1556911220-bff31c812dba?w=600",     # Kitchen
    "https://images.unsplash.com/photo-1552053831-71594a27632d?w=600"      # Dog
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


def create_solid_color_images(img_dir: Path) -> List[Tuple[str, Image.Image]]:
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
        
        # Add a text label
        draw = ImageDraw.Draw(img)
        try:
            # Try to use a system font, fall back to default if not available
            font = ImageFont.truetype("Arial", 18)
        except IOError:
            font = ImageFont.load_default()
            
        title = SAMPLE_IMAGES[i]["title"]
        draw.text((10, 10), title, fill=(255, 255, 255), font=font)
        
        img.save(img_path)
        image_files.append((str(img_path), img))
    
    return image_files


def download_sample_images(img_dir: Path) -> List[Tuple[str, Optional[Image.Image]]]:
    """
    Download sample images from public URLs.
    
    Returns a list of (file_path, image) tuples. If an image fails to download,
    the image will be None and a fallback will be created.
    """
    if not REQUESTS_AVAILABLE:
        logger.warning("requests library not available. Cannot download sample images.")
        return []
    
    image_files = []
    
    for i, url in enumerate(SAMPLE_IMAGE_URLS):
        img_path = img_dir / f"sample_image_{i+1}.jpg"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            
            # Resize image to reasonable dimensions
            img.thumbnail((512, 512))
            
            # Save image
            img.save(img_path)
            image_files.append((str(img_path), img))
            logger.info(f"Downloaded image {i+1} from {url}")
            
        except (requests.RequestException, UnidentifiedImageError, OSError) as e:
            logger.error(f"Failed to download or process image {i+1} from {url}: {e}")
            image_files.append((str(img_path), None))
    
    return image_files


def validate_images(
    image_files: List[Tuple[str, Optional[Image.Image]]],
    img_dir: Path
) -> List[Tuple[str, Image.Image]]:
    """
    Validate downloaded images and create fallbacks for any that failed.
    """
    validated_files = []
    
    for i, (path, img) in enumerate(image_files):
        if img is None or img.width == 0 or img.height == 0:
            # Create a fallback image with solid color
            logger.info(f"Creating fallback image for file {i+1}")
            
            # Generate a random color as fallback
            color = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200)
            )
            
            fallback = Image.new('RGB', (256, 256), color)
            
            # Add a text label
            draw = ImageDraw.Draw(fallback)
            try:
                font = ImageFont.truetype("Arial", 18)
            except IOError:
                font = ImageFont.load_default()
                
            title = SAMPLE_IMAGES[i]["title"]
            draw.text((10, 10), title, fill=(255, 255, 255), font=font)
            
            # Add a note about fallback
            draw.text(
                (10, 30), 
                "Fallback image", 
                fill=(255, 255, 255), 
                font=font
            )
            
            fallback_path = img_dir / f"fallback_image_{i+1}.jpg"
            fallback.save(fallback_path)
            
            validated_files.append((str(fallback_path), fallback))
        else:
            validated_files.append((path, img))
    
    return validated_files


async def index_sample_images(
    pipeline: LateInteractionPipeline, 
    image_files: List[Tuple[str, Image.Image]]
) -> List[uuid.UUID]:
    """Index sample images into the ColPali pipeline."""
    logger.info("Indexing sample images...")
    doc_ids = []
    
    # Index first image individually
    first_img_path, first_img = image_files[0]
    first_doc_id = uuid.uuid4()
    
    # Create document row with image content and metadata
    document_row = {
        "content": first_img,
        "doc_id": first_doc_id,
        "image_url": first_img_path,  # Store image path for visualization
        **SAMPLE_IMAGES[0]  # Include title, category, tags, etc.
    }
    
    try:
        doc_id = await pipeline.index_document(document_row)
        if doc_id:
            doc_ids.append(doc_id)
            logger.info(f"Indexed image 1/{len(image_files)} with ID: {doc_id}")
    except Exception as e:
        logger.error(f"Failed to index first image: {e}")
    
    # Index remaining images in bulk
    remaining_images = image_files[1:]
    document_rows = []
    
    # Create document rows with images and metadata
    for i, (img_path, img) in enumerate(remaining_images):
        document_rows.append({
            "content": img,
            "doc_id": uuid.uuid4(),
            "image_url": img_path,  # Store image path for visualization
            **SAMPLE_IMAGES[i+1]  # Include title, category, tags, etc. for this image
        })
    
    try:
        bulk_ids = await pipeline.bulk_index_documents(
            document_rows=document_rows,
            concurrency=3,
            batch_size=2
        )
            
        doc_ids.extend(bulk_ids)
        logger.info(f"Bulk indexed {len(bulk_ids)} additional images")
    except Exception as e:
        logger.error(f"Failed to bulk index images: {e}")
    
    return doc_ids


async def fetch_document_metadata(
    pipeline: LateInteractionPipeline, 
    doc_id: uuid.UUID
) -> Dict[str, Any]:
    """Fetch document metadata for visualization."""
    try:
        cursor = await pipeline._doc_table.find(
            filter={"doc_id": str(doc_id)},
        )
        docs = await cursor.to_list()
        
        if docs:
            return docs[0]
        return {}
    except Exception as e:
        logger.error(f"Error fetching document metadata: {e}")
        return {}


async def perform_searches(pipeline: LateInteractionPipeline, image_dir: Path):
    """Perform different searches to demonstrate ColPali capabilities."""
    logger.info("Performing searches...")
    
    # Create a directory for visualization results
    results_dir = Path("./search_results")
    results_dir.mkdir(exist_ok=True)
    
    # Basic text-to-image search
    query = "mountains"
    results = await pipeline.search(query=query, k=3)
    
    print(f"\nSearch results for '{query}':")
    await visualize_results(pipeline, results, query, results_dir / "mountains_results.jpg")
    
    # Search with custom parameters
    query = "urban setting"
    results = await pipeline.search(
        query=query, 
        k=2
    )
    
    print(f"\nSearch results for '{query}':")
    await visualize_results(pipeline, results, query, results_dir / "urban_results.jpg")
    
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
    await visualize_results(pipeline, results, query, results_dir / "beach_results.jpg")


async def visualize_results(
    pipeline: LateInteractionPipeline,
    results: List[Tuple[uuid.UUID, float, str]],
    query: str,
    output_path: Path
):
    """
    Visualize search results by creating a composite image of results.
    
    Args:
        pipeline: The LateInteractionPipeline instance
        results: The search results from pipeline.search()
        query: The search query used
        output_path: Where to save the visualization
    """
    if not results:
        print("No results to visualize")
        return
    
    # First print text results
    for i, (doc_id, score, content) in enumerate(results):
        print(f"{i+1}. Score: {score:.4f}")
        
        # Fetch document metadata for better display
        metadata = await fetch_document_metadata(pipeline, doc_id)
        
        # Display available metadata
        if metadata:
            print(f"   Title: {metadata.get('title', 'N/A')}")
            print(f"   Category: {metadata.get('category', 'N/A')}")
            print(f"   Description: {metadata.get('description', 'N/A')}")
            print(f"   Tags: {', '.join(metadata.get('tags', []))}")
        else:
            print(f"   Content: {content}")
        
        print(f"   Document ID: {doc_id}")
    
    # Then create a visual representation of results
    try:
        # Calculate grid size based on number of results
        num_results = len(results)
        
        # Create a new blank image for the visualization
        result_width = 256
        result_height = 256
        
        # Header height for the query info
        header_height = 50
        
        # Create a grid of images with header
        grid_img = Image.new('RGB', 
                          (result_width * min(num_results, 3), 
                           result_height + header_height),
                          color=(240, 240, 240))
        
        # Add query info to header
        draw = ImageDraw.Draw(grid_img)
        try:
            font = ImageFont.truetype("Arial", 16)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((10, 10), f"Query: '{query}'", fill=(0, 0, 0), font=font)
        draw.text((10, 30), f"Results: {num_results}", fill=(0, 0, 0), font=font)
        
        # Add result images to grid
        for i, (doc_id, score, _) in enumerate(results):
            if i >= 3:  # Limit to 3 results for visualization
                break
                
            # Fetch document metadata
            metadata = await fetch_document_metadata(pipeline, doc_id)
            
            if not metadata or "image_url" not in metadata:
                # Draw placeholder if no image
                placeholder = Image.new('RGB', (result_width, result_height), (200, 200, 200))
                draw_placeholder = ImageDraw.Draw(placeholder)
                draw_placeholder.text((10, 10), f"No image", fill=(0, 0, 0), font=font)
                grid_img.paste(placeholder, (i * result_width, header_height))
                continue
                
            try:
                # Load image from stored path
                img_path = metadata["image_url"]
                img = Image.open(img_path)
                
                # Resize to fit grid
                img.thumbnail((result_width, result_height))
                
                # Add score and title overlay
                draw_img = ImageDraw.Draw(img)
                draw_img.rectangle([(0, 0), (result_width, 30)], fill=(0, 0, 0, 128))
                draw_img.text(
                    (5, 5), 
                    f"Score: {score:.2f}", 
                    fill=(255, 255, 255), 
                    font=font
                )
                
                # Paste into grid
                grid_img.paste(img, (i * result_width, header_height))
                
            except (IOError, UnidentifiedImageError) as e:
                logger.error(f"Error visualizing image result: {e}")
                # Draw error placeholder
                placeholder = Image.new('RGB', (result_width, result_height), (200, 200, 200))
                draw_placeholder = ImageDraw.Draw(placeholder)
                draw_placeholder.text((10, 10), f"Error loading image", fill=(0, 0, 0), font=font)
                grid_img.paste(placeholder, (i * result_width, header_height))
        
        # Save the visualization
        grid_img.save(output_path)
        print(f"Results visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating results visualization: {e}")
        print(f"Could not create visualization: {e}")


async def main():
    # Create image directory
    img_dir = create_image_dir()
    
    # Ask user if they want to download real images
    use_real_images = False
    if REQUESTS_AVAILABLE:
        user_input = input("Do you want to download sample images from the internet? (y/n): ").lower()
        use_real_images = user_input in ['y', 'yes']
    
    # Get images based on user choice
    if use_real_images:
        logger.info("Downloading sample images...")
        image_files = download_sample_images(img_dir)
        
        # Validate images and create fallbacks if needed
        image_files = validate_images(image_files, img_dir)
    else:
        logger.info("Creating solid color sample images...")
        image_files = create_solid_color_images(img_dir)
    
    # Set up the pipeline
    try:
        pipeline = await setup_pipeline()
        
        # Index images
        doc_ids = await index_sample_images(pipeline, image_files)
        
        if not doc_ids:
            logger.error("No documents were successfully indexed. Exiting.")
            return
            
        # Perform searches
        await perform_searches(pipeline, img_dir)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())