from flask import Flask, request, jsonify, render_template
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
from pathlib import Path
from system_prototype import get_image_description, get_text_embeddings, load_images_paths
from tqdm import tqdm
import argparse
import os

app = Flask(__name__)
VECTOR_SIZE = 1536  # embedding size of Alibaba-NLP/gte-Qwen2-1.5B-instruct
COLLECTION_NAME = "image_search"

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", port=6333)

def init_collection():
    """Initialize or get existing collection"""
    try:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    except Exception:
        # Collection already exists
        pass

def get_image_hash(image_path):
    """Generate a unique hash for an image file"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def create_app(image_dir):
    global COLLECTION_NAME
    # Use the directory name as collection name (sanitized)
    COLLECTION_NAME = os.path.basename(os.path.normpath(image_dir))
    
    # Create symbolic link to image directory in static folder
    static_images_dir = os.path.join('static', 'images')
    os.makedirs(static_images_dir, exist_ok=True)
    symlink_path = os.path.join(static_images_dir, COLLECTION_NAME)
    
    # Remove existing symlink if it exists
    if os.path.exists(symlink_path):
        os.remove(symlink_path)
    
    # Create new symlink
    os.symlink(os.path.abspath(image_dir), symlink_path)
    
    # Initialize collection and index images here instead of in main
    init_collection()
    index_images(image_dir)
    
    return app

def index_images(image_dir):
    """Index all images in the directory if not already indexed"""
    image_paths = load_images_paths(image_dir)
    
    # Get existing image hashes from the collection
    existing_points = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=len(image_paths),
    )[0]
    existing_hashes = {point.payload['image_hash'] for point in existing_points}
    
    # Process only new images
    new_images = []
    for image_path in image_paths:
        image_hash = get_image_hash(image_path)
        if image_hash not in existing_hashes:
            new_images.append((image_path, image_hash))

    # print the number of new images
    # print total number of images
    print(f"Total number of images: {len(image_paths)}")
    print(f"Number of new images: {len(new_images)}")
    
    if new_images:
        print(f"Processing {len(new_images)} new images...")
        next_id = len(existing_hashes) + 1  # Start from the next available ID
        for i, (image_path, image_hash) in enumerate(tqdm(new_images)):
            try:
                description = get_image_description(image_path)
                embedding = get_text_embeddings([description])[0]
                
                # Store in Qdrant
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(
                        id=next_id + i,  # Increment ID for each new image
                        vector=embedding.tolist(),
                        payload={
                            'image_path': os.path.join(COLLECTION_NAME, os.path.basename(image_path)),
                            'description': description,
                            'image_hash': image_hash
                        }
                    )]
                )
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    top_k = int(request.form.get('top_k', 5))
    
    # Get query embedding
    query_embedding = get_text_embeddings([query])[0]
    
    # If top_k is -1 (unlimited), use a large number like 1000
    if top_k == -1:
        top_k = 1000  # or any other large number that makes sense for your use case
    
    # Search in Qdrant
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    
    return jsonify([{
        'image_path': result.payload['image_path'],
        'description': result.payload['description'],
        'similarity': result.score
    } for result in results])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the image search server')
    parser.add_argument('image_dir', type=str, help='Directory containing images to index')
    args = parser.parse_args()
    
    app = create_app(args.image_dir)
    app.run(debug=True) 