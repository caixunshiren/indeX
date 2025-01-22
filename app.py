from flask import Flask, request, jsonify, render_template
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
from pathlib import Path
from system_prototype import get_image_description, get_text_embeddings, load_images_paths, get_multi_modal_embeddings
from tqdm import tqdm
import argparse
import os

app = Flask(__name__)
VECTOR_SIZE = None
COLLECTION_NAME = None
EMBEDDING_TYPE = None
IMAGE_DIR = None

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", port=6333)

def init_collection():
    """Initialize or get existing collection"""
    assert COLLECTION_NAME is not None, "Collection name is not set"
    assert VECTOR_SIZE is not None, "Vector size is not set"
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

def create_app(image_dir, embedding_type):
    global IMAGE_DIR, COLLECTION_NAME, EMBEDDING_TYPE, VECTOR_SIZE
    # Use the directory name as collection name (sanitized)
    COLLECTION_NAME = os.path.basename(os.path.normpath(image_dir)) + '_' + embedding_type
    EMBEDDING_TYPE = embedding_type
    VECTOR_SIZE = 1536 if EMBEDDING_TYPE == "separated" else 512
    IMAGE_DIR = image_dir
    # 1536 is the embedding size of Alibaba-NLP/gte-Qwen2-1.5B-instruct
    # 512 is the embedding size of openai/clip-vit-base-patch16

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

    print(f"Total number of images: {len(image_paths)}")
    print(f"Number of new images: {len(new_images)}")
    
    if new_images:
        print(f"Processing {len(new_images)} new images...")
        next_id = len(existing_hashes) + 1
        paths = [img_path for img_path, _ in new_images]
        
        if EMBEDDING_TYPE == "aligned":
            # Get embeddings for all images at once using CLIP
            embeddings = get_multi_modal_embeddings(paths, is_image=True)
            descriptions = [""] * len(paths)  # Empty descriptions for aligned mode
        else:
            # Original separated mode - process one by one
            embeddings = []
            descriptions = []
            for image_path, _ in tqdm(new_images):
                try:
                    description = get_image_description(image_path)
                    embedding = get_text_embeddings([description])[0]
                    embeddings.append(embedding)
                    descriptions.append(description)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue

        # Store in Qdrant
        points = []
        for i, ((image_path, image_hash), embedding, description) in enumerate(zip(new_images, embeddings, descriptions)):
            points.append(PointStruct(
                id=next_id + i,
                vector=embedding.tolist(),
                payload={
                    'image_path': os.path.join(IMAGE_DIR, os.path.basename(image_path)),
                    'description': description,
                    'image_hash': image_hash
                }
            ))
        
        if points:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    top_k = int(request.form.get('top_k', 5))
    
    # If top_k is -1 (unlimited), use a large number
    if top_k == -1:
        top_k = 1000
    
    if EMBEDDING_TYPE == "aligned":
        all_results = []
        
        # Process text query if present
        if query.strip():
            text_embedding = get_multi_modal_embeddings([query], is_image=False)[0]
            text_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=text_embedding.tolist(),
                limit=top_k
            )
            all_results.extend(text_results)
        
        # Process image queries
        image_index = 0
        while f'image_{image_index}' in request.files:
            image_file = request.files[f'image_{image_index}']
            if image_file:
                # Save temporarily
                temp_path = os.path.join('temp', image_file.filename)
                os.makedirs('temp', exist_ok=True)
                image_file.save(temp_path)
                
                try:
                    # Get image embedding and search
                    image_embedding = get_multi_modal_embeddings([temp_path], is_image=True)[0]
                    image_results = qdrant_client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=image_embedding.tolist(),
                        limit=top_k
                    )
                    all_results.extend(image_results)
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            image_index += 1
        
        # Combine results and remove duplicates, keeping highest scores
        results_dict = {}
        for result in all_results:
            image_path = result.payload['image_path']
            if image_path not in results_dict or results_dict[image_path].score < result.score:
                results_dict[image_path] = result
        
        # Convert back to list and sort by score
        combined_results = sorted(
            results_dict.values(),
            key=lambda x: x.score,
            reverse=True
        )[:top_k]
        
    else:
        # Original separated mode implementation
        image_descriptions = []
        image_index = 0
        while f'image_{image_index}' in request.files:
            image_file = request.files[f'image_{image_index}']
            if image_file:
                temp_path = os.path.join('temp', image_file.filename)
                os.makedirs('temp', exist_ok=True)
                image_file.save(temp_path)
                
                try:
                    image_description = get_image_description(temp_path)
                    image_descriptions.append(image_description)
                finally:
                    os.remove(temp_path)
            image_index += 1
        
        if image_descriptions:
            descriptions_text = "\n".join(f"Image {i} description: {desc}" for i, desc in enumerate(image_descriptions))
            query = f"{query}\n{descriptions_text}"
        
        query_embedding = get_text_embeddings([query])[0]
        combined_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
    
    return jsonify([{
        'image_path': result.payload['image_path'],
        'description': result.payload['description'],
        'similarity': result.score
    } for result in combined_results])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the image search server')
    parser.add_argument('image_dir', type=str, help='Directory containing images to index')
    parser.add_argument('--embedding_type', 
                       type=str,
                       default='separated',
                       choices=['separated', 'aligned'],
                       help='Type of embedding to use: "separated" for separate text and image models, "aligned" for multimodal embedding')
    args = parser.parse_args()
    
    app = create_app(args.image_dir, args.embedding_type)
    app.run(debug=True) 