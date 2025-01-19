from dotenv import load_dotenv
import os
import openai
import random
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

########## API LOADING ##########
# Get API key from environment variables
API_PROVIDER = os.getenv("API_PROVIDER")
if API_PROVIDER == "XAI":
    openai.api_key = os.getenv("XAI_API_KEY")
    base_url = "https://api.x.ai/v1"
    vision_model = "grok-2-vision-1212"
elif API_PROVIDER == "OPENAI":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    base_url = None
    vision_model = "gpt-4o-mini"
else:
    raise ValueError(f"Invalid API provider: {API_PROVIDER}")

client = openai.OpenAI(
  api_key=openai.api_key,
  base_url=base_url,
)
#################################

########## PROMPT LOADING ##########
with open("prompts/image_to_text_prompt_concise.txt", "r") as f:
    IMG_TO_TEXT_PROMPT = f.read().strip()
#################################

########## EMBEDDING MODEL LOADING ##########
embedding_model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
embedding_model.max_seq_length = 8192
#################################

def load_images_paths(base_dir, format="JPEG"):
    image_paths = []
    base_path = Path(base_dir)
    
    # Recursively search for images with the specified format
    for image_path in base_path.rglob(f"*.{format.lower()}"):
        image_paths.append(str(image_path))
    
    return image_paths

def get_image_description(image_path):
    # Convert file path to data URI
    try:
        with Image.open(image_path) as img:
            # Convert image to base64
            import base64
            from io import BytesIO
            
            # Convert to RGB if necessary (in case of RGBA images)
            if img.mode in ('RGBA', 'LA'):
                img = img.convert('RGB')
                
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_url = f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        raise Exception(f"Error processing image {image_path}: {str(e)}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": IMG_TO_TEXT_PROMPT,
                },
            ],
        },
    ]

    try:
        completion = client.chat.completions.create(
            model=vision_model,
            messages=messages,
            temperature=0.01,  # Low temperature for more consistent outputs
            max_tokens=1000,   # Adjust based on needed description length
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error getting description from API: {str(e)}")
    
def get_text_embeddings(text_descriptions):
    return embedding_model.encode(text_descriptions)

class ImageSearch:
    def __init__(self, images: list, descriptions: list, embeddings: np.ndarray):
        """Initialize the dataset with parallel lists/arrays of data."""
        assert len(images) == len(descriptions) == len(embeddings), "All inputs must have the same length"
        self.images = images
        self.descriptions = descriptions
        self.embeddings = embeddings
        
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.images)
    
    def search(self, query: str, top_k: int = 5):
        """Search the dataset using a text query."""
        # Ensure top_k doesn't exceed dataset size
        top_k = min(top_k, len(self))
        
        # Get query embedding and ensure it's 2D
        query_embedding = get_text_embeddings(query)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate similarities using matrix multiplication
        similarities = (query_embedding @ self.embeddings.T) * 100
        
        # Get top k indices (similarities is now a 1D array)
        top_indices = np.argsort(similarities.flatten())[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'image_path': self.images[idx],
                'description': self.descriptions[idx],
                'similarity': similarities.flatten()[idx]
            })
        
        return results

if __name__ == "__main__":
    random.seed(1221)
    NUM_IMAGES = 5

    # Load random images
    base_dir = "imagenet-mini"
    image_paths = load_images_paths(base_dir)
    print(f"{len(image_paths)} images found in {base_dir}")

    # Randomly select NUM_IMAGES images
    selected_images = random.sample(image_paths, min(NUM_IMAGES, len(image_paths)))
    print(f"Selected {len(selected_images)} random images")

    text_descriptions = []
    for i in tqdm(range(len(selected_images))):
        # visualize the images
        # image = Image.open(selected_images[i])
        # image.show()

        text_description = get_image_description(selected_images[i])
        print(text_description)
        text_descriptions.append(text_description)

        # breakpoint()

    text_embeddings = get_text_embeddings(text_descriptions)

    # Create the dataset
    data = ImageSearch(selected_images, text_descriptions, text_embeddings)
    
    # Example search
    results = data.search("show me an image of a bird", top_k=2)
    for result in results:
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Image: {result['image_path']}")
        image = Image.open(result['image_path'])
        image.show()
        print(f"Description: {result['description']}\n")
        breakpoint()
