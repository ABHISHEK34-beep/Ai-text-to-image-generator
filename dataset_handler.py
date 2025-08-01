"""
Dataset Handler for Text-to-Image Generation
Handles various datasets including COCO, Flickr30k, and custom datasets
"""

import os
import json
import pandas as pd
import requests
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import logging

class TextImageDataset(Dataset):
    """Custom dataset class for text-image pairs"""
    
    def __init__(self, data_path, transform=None, max_samples=None):
        """
        Initialize the dataset
        
        Args:
            data_path (str): Path to dataset file (JSON or CSV)
            transform: Image transformations
            max_samples (int): Maximum number of samples to load
        """
        self.data_path = data_path
        self.transform = transform
        self.data = self.load_data(max_samples)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, max_samples=None):
        """Load dataset from file"""
        if self.data_path.endswith('.json'):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
        elif self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
            data = df.to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")
        
        if max_samples:
            data = data[:max_samples]
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'text': item.get('caption', item.get('text', '')),
            'image_path': item.get('image_path', ''),
            'metadata': item
        }

class DatasetCreator:
    """Create and manage datasets for text-to-image generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_sample_dataset(self, output_path="data/sample_dataset.json", num_samples=100):
        """
        Create a sample dataset with diverse text descriptions
        
        Args:
            output_path (str): Output file path
            num_samples (int): Number of samples to create
        """
        # Sample prompts covering various categories
        sample_prompts = [
            # Nature and landscapes
            "A serene mountain lake at sunset with reflection of peaks",
            "Dense forest with sunlight filtering through tall trees",
            "Ocean waves crashing against rocky cliffs",
            "Field of sunflowers under blue sky with white clouds",
            "Snow-covered pine trees in winter landscape",
            
            # Animals
            "Majestic lion resting under acacia tree in savanna",
            "Colorful tropical fish swimming in coral reef",
            "Eagle soaring high above mountain peaks",
            "Cute panda eating bamboo in natural habitat",
            "Butterfly with vibrant wings on blooming flower",
            
            # Architecture and cities
            "Modern skyscraper with glass facade reflecting sunset",
            "Ancient castle on hilltop surrounded by mist",
            "Busy city street at night with neon lights",
            "Traditional Japanese temple with cherry blossoms",
            "Cozy cottage with thatched roof in countryside",
            
            # Fantasy and sci-fi
            "Dragon flying over medieval castle",
            "Futuristic city with flying cars and neon lights",
            "Magical forest with glowing mushrooms and fairies",
            "Space station orbiting distant planet",
            "Wizard casting spell with glowing staff",
            
            # Art and abstract
            "Abstract painting with swirling colors and patterns",
            "Geometric shapes in vibrant colors",
            "Watercolor landscape with soft brushstrokes",
            "Digital art of cyberpunk character",
            "Minimalist design with clean lines and shapes",
            
            # Food and objects
            "Delicious chocolate cake with strawberries",
            "Vintage camera on wooden table",
            "Fresh fruits arranged in colorful display",
            "Steaming cup of coffee with latte art",
            "Antique pocket watch on old map",
            
            # People and portraits
            "Portrait of elderly man with wise expression",
            "Child playing in garden with flowers",
            "Dancer in flowing dress captured mid-movement",
            "Artist painting on canvas in studio",
            "Musician playing guitar on street corner"
        ]
        
        # Generate dataset
        dataset = []
        categories = ["nature", "animals", "architecture", "fantasy", "art", "food", "people"]
        
        for i in range(num_samples):
            prompt = np.random.choice(sample_prompts)
            category = np.random.choice(categories)
            
            # Add variations to prompts
            styles = ["photorealistic", "artistic", "detailed", "vibrant", "dramatic"]
            qualities = ["high quality", "4k", "professional", "masterpiece"]
            
            enhanced_prompt = f"{prompt}, {np.random.choice(styles)}, {np.random.choice(qualities)}"
            
            dataset.append({
                "id": f"sample_{i+1:04d}",
                "caption": prompt,
                "enhanced_caption": enhanced_prompt,
                "category": category,
                "difficulty": np.random.choice(["easy", "medium", "hard"]),
                "tags": self.extract_tags(prompt),
                "metadata": {
                    "created_date": "2024-01-01",
                    "source": "generated",
                    "quality_score": np.random.uniform(0.7, 1.0)
                }
            })
        
        # Save dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        self.logger.info(f"Created sample dataset with {num_samples} entries: {output_path}")
        return dataset
    
    def extract_tags(self, text):
        """Extract relevant tags from text description"""
        # Simple keyword extraction
        keywords = [
            "nature", "animal", "landscape", "portrait", "abstract", "fantasy",
            "architecture", "food", "technology", "art", "vintage", "modern",
            "colorful", "dramatic", "peaceful", "dynamic", "detailed"
        ]
        
        text_lower = text.lower()
        tags = [keyword for keyword in keywords if keyword in text_lower]
        return tags[:5]  # Limit to 5 tags
    
    def download_coco_captions(self, output_dir="data/coco", subset="val2017", max_samples=1000):
        """
        Download and prepare COCO captions dataset
        
        Args:
            output_dir (str): Output directory
            subset (str): Dataset subset (train2017, val2017)
            max_samples (int): Maximum samples to download
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # COCO annotations URL
        annotations_url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        
        self.logger.info("Note: For full COCO dataset, please download manually from:")
        self.logger.info("https://cocodataset.org/#download")
        
        # Create a sample COCO-style dataset structure
        sample_coco_data = {
            "info": {
                "description": "Sample COCO-style dataset for text-to-image generation",
                "version": "1.0",
                "year": 2024
            },
            "images": [],
            "annotations": []
        }
        
        # Generate sample entries
        for i in range(min(max_samples, 100)):
            image_id = i + 1
            sample_coco_data["images"].append({
                "id": image_id,
                "file_name": f"sample_{image_id:06d}.jpg",
                "height": 480,
                "width": 640
            })
            
            # Multiple captions per image (COCO style)
            captions = [
                f"Sample caption {i+1} for image {image_id}",
                f"Alternative description for image {image_id}",
                f"Detailed caption describing image {image_id}"
            ]
            
            for j, caption in enumerate(captions):
                sample_coco_data["annotations"].append({
                    "id": i * 3 + j + 1,
                    "image_id": image_id,
                    "caption": caption
                })
        
        # Save sample dataset
        output_file = os.path.join(output_dir, "sample_captions.json")
        with open(output_file, 'w') as f:
            json.dump(sample_coco_data, f, indent=2)
        
        self.logger.info(f"Created sample COCO-style dataset: {output_file}")
        return output_file
    
    def create_evaluation_dataset(self, output_path="data/evaluation_dataset.json"):
        """Create a dataset specifically for model evaluation"""
        
        evaluation_prompts = [
            # Simple objects
            {"prompt": "red apple", "difficulty": "easy", "category": "object"},
            {"prompt": "blue car", "difficulty": "easy", "category": "vehicle"},
            {"prompt": "white cat", "difficulty": "easy", "category": "animal"},
            
            # Complex scenes
            {"prompt": "sunset over mountain lake with trees", "difficulty": "medium", "category": "landscape"},
            {"prompt": "busy city street with people and cars", "difficulty": "medium", "category": "urban"},
            {"prompt": "cozy living room with fireplace and books", "difficulty": "medium", "category": "interior"},
            
            # Abstract and artistic
            {"prompt": "abstract painting with geometric shapes in primary colors", "difficulty": "hard", "category": "abstract"},
            {"prompt": "surreal landscape with floating islands and waterfalls", "difficulty": "hard", "category": "fantasy"},
            {"prompt": "photorealistic portrait of elderly person with kind eyes", "difficulty": "hard", "category": "portrait"}
        ]
        
        dataset = []
        for i, item in enumerate(evaluation_prompts):
            dataset.append({
                "id": f"eval_{i+1:03d}",
                "prompt": item["prompt"],
                "difficulty": item["difficulty"],
                "category": item["category"],
                "evaluation_criteria": [
                    "visual_quality",
                    "text_alignment",
                    "creativity",
                    "realism"
                ]
            })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        self.logger.info(f"Created evaluation dataset: {output_path}")
        return dataset

def main():
    """Main function to create datasets"""
    creator = DatasetCreator()
    
    # Create sample dataset
    creator.create_sample_dataset(num_samples=200)
    
    # Create evaluation dataset
    creator.create_evaluation_dataset()
    
    # Create sample COCO-style dataset
    creator.download_coco_captions(max_samples=50)
    
    print("Datasets created successfully!")
    print("Available datasets:")
    print("  - data/sample_dataset.json (200 samples)")
    print("  - data/evaluation_dataset.json (evaluation set)")
    print("  - data/coco/sample_captions.json (COCO-style)")

if __name__ == "__main__":
    main()
