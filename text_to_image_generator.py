"""
Text-to-Image Generator using Stable Diffusion and NLP
A comprehensive AI system for generating images from text descriptions
"""

import torch
import nltk
import re
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import json
from datetime import datetime
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextToImageGenerator:
    """Advanced Text-to-Image Generator with NLP preprocessing"""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        """
        Initialize the Text-to-Image Generator
        
        Args:
            model_id (str): Hugging Face model identifier
            device (str): Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP components
        self.setup_nlp_components()
        
        # Load the Stable Diffusion pipeline
        self.load_model()
        
        # Initialize generation history
        self.generation_history = []
    
    def setup_nlp_components(self):
        """Setup NLP preprocessing components"""
        self.logger.info("Setting up NLP components...")
        
        # Initialize sentiment analysis
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        # Initialize text enhancement model
        self.text_enhancer = pipeline(
            "text2text-generation",
            model="google/flan-t5-small"
        )
        
        # Setup stopwords
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))
        
        self.logger.info("NLP components initialized successfully!")
    
    def load_model(self):
        """Load the Stable Diffusion model"""
        self.logger.info(f"Loading Stable Diffusion model: {self.model_id}")
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Optimize the pipeline
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    self.logger.warning(f"Could not enable xformers: {e}")
            
            self.logger.info("Model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_text(self, text):
        """
        Advanced NLP preprocessing of input text
        
        Args:
            text (str): Input text description
            
        Returns:
            dict: Processed text information
        """
        # Clean and normalize text
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer(text)[0]
        
        # Extract key descriptive words
        descriptive_words = [word for word in filtered_tokens 
                           if len(word) > 3 and word.isalpha()]
        
        # Enhance prompt based on sentiment
        if sentiment['label'] == 'LABEL_2':  # Positive
            style_enhancement = ", vibrant colors, beautiful lighting, high quality"
        elif sentiment['label'] == 'LABEL_0':  # Negative
            style_enhancement = ", dark atmosphere, moody lighting, dramatic"
        else:  # Neutral
            style_enhancement = ", balanced composition, natural lighting"
        
        enhanced_prompt = text + style_enhancement
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'descriptive_words': descriptive_words,
            'sentiment': sentiment,
            'enhanced_prompt': enhanced_prompt
        }
    
    def generate_image(self, text, num_images=1, guidance_scale=7.5, 
                      num_inference_steps=50, height=512, width=512, seed=None):
        """
        Generate images from text description
        
        Args:
            text (str): Text description
            num_images (int): Number of images to generate
            guidance_scale (float): Guidance scale for generation
            num_inference_steps (int): Number of denoising steps
            height (int): Image height
            width (int): Image width
            seed (int): Random seed for reproducibility
            
        Returns:
            dict: Generation results
        """
        self.logger.info(f"Generating {num_images} image(s) from text: '{text}'")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        prompt = processed_text['enhanced_prompt']
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        try:
            # Generate images
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt,
                    num_images_per_prompt=num_images,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width
                )
            
            images = result.images
            
            # Post-process images
            processed_images = []
            for i, img in enumerate(images):
                # Apply subtle enhancements
                enhanced_img = self.enhance_image(img)
                processed_images.append(enhanced_img)
            
            # Record generation
            generation_record = {
                'timestamp': datetime.now().isoformat(),
                'original_text': text,
                'enhanced_prompt': prompt,
                'processed_text': processed_text,
                'parameters': {
                    'num_images': num_images,
                    'guidance_scale': guidance_scale,
                    'num_inference_steps': num_inference_steps,
                    'height': height,
                    'width': width,
                    'seed': seed
                },
                'num_generated': len(processed_images)
            }
            
            self.generation_history.append(generation_record)
            
            return {
                'images': processed_images,
                'processed_text': processed_text,
                'generation_info': generation_record
            }
            
        except Exception as e:
            self.logger.error(f"Error generating image: {e}")
            raise
    
    def enhance_image(self, image):
        """
        Apply post-processing enhancements to generated images
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Enhanced image
        """
        # Subtle sharpening
        enhanced = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
        
        # Slight contrast enhancement
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        # Slight color enhancement
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1.05)
        
        return enhanced
    
    def save_images(self, images, output_dir="outputs", prefix="generated"):
        """
        Save generated images to disk
        
        Args:
            images (list): List of PIL Images
            output_dir (str): Output directory
            prefix (str): Filename prefix
            
        Returns:
            list: List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, img in enumerate(images):
            filename = f"{prefix}_{timestamp}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath, "PNG", quality=95)
            saved_paths.append(filepath)
            self.logger.info(f"Saved image: {filepath}")
        
        return saved_paths
    
    def save_generation_history(self, filepath="generation_history.json"):
        """Save generation history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.generation_history, f, indent=2)
        self.logger.info(f"Generation history saved to {filepath}")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text-to-Image Generator")
    parser.add_argument("--text", type=str, required=True, help="Text description")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TextToImageGenerator()
    
    # Generate images
    result = generator.generate_image(
        text=args.text,
        num_images=args.num_images,
        seed=args.seed
    )
    
    # Save images
    saved_paths = generator.save_images(
        result['images'],
        output_dir=args.output_dir
    )
    
    print(f"Generated {len(saved_paths)} images:")
    for path in saved_paths:
        print(f"  - {path}")
    
    # Save history
    generator.save_generation_history()

if __name__ == "__main__":
    main()
