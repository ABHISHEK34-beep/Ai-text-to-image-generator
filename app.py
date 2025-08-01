"""
Text-to-Image Generator Web Application
Streamlit-based interface for the text-to-image generation system
"""

import streamlit as st
import torch
from text_to_image_generator import TextToImageGenerator
from dataset_handler import DatasetCreator
import os
import json
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="AI Text-to-Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .generation-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_generator():
    """Load the text-to-image generator (cached)"""
    return TextToImageGenerator()

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Text-to-Image Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform your words into stunning visual art using advanced AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        # Model selection
        model_options = {
            "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
            "Stable Diffusion v2.1": "stabilityai/stable-diffusion-2-1",
        }
        selected_model = st.selectbox("Select Model", list(model_options.keys()))
        
        # Generation parameters
        num_images = st.slider("Number of Images", 1, 4, 1)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        num_steps = st.slider("Inference Steps", 10, 100, 50, 5)
        
        # Image dimensions
        st.subheader("Image Dimensions")
        col1, col2 = st.columns(2)
        with col1:
            width = st.selectbox("Width", [512, 768, 1024], index=0)
        with col2:
            height = st.selectbox("Height", [512, 768, 1024], index=0)
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            seed = st.number_input("Seed (optional)", value=None, help="Set for reproducible results")
            enable_enhancement = st.checkbox("Enable NLP Enhancement", value=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Text input methods
        input_method = st.radio("Input Method", ["Type Text", "Select from Examples", "Upload Dataset"])
        
        if input_method == "Type Text":
            text_input = st.text_area(
                "Enter your text description:",
                placeholder="A beautiful sunset over a mountain lake with reflection...",
                height=150
            )
        
        elif input_method == "Select from Examples":
            example_prompts = [
                "A majestic lion in the African savanna at golden hour",
                "Futuristic city with flying cars and neon lights",
                "Peaceful zen garden with cherry blossoms and koi pond",
                "Abstract art with swirling colors and geometric patterns",
                "Cozy cabin in snowy mountains with warm light from windows",
                "Underwater scene with colorful coral reef and tropical fish",
                "Medieval castle on a cliff overlooking the ocean",
                "Robot and human shaking hands in a modern laboratory"
            ]
            text_input = st.selectbox("Choose an example:", [""] + example_prompts)
        
        else:  # Upload Dataset
            uploaded_file = st.file_uploader("Upload dataset (JSON/CSV)", type=['json', 'csv'])
            if uploaded_file:
                # Handle dataset upload
                st.success("Dataset uploaded successfully!")
                text_input = "Dataset mode - select prompt from uploaded data"
            else:
                text_input = ""
        
        # Generation button
        generate_button = st.button("🎨 Generate Images", type="primary", use_container_width=True)
        
        # Text analysis section
        if text_input and enable_enhancement:
            st.subheader("📊 Text Analysis")
            with st.spinner("Analyzing text..."):
                try:
                    generator = load_generator()
                    analysis = generator.preprocess_text(text_input)
                    
                    with st.expander("View Analysis Details"):
                        st.json({
                            "Sentiment": analysis['sentiment'],
                            "Key Words": analysis['descriptive_words'][:10],
                            "Enhanced Prompt": analysis['enhanced_prompt']
                        })
                except Exception as e:
                    st.error(f"Analysis error: {e}")
    
    with col2:
        st.header("🖼️ Generated Images")
        
        if generate_button and text_input:
            if text_input.strip():
                with st.spinner("Generating images... This may take a few minutes."):
                    try:
                        # Load generator
                        generator = load_generator()
                        
                        # Generate images
                        start_time = time.time()
                        result = generator.generate_image(
                            text=text_input,
                            num_images=num_images,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_steps,
                            height=height,
                            width=width,
                            seed=seed if seed else None
                        )
                        generation_time = time.time() - start_time
                        
                        # Display images
                        images = result['images']
                        
                        if len(images) == 1:
                            st.image(images[0], caption="Generated Image", use_column_width=True)
                        else:
                            cols = st.columns(2)
                            for i, img in enumerate(images):
                                with cols[i % 2]:
                                    st.image(img, caption=f"Image {i+1}", use_column_width=True)
                        
                        # Generation info
                        st.markdown('<div class="generation-info">', unsafe_allow_html=True)
                        st.write("**Generation Information:**")
                        st.write(f"⏱️ Generation Time: {generation_time:.2f} seconds")
                        st.write(f"🖼️ Images Generated: {len(images)}")
                        st.write(f"📏 Dimensions: {width}x{height}")
                        st.write(f"🎯 Guidance Scale: {guidance_scale}")
                        st.write(f"🔄 Steps: {num_steps}")
                        if seed:
                            st.write(f"🌱 Seed: {seed}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download buttons
                        st.subheader("💾 Download Images")
                        for i, img in enumerate(images):
                            # Convert PIL image to bytes
                            import io
                            img_bytes = io.BytesIO()
                            img.save(img_bytes, format='PNG')
                            img_bytes = img_bytes.getvalue()
                            
                            st.download_button(
                                label=f"Download Image {i+1}",
                                data=img_bytes,
                                file_name=f"generated_image_{i+1}.png",
                                mime="image/png"
                            )
                        
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
                        st.error("Please check your system requirements and try again.")
            else:
                st.warning("Please enter a text description.")
    
    # Footer with additional information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📚 About")
        st.write("""
        This AI-powered text-to-image generator uses state-of-the-art 
        Stable Diffusion models combined with advanced NLP preprocessing 
        to create stunning images from text descriptions.
        """)
    
    with col2:
        st.subheader("🔧 Features")
        st.write("""
        • Advanced NLP text preprocessing
        • Multiple model options
        • Customizable generation parameters
        • Batch image generation
        • Image enhancement post-processing
        """)
    
    with col3:
        st.subheader("💡 Tips")
        st.write("""
        • Be descriptive in your prompts
        • Include style keywords (e.g., "photorealistic", "artistic")
        • Experiment with different guidance scales
        • Use seeds for reproducible results
        """)

if __name__ == "__main__":
    main()
