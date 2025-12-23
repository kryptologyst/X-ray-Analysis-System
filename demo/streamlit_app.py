"""Streamlit demo application for X-ray analysis system."""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import os
import sys
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config, load_config
from utils.device import get_device, move_to_device
from models.xray_classifier import create_model
from explainability.cam import create_explainability_analyzer
from data.dataset import get_transforms

# Page configuration
st.set_page_config(
    page_title="X-ray Analysis System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="disclaimer">
    <h3>⚠️ IMPORTANT DISCLAIMER</h3>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <ul>
        <li>NOT approved for clinical use</li>
        <li>NOT intended for diagnosis or treatment decisions</li>
        <li>Results should NOT be used for patient care</li>
        <li>Always consult qualified healthcare professionals</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🫁 X-ray Analysis System</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")

# Model selection
model_architecture = st.sidebar.selectbox(
    "Model Architecture",
    ["resnet18", "efficientnet_b0", "vit_tiny"],
    index=0
)

# Explanation methods
explanation_methods = st.sidebar.multiselect(
    "Explanation Methods",
    ["gradcam", "scorecam"],
    default=["gradcam"]
)

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# File upload
st.sidebar.title("Upload X-ray Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an X-ray image",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a chest X-ray image for analysis"
)

# Load model function
@st.cache_resource
def load_model(architecture: str):
    """Load model with caching."""
    try:
        # Create a simple config for demo
        config = Config()
        config.model.architecture = architecture
        
        # Create model
        model = create_model(config)
        model.eval()
        
        return model, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Preprocess image function
def preprocess_image(image: Image.Image, config: Config) -> torch.Tensor:
    """Preprocess uploaded image."""
    # Get transforms
    transforms_dict = get_transforms(
        image_size=config.data.image_size,
        augmentation=False,
        normalize=config.data.normalize
    )
    
    # Apply transforms
    transform = transforms_dict["test"]
    image_tensor = transform(image)
    
    return image_tensor.unsqueeze(0)  # Add batch dimension

# Generate prediction function
def generate_prediction(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    """Generate prediction and confidence."""
    model.eval()
    
    with torch.no_grad():
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prediction = outputs.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()
        
        # Get class probabilities
        class_probs = probabilities[0].cpu().numpy()
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "class_names": ["Normal", "Pneumonia"]
        }

# Generate explanations function
def generate_explanations(
    model: torch.nn.Module, 
    image_tensor: torch.Tensor, 
    config: Config,
    methods: list,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """Generate explanation maps."""
    try:
        # Create explainability analyzer
        analyzer = create_explainability_analyzer(model, config.model.architecture)
        
        # Generate explanations
        explanations = analyzer.analyze_sample(
            image_tensor,
            methods=methods
        )
        
        return explanations
    except Exception as e:
        st.error(f"Error generating explanations: {e}")
        return {}

# Main content
if uploaded_file is not None:
    # Load model
    with st.spinner("Loading model..."):
        model, config = load_model(model_architecture)
    
    if model is None:
        st.error("Failed to load model. Please try again.")
        st.stop()
    
    # Get device
    device = get_device("auto")
    
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        
        # Image info
        st.info(f"Image size: {image.size}")
    
    with col2:
        st.subheader("Analysis Results")
        
        # Preprocess image
        with st.spinner("Processing image..."):
            image_tensor = preprocess_image(image, config)
        
        # Generate prediction
        with st.spinner("Generating prediction..."):
            prediction_result = generate_prediction(model, image_tensor, device)
        
        # Display results
        prediction = prediction_result["prediction"]
        confidence = prediction_result["confidence"]
        class_names = prediction_result["class_names"]
        class_probs = prediction_result["class_probabilities"]
        
        # Prediction card
        predicted_class = class_names[prediction]
        confidence_percent = confidence * 100
        
        # Color coding based on confidence
        if confidence >= 0.8:
            color = "green"
        elif confidence >= 0.6:
            color = "orange"
        else:
            color = "red"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Prediction: {predicted_class}</h3>
            <h2 style="color: {color};">{confidence_percent:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Class probabilities
        st.subheader("Class Probabilities")
        for i, (class_name, prob) in enumerate(zip(class_names, class_probs)):
            st.progress(prob, text=f"{class_name}: {prob:.3f}")
        
        # Confidence interpretation
        if confidence >= confidence_threshold:
            st.success(f"High confidence prediction ({confidence_percent:.1f}%)")
        else:
            st.warning(f"Low confidence prediction ({confidence_percent:.1f}%)")
    
    # Explanations section
    if explanation_methods:
        st.subheader("Model Explanations")
        
        with st.spinner("Generating explanations..."):
            explanations = generate_explanations(
                model, image_tensor, config, explanation_methods, device
            )
        
        if explanations:
            # Create explanation visualizations
            fig, axes = plt.subplots(1, len(explanations) + 1, figsize=(5 * (len(explanations) + 1), 5))
            
            if len(explanations) == 0:
                axes = [axes]
            
            # Original image
            original_image = image_tensor[0].permute(1, 2, 0).cpu().numpy()
            if original_image.shape[2] == 3:
                # Denormalize if normalized
                original_image = original_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                original_image = np.clip(original_image, 0, 1)
            
            axes[0].imshow(original_image, cmap='gray' if original_image.shape[2] == 1 else None)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Explanation maps
            for i, (method, explanation_map) in enumerate(explanations.items()):
                axes[i + 1].imshow(explanation_map, cmap='jet')
                axes[i + 1].set_title(f"{method.upper()}")
                axes[i + 1].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Explanation metrics
            st.subheader("Explanation Metrics")
            
            # Create explainability analyzer for metrics
            try:
                analyzer = create_explainability_analyzer(model, config.model.architecture)
                explanation_metrics = analyzer.compute_explanation_metrics(explanations)
                
                for method, metrics in explanation_metrics.items():
                    st.write(f"**{method.upper()}:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean Activation", f"{metrics['mean_activation']:.3f}")
                    with col2:
                        st.metric("Sparsity", f"{metrics['sparsity']:.3f}")
                    with col3:
                        st.metric("Energy Concentration", f"{metrics['energy_concentration']:.3f}")
            except Exception as e:
                st.warning(f"Could not compute explanation metrics: {e}")
    
    # Additional information
    st.subheader("Additional Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Architecture", model_architecture)
    
    with col2:
        st.metric("Device", str(device))
    
    with col3:
        st.metric("Image Size", f"{config.data.image_size}x{config.data.image_size}")

else:
    # Welcome message
    st.markdown("""
    ## Welcome to the X-ray Analysis System
    
    This is a research demonstration of an AI system for analyzing chest X-ray images.
    
    ### How to use:
    1. **Upload an image**: Use the sidebar to upload a chest X-ray image
    2. **Configure settings**: Choose model architecture and explanation methods
    3. **View results**: See predictions, confidence scores, and explanations
    
    ### Features:
    - **Multiple models**: ResNet18, EfficientNet-B0, Vision Transformer
    - **Explainability**: Grad-CAM and Score-CAM visualizations
    - **Confidence scoring**: Uncertainty quantification
    - **Real-time analysis**: Fast inference and visualization
    
    ### Sample Images:
    The system works with synthetic data for demonstration purposes.
    """)
    
    # Sample images section
    st.subheader("Sample Analysis")
    
    # Create a sample synthetic image
    if st.button("Generate Sample Analysis"):
        with st.spinner("Generating sample..."):
            # Create synthetic X-ray-like image
            np.random.seed(42)
            sample_image = np.random.normal(0.5, 0.1, (224, 224))
            
            # Add some structure
            center_x, center_y = 112, 112
            y, x = np.ogrid[:224, :224]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
            sample_image[mask] += 0.3
            
            sample_image = np.clip(sample_image, 0, 1)
            sample_image = (sample_image * 255).astype(np.uint8)
            
            # Convert to PIL Image
            sample_pil = Image.fromarray(sample_image, mode='L')
            
            # Display sample
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(sample_pil, caption="Sample Synthetic X-ray", use_column_width=True)
            
            with col2:
                st.info("This is a synthetic X-ray image generated for demonstration purposes.")
                st.success("Upload your own image using the sidebar to see real analysis results!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>X-ray Analysis System - Research Demo | Not for Clinical Use</p>
    <p>Built with PyTorch, Streamlit, and MONAI</p>
</div>
""", unsafe_allow_html=True)
