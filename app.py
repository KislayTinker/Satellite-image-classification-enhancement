# app.py
# Purpose: Interactive web demo combining our enhancement pipeline (Part 4)
# and trained classification model (Part 5/6) into one user-facing app.

import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from enhancement import enhance_pipeline
from model import SatelliteCNN


# ----- Page Setup -----
st.set_page_config(page_title="Satellite Image Classifier", layout="wide")
st.title("🛰️ Satellite Image Classification & Enhancement")
st.write("Upload a satellite image to enhance it and classify its land-use type.")


# ----- Load Model (only once, cached for speed) -----
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SatelliteCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load("models/best_satellite_cnn.pth", map_location=device))
    model.eval()
    return model, device


model, device = load_model()

class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

classify_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ----- File Upload -----
uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    original_image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(original_image)

    # ----- Step 1: Enhancement (Part 4) -----
    enhanced_np = enhance_pipeline(original_np)

    # ----- Step 2: Classification (Part 5/6) -----
    # We classify the ORIGINAL image (not the upscaled one) since our model
    # was trained on 64x64 images - resizing happens automatically in the transform
    input_tensor = classify_transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    # ----- Step 3: Display Results -----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

    with col2:
        st.subheader("Enhanced Image")
        st.image(enhanced_np, use_container_width=True)

    st.divider()

    st.subheader("Classification Result")
    st.metric(label="Predicted Land-Use Class", value=predicted_class)
    st.progress(confidence.item())
    st.write(f"Confidence: **{confidence_pct:.1f}%**")

    # Show all class probabilities as a bar chart - adds transparency/credibility
    st.subheader("All Class Probabilities")
    prob_dict = {class_names[i]: probabilities[i].item() for i in range(10)}
    st.bar_chart(prob_dict)

else:
    st.info("👆 Upload a satellite image (JPG or PNG) to get started.")