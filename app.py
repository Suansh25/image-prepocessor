import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Setting up page title and layout
st.set_page_config(page_title="Pixel Matrix", layout="wide")
st.title("🔍 Data management techniques on Images.")

# Setting up sidebar for image upload and processing settings
st.sidebar.title("Upload Images for Comparison")
st.sidebar.markdown("Choose two images to compare their features.")

# Section 1: Image Upload two images
st.markdown("## 1️⃣ Upload Images")
col1, col2 = st.columns(2)
with col1:
    uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "png", "jpeg"], key="image1")
    if uploaded_file1 is not None:
        st.image(uploaded_file1, caption="First Image", use_container_width=True)

with col2:
    uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "png", "jpeg"], key="image2")
    if uploaded_file2 is not None:
        st.image(uploaded_file2, caption="Second Image", use_container_width=True)

# Function to adjust contrast
def adjust_contrast(image, level):
    return cv2.convertScaleAbs(image, alpha=level, beta=0)

# Function for image matching with feature percentage calculation
def match_images(img1, img2):
    """Match keypoints between two images using ORB and calculate the match percentage."""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # BFMatcher to match descriptors based on Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Calculate match percentage
    match_percentage = len(matches) / min(len(kp1), len(kp2)) * 100 if min(len(kp1), len(kp2)) > 0 else 0

    # Draw matches
    matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image, match_percentage

# Perform feature matching and display results
if uploaded_file1 and uploaded_file2:
    st.markdown("## 2️⃣ Feature Matching Results")
    image1 = np.array(Image.open(uploaded_file1))
    image2 = np.array(Image.open(uploaded_file2))

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Perform matching and display result
    matched_result, match_percentage = match_images(gray_image1, gray_image2)
    st.image(matched_result, caption="Matched Features Between Images", use_container_width=True)

    # Display match percentage
    st.markdown(f"### Match Percentage: **{match_percentage:.2f}%**")

# Sidebar settings for image processing
st.sidebar.header("🛠 Image Processing Settings")

if uploaded_file1:
    # Edge Detection
    st.sidebar.markdown("### 📌 Edge Detection Settings")
    edge_threshold1 = st.sidebar.slider("Threshold1", 50, 300, 100)
    edge_threshold2 = st.sidebar.slider("Threshold2", 50, 300, 200)
    
    # Apply edge detection on first image
    gray_image = cv2.cvtColor(np.array(Image.open(uploaded_file1)), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, edge_threshold1, edge_threshold2)
    st.image(edges, caption="Edges Detected", use_container_width=True)

    # Blurring Settings
    st.sidebar.markdown("### 📌 Blurring Settings")
    blur_strength = st.sidebar.slider("Blur Kernel Size", 1, 49, 15, step=2)
    blur = cv2.GaussianBlur(np.array(Image.open(uploaded_file1)), (blur_strength, blur_strength), 0)
    st.image(blur, caption="Blurred Image", use_container_width=True)

    # Thresholding Settings
    st.sidebar.markdown("### 📌 Thresholding Settings")
    threshold_value = st.sidebar.slider("Threshold Value", 0, 255, 127)
    _, thresh = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    st.image(thresh, caption="Threshold Image", use_container_width=True)

    # Noise Reduction Settings
    st.sidebar.markdown("### 📌 Noise Reduction Settings")
    d = st.sidebar.slider("Diameter", 1, 20, 9)
    sigmaColor = st.sidebar.slider("Sigma Color", 10, 150, 75)
    sigmaSpace = st.sidebar.slider("Sigma Space", 10, 150, 75)
    
    img_array = np.array(Image.open(uploaded_file1))
    noise_reduced = cv2.bilateralFilter(img_array, d, sigmaColor, sigmaSpace)
    st.image(noise_reduced, caption="Noise Reduced Image", use_container_width=True)

    # Feature extraction with ORB
    st.sidebar.markdown("### 📌 Feature Extraction Settings")
    n_features = st.sidebar.slider("Number of Features", 100, 1000, 500)
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    keypoints_image = cv2.drawKeypoints(gray_image, keypoints, None, color=(0, 255, 0))
    st.image(keypoints_image, caption="Feature Extraction with ORB", use_container_width=True)

# Footer: Instructions or Additional Information
st.markdown("---")
st.markdown("### About this Application")
st.markdown("""
This tool allows to upload two images and compare their features using the ORB (Oriented FAST and Rotated BRIEF) algorithm.
One can adjust various settings to control the matching process, including contrast, match threshold, and more.
""")
st.markdown("### Tips:")
st.markdown("""
-Recommended to use the **Contrast Adjustment** slider to enhance the visibility of features in images before matching.
- The **Edge Detection** and **Noise Reduction** settings help prepare images for better feature matching.
""")
