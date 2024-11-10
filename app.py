import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd

st.title("Data Management Techniques on Images")

# Uploading two images for comparison (for feature matching)
uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "png", "jpeg"])
uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "png", "jpeg"])

def match_images(img1, img2):
    """Match keypoints between two images using ORB and calculate the match percentage"""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # BFMatcher to match descriptors based on Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance

    # Calculate the percentage of good matches
    total_matches = len(matches)
    total_keypoints_img1 = len(kp1)
    total_keypoints_img2 = len(kp2)

    if total_matches == 0 or total_keypoints_img1 == 0 or total_keypoints_img2 == 0:
        match_percentage = 0
    else:
        match_percentage = (total_matches / min(total_keypoints_img1, total_keypoints_img2)) * 100

    # Draw matches
    matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return matched_image, match_percentage

if uploaded_file1 is not None and uploaded_file2 is not None:
    # Open the uploaded images
    image1 = np.array(Image.open(uploaded_file1))
    image2 = np.array(Image.open(uploaded_file2))

    # Convert to grayscale (ORB works on grayscale images)
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Display original images
    st.image(image1, caption="First Image", use_column_width=True)
    st.image(image2, caption="Second Image", use_column_width=True)

    # Perform image matching and get the match percentage
    matched_result, match_percentage = match_images(gray_image1, gray_image2)
    
    # Display the result with match percentage
    st.image(matched_result, caption="Matched Features Between Images", use_column_width=True)
    st.write(f"Match Percentage: {match_percentage:.2f}%")

# For the first uploaded image (image processing techniques)
if uploaded_file1 is not None:
    image = Image.open(uploaded_file1)
    img_array = np.array(image)

    # Display the uploaded image
    st.image(img_array, caption="Uploaded Image.", use_column_width=True)

    # Convert to grayscale
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    st.image(gray_image, caption="Grayscale Image", use_column_width=True)

    # Edge detection with adjustable thresholds
    st.sidebar.header("Edge Detection Settings")
    edge_threshold1 = st.sidebar.slider("Threshold1", 50, 300, 100)
    edge_threshold2 = st.sidebar.slider("Threshold2", 50, 300, 200)
    edges = cv2.Canny(gray_image, edge_threshold1, edge_threshold2)
    st.image(edges, caption="Edges Detected", use_column_width=True)

    # Blurring with adjustable kernel size
    st.sidebar.header("Blurring Settings")
    blur_strength = st.sidebar.slider("Blur Kernel Size", 1, 49, 15, step=2)
    blur = cv2.GaussianBlur(img_array, (blur_strength, blur_strength), 0)
    st.image(blur, caption="Blurred Image", use_column_width=True)

    # Thresholding with adjustable threshold value
    st.sidebar.header("Thresholding Settings")
    threshold_value = st.sidebar.slider("Threshold Value", 0, 255, 127)
    _, thresh = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    st.image(thresh, caption="Threshold Image", use_column_width=True)

    # # Noise reduction with adjustable parameters
    # st.sidebar.header("Noise Reduction Settings")
    # d = st.sidebar.slider("Diameter", 1, 20, 9)
    # sigmaColor = st.sidebar.slider("Sigma Color", 10, 150, 75)
    # sigmaSpace = st.sidebar.slider("Sigma Space", 10, 150, 75)
    # noise_reduced = cv2.bilateralFilter(img_array, d, sigmaColor, sigmaSpace)
    # st.image(noise_reduced, caption="Noise Reduced Image", use_column_width=True)
    # Noise reduction with adjustable parameters
    st.sidebar.header("Noise Reduction Settings")
    d = st.sidebar.slider("Diameter", 1, 20, 9)
    sigmaColor = st.sidebar.slider("Sigma Color", 10, 150, 75)
    sigmaSpace = st.sidebar.slider("Sigma Space", 10, 150, 75)

    # Ensure the image is either grayscale or RGB (removes alpha if present)
    if img_array.shape[-1] == 4:  # Check if the image has an alpha channel
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    noise_reduced = cv2.bilateralFilter(img_array, d, sigmaColor, sigmaSpace)
    st.image(noise_reduced, caption="Noise Reduced Image", use_column_width=True)


    # Feature extraction with ORB
    st.sidebar.header("Feature Extraction Settings")
    n_features = st.sidebar.slider("Number of Features", 100, 1000, 500)
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    keypoints_image = cv2.drawKeypoints(gray_image, keypoints, None, color=(0, 255, 0))
    st.image(keypoints_image, caption="Feature Extraction with ORB", use_column_width=True)
# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image

# st.title("Data Management Techniques on Images")

# # Uploading two images for comparison
# uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "png", "jpeg"])
# uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "png", "jpeg"])

# def match_images(img1, img2):
#     """Match keypoints between two images using ORB and calculate the match percentage"""
#     orb = cv2.ORB_create()
#     kp1, des1 = orb.detectAndCompute(img1, None)
#     kp2, des2 = orb.detectAndCompute(img2, None)

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#     matches = sorted(matches, key=lambda x: x.distance)

#     total_matches = len(matches)
#     total_keypoints_img1 = len(kp1)
#     total_keypoints_img2 = len(kp2)

#     match_percentage = (total_matches / min(total_keypoints_img1, total_keypoints_img2)) * 100 if total_keypoints_img1 and total_keypoints_img2 else 0
#     matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#     return matched_image, match_percentage

# if uploaded_file1 is not None and uploaded_file2 is not None:
#     # Open the uploaded images
#     image1 = np.array(Image.open(uploaded_file1))
#     image2 = np.array(Image.open(uploaded_file2))

#     # Display original images
#     st.image(image1, caption="First Image", use_column_width=True)
#     st.image(image2, caption="Second Image", use_column_width=True)

#     # Convert to grayscale (ORB works on grayscale images)
#     gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#     # Perform image matching and get the match percentage
#     matched_result, match_percentage = match_images(gray_image1, gray_image2)
    
#     # Display the result with match percentage
#     st.image(matched_result, caption="Matched Features Between Images", use_column_width=True)
    
#     # Match percentage with visual appeal:
#     # 1. Progress Bar
#     st.progress(match_percentage / 100)  # Display progress bar

#     # 2. Color-coded match percentage
#     if match_percentage >= 75:
#         st.markdown(f"<h3 style='color:green;'>Match Percentage: {match_percentage:.2f}% - Excellent Match!</h3>", unsafe_allow_html=True)
#     elif match_percentage >= 50:
#         st.markdown(f"<h3 style='color:orange;'>Match Percentage: {match_percentage:.2f}% - Good Match</h3>", unsafe_allow_html=True)
#     else:
#         st.markdown(f"<h3 style='color:red;'>Match Percentage: {match_percentage:.2f}% - Low Match</h3>", unsafe_allow_html=True)

# # For the first uploaded image (image processing techniques)
# if uploaded_file1 is not None:
#     image = Image.open(uploaded_file1)
#     img_array = np.array(image)

#     # Display the uploaded image
#     st.image(img_array, caption="Uploaded Image.", use_column_width=True)

#     # Convert to grayscale
#     gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
#     st.image(gray_image, caption="Grayscale Image", use_column_width=True)

#     # Edge detection with adjustable thresholds
#     st.sidebar.header("Edge Detection Settings")
#     edge_threshold1 = st.sidebar.slider("Threshold1", 50, 300, 100)
#     edge_threshold2 = st.sidebar.slider("Threshold2", 50, 300, 200)
#     edges = cv2.Canny(gray_image, edge_threshold1, edge_threshold2)
#     st.image(edges, caption="Edges Detected", use_column_width=True)

#     # Blurring with adjustable kernel size
#     st.sidebar.header("Blurring Settings")
#     blur_strength = st.sidebar.slider("Blur Kernel Size", 1, 49, 15, step=2)
#     blur = cv2.GaussianBlur(img_array, (blur_strength, blur_strength), 0)
#     st.image(blur, caption="Blurred Image", use_column_width=True)

#     # Thresholding with adjustable threshold value
#     st.sidebar.header("Thresholding Settings")
#     threshold_value = st.sidebar.slider("Threshold Value", 0, 255, 127)
#     _, thresh = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
#     st.image(thresh, caption="Threshold Image", use_column_width=True)

#     # Noise reduction with adjustable parameters
#     st.sidebar.header("Noise Reduction Settings")
#     d = st.sidebar.slider("Diameter", 1, 20, 9)
#     sigmaColor = st.sidebar.slider("Sigma Color", 10, 150, 75)
#     sigmaSpace = st.sidebar.slider("Sigma Space", 10, 150, 75)
#     noise_reduced = cv2.bilateralFilter(img_array, d, sigmaColor, sigmaSpace)
#     st.image(noise_reduced, caption="Noise Reduced Image", use_column_width=True)

#     # Feature extraction with ORB
#     st.sidebar.header("Feature Extraction Settings")
#     n_features = st.sidebar.slider("Number of Features", 100, 1000, 500)
#     orb = cv2.ORB_create(nfeatures=n_features)
#     keypoints, descriptors = orb.detectAndCompute(gray_image, None)
#     keypoints_image = cv2.drawKeypoints(gray_image, keypoints, None, color=(0, 255, 0))
#     st.image(keypoints_image, caption="Feature Extraction with ORB", use_column_width=True)
