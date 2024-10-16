import streamlit as st
import cv2
from PIL import Image
import numpy as np

st.title("Image Analyzer")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Display the uploaded imagez
    st.image(img_array, caption="Uploaded Image.", use_column_width=True)

    # Convert the image to OpenCV format
    img_array = np.array(image)

    # Process the image: Convert to grayscale
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray_image, 100, 200)

    # Display the processed images
    st.image(gray_image, caption="Processed Image - Grayscale", use_column_width=True)
    st.image(edges, caption="Processed Image - Edges", use_column_width=True)
