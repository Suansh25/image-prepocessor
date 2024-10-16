import cv2

# Check OpenCV version
print("OpenCV version:", cv2.__version__)

# Simple test to see if the library loads correctly
img = cv2.imread(".venv\WhatsApp Image 2023-11-25 at 07.23.42_941bf54a.jpg")  # Replace with an actual image path
if img is None:
    print("Image not loaded correctly!")
else:
    print("OpenCV is installed and working correctly!")
