import cv2
import numpy as np

# Load the image
image_path ='img.jpg'  # Replace with the path to your image
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale

# Apply Gaussian blur to reduce noise and enhance edge detection
blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)

cv2.imshow('Blurred_image', blurred_image)


# Apply Canny edge detection
edges = cv2.Canny(blurred_image, threshold1=30, threshold2=100)

# Create a mask for the edges
sharpened_edges = edges

c = 2


sharpened_image = cv2.addWeighted(original_image, 1, sharpened_edges, c, 0)

# Display the results
cv2.imshow('Original Image', original_image)
cv2.imshow('Edges (Sharpened)', sharpened_edges)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
