import cv2
import numpy as np

# Load the image
image_path = 'img.jpg'  # Replace with the path to your image
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale

# Step 3: Apply Gaussian smoothing to reduce noise
blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)

# Step 4: Compute the Laplacian of the smoothed image
laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

# Step 5: Detect zero-crossings in the Laplacian image
# Define a threshold for zero-crossings (adjust as needed)
threshold = 0.01

def zero_crossings(image):
    rows, cols = image.shape
    zero_crossing_map = np.zeros_like(image, dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [image[i - 1, j], image[i + 1, j], image[i, j - 1], image[i, j + 1]]
            if all(np.sign(image[i, j]) != np.sign(neighbor) for neighbor in neighbors) and abs(image[i, j]) > threshold:
                zero_crossing_map[i, j] = 255

    return zero_crossing_map

edge_map = zero_crossings(laplacian)

# Step 6: Display the edge map
cv2.imshow('Original Image', original_image)
cv2.imshow('Edge Map (Marr-Hildreth)', edge_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
