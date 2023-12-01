import numpy as np
import cv2

def marr_hildreth_edge_detection(image, sigma):
    # Step 1: Apply Gaussian smoothing


    smoothed_image = cv2.GaussianBlur(image, (5, 5), sigma)

    # Step 2: Compute the Laplacian of the smoothed image
    laplacian_image = cv2.Laplacian(smoothed_image, cv2.CV_64F)

    # Step 3: Find zero crossings
    edges = find_zero_crossings(laplacian_image)

    edges = 255 - edges

    return edges

def find_zero_crossings(image):
    rows, cols = image.shape
    edge_image = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Check for zero crossing in a 3x3 neighborhood
            neighbors = [image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                         image[i, j-1], image[i, j], image[i, j+1],
                         image[i+1, j-1], image[i+1, j], image[i+1, j+1]]

            is_zero_crossing = False
            for k in range(8):
                if neighbors[k] * neighbors[k+1] < 0:
                    is_zero_crossing = True
                    break

            if is_zero_crossing:
                edge_image[i, j] = 255

    return edge_image




image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

# Set the Gaussian smoothing parameter (sigma)
sigma = 2

# Perform edge detection
edges = marr_hildreth_edge_detection(image, sigma)

c = 2

sharpened_image = cv2.addWeighted(image, 1, edges, c, 0)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Marr-Hildreth Edge Detection', edges)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
