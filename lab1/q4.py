import cv2
import numpy as np

def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def marr_hildreth_edge_detection(image, sigma):
    # Step 1: Apply Gaussian smoothing
    smoothed_image = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Step 2: Compute Laplacian of Gaussian (LoG)
    laplacian_of_gaussian = convolve(smoothed_image, get_gaussian_kernel(sigma))
    
    # Step 3: Find zero crossings
    edge_image = find_zero_crossings(laplacian_of_gaussian)
    
    return edge_image

def get_gaussian_kernel(sigma):
    size = int(6 * sigma)
    if size % 2 == 0:
        size += 1
    return cv2.getGaussianKernel(size, sigma) * cv2.getGaussianKernel(size, sigma).T

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


# Load your image here
image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

# Set the Gaussian smoothing parameter (sigma)
sigma = 1.4

# Perform edge detection
edges = marr_hildreth_edge_detection(image, sigma)

# Display the result
cv2.imshow('Marr-Hildreth Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
