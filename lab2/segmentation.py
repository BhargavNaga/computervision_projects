import cv2
import numpy as np

def replace_background(input_image_path, background_image_path, output_image_path):
    # Load the input image
    person_img = cv2.imread(input_image_path)

    # Load the background image
    background_img = cv2.imread(background_image_path)

    # Resize the images to the same dimensions
    width, height = person_img.shape[1], person_img.shape[0]
    background_img = cv2.resize(background_img, (width, height))

    # Create a mask for the person using GrabCut
    mask = np.zeros(person_img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (10, 10, width - 10, height - 10)  # Define a rectangle around the person

    # Run GrabCut with initial rectangle
    cv2.grabCut(person_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create a mask where the person is labeled as probable foreground (3) or foreground (1)
    mask2 = np.where((mask == 3) | (mask == 1), 254, 0).astype('uint8')

    # Allow user to interactively refine the mask
    cv2.imshow("Segmentation Result (Press '0' for Background, '1' for Foreground)", person_img)
    mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('0'):  # Mark as background
            mask2 = cv2.subtract(mask2, cv2.bitwise_and(person_img, person_img, mask=mask2[:, :, 0]))
        elif key == ord('1'):  # Mark as foreground
            mask2 = cv2.add(mask2, cv2.bitwise_and(person_img, person_img, mask=mask2[:, :, 0]))
        elif key == 13:  # Enter key (when done)
            break

    # Apply the refined mask to the person image
    person_no_bg = cv2.bitwise_and(person_img, person_img, mask=mask2[:, :, 0])

    # Create a mask for the new background
    new_background_mask = cv2.bitwise_not(mask2[:, :, 0])

    # Apply the mask to the background image
    new_background = cv2.bitwise_and(background_img, background_img, mask=new_background_mask)

    # Combine the person with the new background
    output_image = person_no_bg + new_background

    # Save the result
    cv2.imwrite(output_image_path, output_image)

if __name__ == "__main__":
    input_image_path = 'person_image.jpg'  # Replace with your input image path
    background_image_path = 'historical_image.jpg'  # Replace with your background image path
    output_image_path = 'output_image.jpg'  # Replace with the desired output image path

    replace_background(input_image_path, background_image_path, output_image_path)