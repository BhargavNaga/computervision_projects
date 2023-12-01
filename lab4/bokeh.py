import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog




def apply_custom_bokeh(image_path, output_path):
   

    img = cv2.imread(image_path)

    # Create a mask initialized with zeros
    mask = np.zeros(img.shape[:2], np.uint8)

    # Define background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define a rectangle to specify the initial region for segmentation
    rect = (50, 50, img.shape[1]-50, img.shape[0]-50)

    # Apply GrabCut algorithm
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to create a binary mask for foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply the original image with the binary mask to get the foreground
    result = img * mask2[:, :, np.newaxis]

    
    blurImg = cv2.blur(img, (10, 10))

    blurImg[mask2 != 0] = [0, 0, 0]


    bokeh = result+ blurImg

    cv2.imwrite(output_path,bokeh)






def openfile():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
    input_entry.delete(0, tk.END)
    input_entry.insert(0, file_path)


def apply_custom_bokeh_effect():
    input_image_path = input_entry.get()
    output_image_path = output_entry.get()


    if not output_image_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        output_image_path += ".jpg"

    apply_custom_bokeh(input_image_path, output_image_path)
    status_label.config(text=f"Bokeh effect applied and saved as {output_image_path}")


root = tk.Tk()
root.title("Bokeh Effect")

input_label = tk.Label(root, text="Choose your Image:")
input_label.pack()
input_entry = tk.Entry(root, width=50)
input_entry.pack()
browse_button = tk.Button(root, text="Open Image", command=openfile)
browse_button.pack()

output_label = tk.Label(root, text="Final Image:")
output_label.pack()
output_entry = tk.Entry(root, width=50)
output_entry.pack()

apply_button = tk.Button(root, text="Apply", command=apply_custom_bokeh_effect)
apply_button.pack()

status_label = tk.Label(root, text="")
status_label.pack()

root.mainloop()