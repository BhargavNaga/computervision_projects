import cv2
import numpy as np 
from IPython.display import Image, display 
from matplotlib import pyplot as plt

path = 'dot2.jpg'

#convert to grayscale
img = cv2.imread(path,0)
#display image




cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret,bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 

cv2.imshow('image',bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
