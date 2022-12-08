import cv2 as cv
from PIL import Image
import numpy as np
  

def adjust_brightness(img):

    cols, rows = img.shape
    brightness = np.sum(img) / (255 * cols * rows)

    minimum_brightness = 0.6                                                                                                                                
    alpha = brightness / minimum_brightness


    bright_img = cv.convertScaleAbs(img, alpha = 1, beta = 255 * (minimum_brightness - brightness))
    return bright_img


def adjust_dark_spots(img):

    img2 = cv.equalizeHist(img)
    return img2


