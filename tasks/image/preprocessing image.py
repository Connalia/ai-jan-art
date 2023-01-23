import cv2
import numpy as np

if __name__ == "__main__":
    image = cv2.imread('../../data/images_high/sc133188_cartouches_red1.png', 1)

    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    cv2.imshow('original', image)
    cv2.imshow('adjusted', adjusted)
    cv2.waitKey()