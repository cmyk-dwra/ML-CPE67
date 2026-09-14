import cv2
import numpy as np


def preprocess_image(image, img_size=100):
    """Resize one image. Return None if unusable."""

    if image is None or image.size == 0:
        return None
    
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def to_features(images):
    """(n, h, w) uint8 -> (n, h*w) float32 in 0-1."""

    features = images.reshape(len(images), -1).astype(np.float32)
    # Normalize pixel values from 0-255 to 0-1
    features /= 255.0

    return features

