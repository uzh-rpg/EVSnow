import cv2
import numpy as np

def process_img(img, isgray=True):
    if isgray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    img = np.clip(img, 0, 1)
    return img

def load_img(filepath, isgray=True):
    img = cv2.imread(filepath)
    if img is None:
        print(f"Error: Failed to load image from {filepath}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isgray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = process_img(img, isgray)
    return img