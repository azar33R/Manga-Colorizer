import cv2
import numpy as np
import PIL.Image
import PIL.ImageEnhance


def unsharp_mask(image, kernel_size=5, sigma=1.0, amount=0.5):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def adaptive_saturation(image, target_mean=100, max_factor=1.5, min_factor=0.5):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    pil_img = PIL.Image.fromarray(image)
    
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    current_mean = np.mean(img_hsv[:, :, 1])
    
    if current_mean < 1:
        return image
    
    factor = min(max(target_mean / current_mean, min_factor), max_factor)
    
    enhancer = PIL.ImageEnhance.Color(pil_img)
    result = enhancer.enhance(factor)
    
    return np.array(result)


def apply_post_processing(image, sharpen=True, sharpen_amount=0.4, adjust_saturation=True, sat_target=100):
    result = image.copy()
    
    if sharpen:
        result = unsharp_mask(result, kernel_size=5, sigma=1.0, amount=sharpen_amount)
    
    if adjust_saturation:
        result = adaptive_saturation(result, target_mean=sat_target)
    
    return result
