import cv2
import numpy as np


def get_text_mask(detector, image):
    text_mask = detector.detect(image)
    return text_mask


def dilate_mask(mask, kernel_size=3, iterations=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated


def blur_mask_edges(mask, kernel_size=5):
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    return blurred


def composite_text_back(original_image, colorized_image, text_mask):
    if len(text_mask.shape) == 2:
        mask_3ch = np.stack([text_mask, text_mask, text_mask], axis=2)
    else:
        mask_3ch = text_mask

    mask_float = mask_3ch.astype(np.float32) / 255.0

    original_float = original_image.astype(np.float32)
    colorized_float = colorized_image.astype(np.float32)

    composited = colorized_float * (1.0 - mask_float) + original_float * mask_float
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    return composited


def create_text_preservation_pipeline(original_image, colorized_image, detector, dilate_kernel=3, dilate_iters=2, blur_kernel=5):
    text_mask = get_text_mask(detector, original_image)

    if np.sum(text_mask) == 0:
        return colorized_image, text_mask

    dilated_mask = dilate_mask(text_mask, kernel_size=dilate_kernel, iterations=dilate_iters)
    smoothed_mask = blur_mask_edges(dilated_mask, kernel_size=blur_kernel)

    result = composite_text_back(original_image, colorized_image, smoothed_mask)

    return result, smoothed_mask
