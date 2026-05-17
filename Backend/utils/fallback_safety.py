import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim(img1, img2, multichannel=True):
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (min_w, min_h))
        img2 = cv2.resize(img2, (min_w, min_h))
    
    if img1.dtype != np.uint8:
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2, 0, 255).astype(np.uint8)
    
    if len(img1.shape) == 2:
        img1 = np.stack([img1]*3, axis=-1)
    if len(img2.shape) == 2:
        img2 = np.stack([img2]*3, axis=-1)
    
    score, _ = ssim(img1, img2, channel_axis=2 if multichannel else None, full=True)
    return score


def detect_artifacts(original_img, processed_img, ssim_threshold=0.3):
    gray_orig = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY) if len(original_img.shape) == 3 else original_img
    gray_proc = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY) if len(processed_img.shape) == 3 else processed_img
    
    score = compute_ssim(gray_orig, gray_proc, multichannel=False)
    
    if score < ssim_threshold:
        print(f"[!] Artifact detected: SSIM={score:.3f} < {ssim_threshold}. Falling back to original.")
        return True, score
    
    return False, score


def safe_process(original_img, process_fn, ssim_threshold=0.3, *args, **kwargs):
    try:
        processed_img = process_fn(original_img, *args, **kwargs)
        
        has_artifacts, ssim_score = detect_artifacts(original_img, processed_img, ssim_threshold)
        
        if has_artifacts:
            return original_img, False, ssim_score
        
        return processed_img, True, ssim_score
    
    except Exception as e:
        print(f"[!] Processing error: {e}. Returning original image.")
        return original_img, False, 0.0
