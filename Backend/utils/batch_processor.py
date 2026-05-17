import torch
import numpy as np
import cv2


def pad_to_largest(images):
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    padded_images = []
    pad_info = []
    
    for img in images:
        h, w = img.shape[:2]
        pad_h = max_h - h
        pad_w = max_w - w
        
        padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        padded_images.append(padded)
        pad_info.append((h, w))
    
    return padded_images, pad_info


def create_batch(images, device='cuda'):
    if len(images) == 0:
        return None, []
    
    padded_images, pad_info = pad_to_largest(images)
    
    batch = np.stack(padded_images, axis=0)
    
    if batch.dtype == np.uint8:
        batch = batch.astype(np.float32) / 255.0
    
    batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
    
    return batch_tensor, pad_info


def unbatch_results(batch_tensor, pad_info):
    results = []
    batch_np = batch_tensor.permute(0, 2, 3, 1).cpu().numpy()
    
    for i, (orig_h, orig_w) in enumerate(pad_info):
        img = batch_np[i, :orig_h, :orig_w, :]
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        results.append(img)
    
    return results


def process_batch(model_fn, images, device='cuda', batch_size=4):
    if len(images) <= 1:
        if len(images) == 1:
            return [model_fn(images[0])]
        return []
    
    all_results = []
    
    for i in range(0, len(images), batch_size):
        chunk = images[i:i+batch_size]
        batch_tensor, pad_info = create_batch(chunk, device)
        
        with torch.no_grad():
            output_tensor = model_fn(batch_tensor)
        
        chunk_results = unbatch_results(output_tensor, pad_info)
        all_results.extend(chunk_results)
    
    return all_results
