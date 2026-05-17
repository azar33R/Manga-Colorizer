import torch


def get_vram_info():
    if not torch.cuda.is_available():
        return 0, 0, 0
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    reserved_vram = torch.cuda.memory_reserved(0) / (1024**3)
    allocated_vram = torch.cuda.memory_allocated(0) / (1024**3)
    free_vram = total_vram - allocated_vram
    
    return total_vram, free_vram, allocated_vram


def calculate_optimal_workers(vram_per_worker_gb=1.5, safety_margin_gb=2.0):
    if not torch.cuda.is_available():
        return 1
    
    total_vram, free_vram, _ = get_vram_info()
    
    usable_vram = max(0, free_vram - safety_margin_gb)
    num_workers = max(1, int(usable_vram / vram_per_worker_gb))
    
    print(f"[+] VRAM Analysis: Total={total_vram:.1f}GB, Free={free_vram:.1f}GB")
    print(f"[+] Calculated Workers: {num_workers} (using {vram_per_worker_gb}GB/worker + {safety_margin_gb}GB margin)")
    
    return num_workers


def check_vram_before_process(min_vram_gb=0.5):
    if not torch.cuda.is_available():
        return True
    
    _, free_vram, _ = get_vram_info()
    if free_vram < min_vram_gb:
        print(f"[!] Low VRAM Warning: {free_vram:.2f}GB free. Clearing cache...")
        torch.cuda.empty_cache()
        _, free_vram, _ = get_vram_info()
        return free_vram >= min_vram_gb
    return True
