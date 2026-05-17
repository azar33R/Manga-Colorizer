# Manga Colorization V3 — Project Plan

## Overview

Full rewrite of [manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2) targeting 50%+ accuracy improvement while maintaining fast inference on T4 GPUs (Kaggle/Colab).

## Key Constraints

- **Hardware**: Kaggle T4 (16GB VRAM, 12h/session, 30h/week) + Colab free tier
- **Inference speed**: Must remain fast (~130-190ms per 576px image on T4)
- **Text preservation**: Must NOT destroy text in manga panels — CRAFT detect-mask-composite
- **Training**: Cloud-only; minimize training hours; fine-tune existing weights preferred but retraining is acceptable
- **Dependencies**: Use `timm` for ConvNeXt-V2 backbone (pretrained weights)

## Architecture Decisions

| Component | Old (v2) | New (v3) | Reason |
|---|---|---|---|
| Encoder | SEResNeXt50 | ConvNeXt-V2-Tiny (timm, ImageNet pretrained) | 2x faster, better features, naturally hierarchical for U-Net |
| Denoiser | FFDNet (separate pipeline) | NAFNet-S SimpleGate blocks (integrated) | End-to-end, 2-3x faster, simpler |
| Bottleneck | 20x ResNeXt (no attention) | 6x ConvNeXt blocks + 2 self-attention layers | Long-range color consistency |
| Decoder | PixelShuffle | PixelShuffle + skip connections (kept) | Works well, no change needed |
| Discriminator | Basic PatchGAN (assumed) | Multi-scale PatchGAN (70px + 140px) + Projected VGG head | Better spatial + semantic feedback |
| Text handling | None | CRAFT detector → dilated mask → composite | Guaranteed text preservation |
| Losses | L1 + GAN (assumed) | L1(Lab) + Perceptual(VGG) + LPIPS + MS-SSIM + line preservation + text-region L1 + GAN hinge | 50%+ quality improvement |

## Architecture Diagram

```
Input (grayscale manga)
    │
    ├──► CRAFT Text Detector ──► Text Mask
    │                                      │
    ▼                                      ▼
ConvNeXt-V2-Tiny Encoder (pretrained)  Text Mask used for:
    │                                    - Compositing at inference
    │                                    - Auxiliary text-region loss
    ▼                                    - Masking hint input
NAFNet-style Denoising Block (integrated)
    │
    ▼
Dilated ConvNeXt Bottleneck + Self-Attention (2 layers)
    │
    ▼
PixelShuffle Decoder with skip connections
    │
    ▼
Colorized Output
    │
    ├──► Text Mask Composite (paste original text back)
    │
    ▼
Final Output (colorized, text preserved)
```

## Text Preservation Strategy

### Inference Pipeline
1. Detect text regions with CRAFT (~30-50ms on T4)
2. Dilate mask by 2-3px to prevent color bleed at text edges
3. Run colorizer normally on full image
4. Composite: paste original grayscale pixels back onto colorized output using the text mask

### Training
- Add text-region weighted L1 loss (5x weight on text pixels)
- This teaches the model to prefer not coloring text, so even without compositing it's better

## Loss Functions

| Loss | Weight | Details |
|---|---|---|
| L1 (Lab space) | 1.0 | Primary reconstruction loss; Lab is perceptually uniform |
| Perceptual (VGG-19) | 0.3 | Layers: conv3_4 + conv4_4 + conv5_2 |
| LPIPS | 0.05 | Learned perceptual metric (VGG backbone) |
| MS-SSIM | 0.15 | Multi-scale structural similarity |
| GAN (hinge) | 1.0 | Hinge loss for training stability |
| Line preservation | 0.5 | Canny edge L1 between input and output edges |
| Text-region L1 | 0.5 | 5x weighted L1 in detected text regions |
| Color histogram | 0.02 | EMD on Lab histograms; prevents mode collapse to gray |

## Parameter & Speed Budget

| Component | Params | FP16 Inference Time (T4) |
|---|---|---|
| Generator (ConvNeXt-V2-Tiny + decoder) | ~30M | ~80-120ms |
| Discriminator | ~8M | ~20ms (training only) |
| CRAFT text detector | ~5M | ~30-50ms |
| **Total (inference)** | **~35M** | **~130-190ms** |

## Training Plan

### Data Source
- **Danbooru2020** (Kaggle dataset) — filter `manga` + `colored` tags → ~50-100k panels
- Convert colored → grayscale as input pairs
- **Manga109** for validation (has text annotations)

### Phase 1 — Encoder Warmup (2-4h, Kaggle T4)
- Freeze ConvNeXt-V2 pretrained encoder
- Train only decoder + bottleneck with L1 + perceptual loss
- Gives reasonable baseline fast

### Phase 2 — End-to-End Fine-Tune (6-8h, Kaggle T4)
- Unfreeze encoder, lower LR by 10x
- Add GAN loss + discriminator
- Batch=4 with AMP + gradient accumulation (effective=16)
- 8-bit Adam for memory savings

### Phase 3 — Text-Aware Fine-Tune (2-4h, Kaggle T4)
- Add text-region loss + line preservation loss
- Fine-tune with text masks

**Total: ~14-16h on free Kaggle** (fits within 30h/week limit)

### Training Configuration
- Resolution: 576x576
- Batch size: 4 with AMP + 8-bit Adam
- Optimizer: AdamW (β1=0.9, β2=0.999, weight_decay=0.01)
- Scheduler: Cosine annealing with warmup (1000 steps)
- EMA: decay=0.999
- Gradient checkpointing: enabled
- Gradient accumulation: 4 steps (effective batch=16)
- Mixed precision: FP16 via `torch.cuda.amp`

## File Structure

```
manga-colorization-v3/
├── networks/
│   ├── __init__.py
│   ├── generator.py          # ConvNeXt-V2 encoder + NAFNet denoiser + decoder
│   ├── discriminator.py      # Multi-scale PatchGAN + Projected VGG head
│   ├── convnext_v2.py        # ConvNeXt-V2 backbone wrapper (timm)
│   └── nafnet.py             # NAFNet SimpleGate + SC blocks
├── text_detection/
│   ├── __init__.py
│   ├── craft.py              # CRAFT text detector
│   └── text_mask.py          # Mask generation + dilation + compositing
├── losses/
│   ├── __init__.py
│   ├── perceptual.py         # VGG perceptual loss (conv3_4/conv4_4/conv5_2)
│   ├── lpips_loss.py         # LPIPS wrapper
│   └── combined.py           # Combined loss with all components
├── utils/
│   ├── __init__.py
│   ├── utils.py              # Image processing utilities
│   └── lab.py                # RGB ↔ Lab conversion
├── colorizator.py            # Main inference class
├── inference.py              # CLI inference script
├── train.py                  # Training loop (Kaggle/Colab compatible)
├── config.py                 # All hyperparameters
├── requirements.txt          # Dependencies
└── PLAN.md                   # This file
```

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| ConvNeXt-V2 pretrained weights need grayscale adaptation | Replicate grayscale input to 3 channels (standard, works well) |
| CRAFT misses stylized manga text | Fallback: Sobel edge detection as secondary mask |
| T4 OOM at 576px batch=4 | Gradient checkpointing reduces activation memory 30-50% |
| Training from scratch is unstable | Phase 1 encoder freeze gives stable warmup; use EMA |
| Colab disconnects mid-training | Save checkpoints every 1000 steps; resume capability |
| Text composite boundary artifacts | Dilate mask 2-3px + Gaussian blur on mask edges |

## Expected Improvements

| Metric | Expected Improvement | Source |
|---|---|---|
| Color accuracy | +40-60% | ConvNeXt features + perceptual/LPIPS losses |
| Text preservation | ~100% | CRAFT detect-mask-composite guarantees it |
| Coherence | +50% | Self-attention + NAFNet denoiser |
| Denoising quality | +40% | NAFNet-S vs FFDNet |
| Overall FID | 50%+ improvement | All combined |

## Progress Tracker

- [x] Project structure + PLAN.md
- [x] networks/convnext_v2.py — ConvNeXt-V2 encoder wrapper via timm
- [x] networks/nafnet.py — NAFNet SimpleGate + SC blocks with DropPath
- [x] networks/generator.py — Full generator (ConvNeXt-V2 encoder + NAFNet denoiser + self-attention bottleneck + PixelShuffle decoder)
- [x] networks/discriminator.py — Multi-scale PatchGAN + Projected VGG discriminator
- [x] text_detection/craft.py — CRAFT text detector
- [x] text_detection/text_mask.py — Mask generation, dilation, compositing + Sobel fallback
- [x] losses/perceptual.py — VGG perceptual loss (conv3_4/conv4_4/conv5_2)
- [x] losses/lpips_loss.py — LPIPS loss with optional builtin lpips package
- [x] losses/combined.py — Combined loss (L1, Perceptual, LPIPS, MS-SSIM, Line, TextRegion, Histogram, GAN)
- [x] utils/utils.py — resize_pad, unpad, to_gray
- [x] utils/lab.py — RGB ↔ Lab conversion (tensor + numpy)
- [x] config.py — TrainConfig, ModelConfig, LossConfig, InferConfig dataclasses
- [x] train.py — Full training loop (AMP, gradient accumulation, EMA, cosine schedule, phase support)
- [x] colorizator.py — Inference class with text detection + compositing
- [x] inference.py — CLI inference script
- [x] requirements.txt
- [ ] Testing & validation (requires GPU + training data)
- [ ] Phase 1 training on Kaggle (encoder warmup)
- [ ] Phase 2 training on Kaggle (end-to-end)
- [ ] Phase 3 training on Kaggle (text-aware fine-tune)
