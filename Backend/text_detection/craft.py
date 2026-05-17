import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        # VGG16 backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True)
        )

        # U-Net decoder
        self.upconv1 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.upconv4 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True)
        )

        # Output heads
        self.region_head = nn.Conv2d(32, 1, 1)
        self.affinity_head = nn.Conv2d(32, 1, 1)

        if pretrained:
            self._load_pretrained()

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def _load_pretrained(self):
        weights_dir = 'text_detection/models'
        weights_path = os.path.join(weights_dir, 'craft_mlt_25k.pth')
        if not os.path.exists(weights_path):
            os.makedirs(weights_dir, exist_ok=True)
            print("[+] Downloading pretrained CRAFT weights...")
            urls = [
                'https://github.com/clovaai/CRAFT-pytorch/raw/master/craft_mlt_25k.pth',
                'https://huggingface.co/spaces/akhaliq/CRAFT/resolve/main/craft_mlt_25k.pth',
                'https://github.com/BinitDOX/Manga-Colorizer/releases/download/v2.0/craft_mlt_25k.pth'
            ]
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            success = False
            for url in urls:
                try:
                    print(f"    Trying: {url}")
                    r = requests.get(url, headers=headers, timeout=30, stream=True)
                    if r.status_code == 200:
                        with open(weights_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        success = True
                        break
                except Exception as e:
                    print(f"    Failed: {e}")
                    continue
            if not success:
                print(f"[!] CRAFT download failed. Text preservation will be disabled.")
                print(f"    Manual fix: download craft_mlt_25k.pth to {weights_path}")
                return False
            print(f"[+] CRAFT weights downloaded to {weights_path}")

        if not os.path.exists(weights_path):
            return False
        state_dict = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = v
        self.load_state_dict(new_state_dict, strict=False)
        print(f"[+] Loaded CRAFT weights")
        return True

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Decoder with skip connections
        up1 = F.interpolate(conv5, scale_factor=2, mode='bilinear', align_corners=False)
        cat1 = torch.cat([up1, conv4], dim=1)
        upconv1 = self.upconv1(cat1)

        up2 = F.interpolate(upconv1, scale_factor=2, mode='bilinear', align_corners=False)
        cat2 = torch.cat([up2, conv3], dim=1)
        upconv2 = self.upconv2(cat2)

        up3 = F.interpolate(upconv2, scale_factor=2, mode='bilinear', align_corners=False)
        cat3 = torch.cat([up3, conv2], dim=1)
        upconv3 = self.upconv3(cat3)

        up4 = F.interpolate(upconv3, scale_factor=2, mode='bilinear', align_corners=False)
        cat4 = torch.cat([up4, conv1], dim=1)
        upconv4 = self.upconv4(cat4)

        # Output
        region = torch.sigmoid(self.region_head(upconv4))
        affinity = torch.sigmoid(self.affinity_head(upconv4))

        return region, affinity


class CraftTextDetector:
    def __init__(self, device, text_threshold=0.75, low_text=0.5, link_threshold=0.2):
        self.device = device
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.model = CRAFT(pretrained=True).to(device)
        self.model.eval()

    def detect(self, image):
        # image: numpy array HxWx3, uint8
        # returns: binary mask HxW, uint8

        orig_h, orig_w = image.shape[:2]

        # Preprocess
        img_resized = cv2.resize(image, (orig_w, orig_h))
        img_float = img_resized.astype(np.float32)
        img_float = (img_float / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        # Convert to tensor
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            region, affinity = self.model(img_tensor)

        # Post-process
        region = region[0, 0].cpu().numpy()
        affinity = affinity[0, 0].cpu().numpy()

        # Resize back to original
        region = cv2.resize(region, (orig_w, orig_h))
        affinity = cv2.resize(affinity, (orig_w, orig_h))

        # Generate text score
        text_score = region + affinity
        text_score = np.clip(text_score, 0, 1)

        # Binary mask
        text_mask = (text_score > self.text_threshold).astype(np.uint8) * 255

        return text_mask
