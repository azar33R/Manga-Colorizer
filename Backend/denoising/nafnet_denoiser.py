import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from urllib.request import urlretrieve


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        prev_type = x.dtype
        x = x.float()
        weight = weight.float()
        bias = bias.float()
        mu = x.mean(1, keepdim=True)
        sigma = (x - mu).pow(2).mean(1, keepdim=True)
        ctx.save_for_backward(x, weight, bias, mu, sigma)
        ctx.eps = eps
        x = (x - mu) / (sigma + eps).sqrt()
        res = (x * weight + bias).to(prev_type)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, mu, sigma = ctx.saved_tensors
        eps = ctx.eps
        grad_output = grad_output.float()
        N, C, H, W = x.shape
        diff = x - mu
        inv_std = (sigma + eps).rsqrt()
        x_hat = diff * inv_std
        grad_weight = (grad_output * x_hat).sum((0, 2, 3), keepdim=True)
        grad_bias = grad_output.sum((0, 2, 3), keepdim=True)
        grad_x_hat = grad_output * weight
        grad_sigma = (grad_x_hat * diff).sum((0, 2, 3), keepdim=True) * (-0.5) * inv_std.pow(3)
        grad_mu = (grad_x_hat * (-inv_std)).sum((0, 2, 3), keepdim=True) + grad_sigma * (-2.0) * diff.mean((0, 2, 3), keepdim=True)
        grad_x = grad_x_hat * inv_std + grad_sigma * (2.0 / (C * H * W)) * diff + grad_mu / (C * H * W)
        return grad_x, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(1, channels, 1, 1)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(1, channels, 1, 1)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ff_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = SimpleGate()
        ff_channel = c * ff_expand
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ff_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ff_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = y
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetDenoiser:
    def __init__(self, device, weights_dir='denoising/models/'):
        self.device = device
        self.weights_dir = weights_dir
        self.weights_path = os.path.join(weights_dir, 'nafnet-sidd-width32.pth')
        self.model = NAFNet(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
        self.load_weights()
        self.model.eval()

    def load_weights(self):
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_dir, exist_ok=True)
            print(f"[+] Downloading pretrained NAFNet denoiser weights...")
            url = 'https://github.com/megvii-research/NAFNet/releases/download/v0.0.1/NAFNet-SIDD-width32.pth'
            urlretrieve(url, self.weights_path)
            print(f"[+] NAFNet weights downloaded to {self.weights_path}")
        
        state_dict = torch.load(self.weights_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if 'params' in state_dict:
            state_dict = state_dict['params']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = v
        
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model = self.model.to(self.device)
        print(f"[+] Loaded NAFNet denoiser weights from {self.weights_path}")

    def get_denoised_image(self, imorig, sigma=None):
        if len(imorig.shape) < 3 or imorig.shape[2] == 1:
            imorig = np.repeat(np.expand_dims(imorig, 2), 3, 2)
        imorig = imorig[..., :3]
        
        orig_h, orig_w = imorig.shape[:2]
        
        imorig_float = imorig.astype(np.float32)
        if imorig_float.max() > 1.2:
            imorig_float = imorig_float / 255.0
        
        imorig_tensor = torch.from_numpy(imorig_float).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outim = self.model(imorig_tensor)
        
        outim = outim.squeeze(0).permute(1, 2, 0).cpu().numpy()
        outim = np.clip(outim, 0, 1)
        outim = (outim * 255.0).astype(np.uint8)
        
        return outim
