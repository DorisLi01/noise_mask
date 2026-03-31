import torch
import torch.nn.functional as F

class NoiseMaskDefender:
    def __init__(self, noise_scale=0.1, mask_ratio=0.15):
        self.noise_scale = noise_scale
        self.mask_ratio = mask_ratio

    def defend(self, feature):
        # 噪声 + 掩码 + 平滑 = 同时防 PIA + SecMI
        noise = torch.randn_like(feature) * self.noise_scale
        B, C, H, W = feature.shape
        mask = torch.rand((B, 1, H, W), device=feature.device) > self.mask_ratio
        feature = feature * mask.float()
        feature = F.avg_pool2d(feature, 3, padding=1, stride=1)
        return feature + noise
