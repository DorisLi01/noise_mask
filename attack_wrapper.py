import torch
import sys
sys.path.append('./pia')
sys.path.append('./secmi')

class AttackWrapper:
    def __init__(self, attack_type='pia', strength=0.1):
        self.attack_type = attack_type
        self.strength = strength

    def attack(self, feat):
        if self.attack_type == 'pia':
            return feat + torch.randn_like(feat) * self.strength
        elif self.attack_type == 'secmi':
            return feat * (1 + 0.1 * torch.randn_like(feat))
        elif self.attack_type == 'both':
            feat = feat + torch.randn_like(feat) * self.strength
            feat = feat * (1 + 0.1 * torch.randn_like(feat))
        return feat
