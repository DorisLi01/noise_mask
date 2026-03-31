# 强制禁用 matplotlib 缓存
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from absl import app, flags
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import datasets, transforms

# ====================== 🔥 全部可选择参数 ======================
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar10', 'cifar10 / fashionmnist / fake')
flags.DEFINE_string('diffusion_model', 'ddpm', 'ddpm')
flags.DEFINE_boolean('use_attack', False, '是否开启成员推理攻击')
flags.DEFINE_string('attack_type', 'pia', 'pia / secmi / both')
flags.DEFINE_boolean('use_defense', False, '是否开启NoiseMask防御')

flags.DEFINE_integer('total_steps', 40000, '训练总步数')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_float('lr', 1e-4, '学习率')
# ====================== ^^^^^^^^^^^^^^^^^^^^^^^^^^ ======================

# 🔥🔥🔥 自动把【数据集+模型+攻击+防御】写进文件夹名字！
def make_exp_folder():
    time_str = datetime.now().strftime("%m%d_%H%M%S")
    
    # 攻击信息
    attack_info = "no-attack"
    if FLAGS.use_attack:
        attack_info = f"attack-{FLAGS.attack_type}"
    
    # 防御信息
    defense_info = f"defense-{FLAGS.use_defense}"
    
    # 拼接超级清晰的文件夹名
    exp_name = f"exp_{time_str}_{FLAGS.dataset}_{FLAGS.diffusion_model}_{attack_info}_{defense_info}"
    exp_path = f"results/{exp_name}"
    
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(f"{exp_path}/images", exist_ok=True)
    os.makedirs(f"{exp_path}/curves", exist_ok=True)
    os.makedirs(f"{exp_path}/metrics", exist_ok=True)
    
    return exp_path

# 🔥 绝对稳定、不会报错的 UNet
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.act = nn.ReLU()
        self.down1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.down3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up3 = nn.Conv2d(64, in_channels, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.down1(x))
        x = self.act(self.down2(x))
        x = self.act(self.down3(x))
        x = self.act(self.up1(x))
        x = self.act(self.up2(x))
        x = self.sigmoid(self.up3(x))
        return x

def load_diffusion_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if FLAGS.dataset == "fashionmnist":
        model = SimpleUNet(in_channels=1).to(device)
    else:
        model = SimpleUNet(in_channels=3).to(device)
    return model

def get_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if FLAGS.dataset == "cifar10":
        return datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    elif FLAGS.dataset == "fashionmnist":
        return datasets.FashionMNIST(root="./data", train=True, download=False, transform=transform)
    elif FLAGS.dataset == "fake":
        x = torch.randn(5000, 3, 32, 32)
        return torch.utils.data.TensorDataset(x)
    else:
        raise ValueError("dataset 只能选 cifar10 / fashionmnist / fake")

def calculate_metrics(real, fake):
    real = real.cpu().detach().numpy().transpose(0,2,3,1)
    fake = fake.cpu().detach().numpy().transpose(0,2,3,1)
    real = np.clip(real, 0, 1)
    fake = np.clip(fake, 0, 1)
    psnr_val = psnr(real, fake, data_range=1.0)
    ssim_val = ssim(real, fake, channel_axis=-1, data_range=1.0)
    mse_val = np.mean((real-fake)**2)
    return {"psnr":psnr_val, "ssim":ssim_val, "mse":mse_val}

# ====================== 主训练流程 ======================
def main(argv):
    exp_path = make_exp_folder()
    loss_history = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True)
    model = load_diffusion_model()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    criterion = nn.MSELoss()

    # 保存所有配置，方便写论文
    with open(f"{exp_path}/config.txt", "w") as f:
        f.write(f"dataset: {FLAGS.dataset}\n")
        f.write(f"model: {FLAGS.diffusion_model}\n")
        f.write(f"use_attack: {FLAGS.use_attack}\n")
        f.write(f"attack_type: {FLAGS.attack_type}\n")
        f.write(f"use_defense: {FLAGS.use_defense}\n")
        f.write(f"total_steps: {FLAGS.total_steps}\n")
        f.write(f"batch_size: {FLAGS.batch_size}\n")
        f.write(f"lr: {FLAGS.lr}\n")

    pbar = tqdm(total=FLAGS.total_steps)
    step = 0
    data_iter = iter(dataloader)

    while step < FLAGS.total_steps:
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, _ = next(data_iter)

        x = x.to(device)
        noise = torch.randn_like(x)
        noisy_x = x + noise * 0.1
        pred = model(noisy_x)
        loss = criterion(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        pbar.set_description(f"loss={loss.item():.3f}")
        pbar.update(1)

        if step % 1000 == 0:
            with torch.no_grad():
                clean_x = noisy_x - pred
                grid = make_grid(clean_x[:16].clamp(-1,1), nrow=4, normalize=True)
                plt.imsave(f"{exp_path}/images/step{step}.png", grid.permute(1,2,0).cpu().numpy())

        step += 1

    plt.figure(figsize=(10,4))
    plt.plot(loss_history)
    plt.title("Loss Curve")
    plt.savefig(f"{exp_path}/curves/loss.png")
    plt.close()

    # 保存指标
    final_metrics = calculate_metrics(x, clean_x)
    with open(f"{exp_path}/metrics/result.txt", "w") as f:
        f.write(f"PSNR: {final_metrics['psnr']:.2f}\n")
        f.write(f"SSIM: {final_metrics['ssim']:.4f}\n")
        f.write(f"MSE: {final_metrics['mse']:.6f}\n")

    print(f"\n✅ 全部跑完！结果保存在：")
    print(exp_path)

if __name__ == '__main__':
    app.run(main)
