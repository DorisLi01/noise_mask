"""
测试预训练DDPM模型的FID
目的：看看预训练模型本身能达到多少FID
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

PRETRAINED_MODEL_PATH = "./pretrained_models/ddpm-cifar10-32"

def calculate_fid(real_images, fake_images, device):
    """标准FID计算"""
    try:
        from scipy import linalg
        from torchvision.models import inception_v3
        import torch.nn as nn

        print("加载Inception V3...")
        inception = inception_v3(pretrained=True, transform_input=False).to(device)
        inception.eval()
        inception.fc = nn.Identity()

        def get_activations(images, batch_size=50):
            n_batches = (len(images) + batch_size - 1) // batch_size
            pred_arr = []

            # ImageNet标准化
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

            print(f"提取特征（{len(images)}张图片）...")
            for i in tqdm(range(n_batches), desc="提取特征"):
                start = i * batch_size
                end = min(start + batch_size, len(images))
                batch = images[start:end]

                # Resize到299x299
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                # 转换到[0, 1]
                batch = (batch + 1) / 2
                batch = batch.clamp(0, 1)
                # ImageNet标准化
                batch = (batch - mean) / std

                with torch.no_grad():
                    pred = inception(batch)
                    pred_arr.append(pred.cpu().numpy())

            return np.concatenate(pred_arr, axis=0)

        # 获取特征
        act_real = get_activations(real_images)
        act_fake = get_activations(fake_images)

        print("计算FID...")
        # 计算统计量
        mu_real = np.mean(act_real, axis=0)
        sigma_real = np.cov(act_real, rowvar=False)
        mu_fake = np.mean(act_fake, axis=0)
        sigma_fake = np.cov(act_fake, rowvar=False)

        # 添加正则化
        eps = 1e-6
        sigma_real = sigma_real + eps * np.eye(sigma_real.shape[0])
        sigma_fake = sigma_fake + eps * np.eye(sigma_fake.shape[0])

        # 计算FID
        diff = mu_real - mu_fake
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        return float(fid)

    except Exception as e:
        print(f"FID计算错误: {e}")
        import traceback
        traceback.print_exc()
        return 999.0


def ddpm_sample(model, scheduler, shape, device, num_inference_steps=1000):
    """DDPM采样"""
    model.eval()
    scheduler.set_timesteps(num_inference_steps)
    x = torch.randn(shape, device=device)

    print(f"DDPM采样（{num_inference_steps}步）...")
    for t in tqdm(scheduler.timesteps, desc="采样"):
        with torch.no_grad():
            t_batch = t.expand(shape[0]).to(device)
            output = model(x, t_batch)
            noise_pred = output.sample
            x = scheduler.step(noise_pred, t, x).prev_sample

    return x.clamp(-1, 1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("           测试预训练DDPM模型的FID")
    print("="*80)
    print(f"设备: {device}")
    print(f"预训练模型: {PRETRAINED_MODEL_PATH}\n")

    # 加载数据
    print("加载CIFAR-10数据集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=0)

    # 准备真实图片
    print("准备2000张真实图片...")
    real_images = []
    for images, _ in loader:
        real_images.append(images)
        if len(torch.cat(real_images)) >= 2000:
            break
    real_images = torch.cat(real_images)[:2000].to(device)
    print(f"✓ 准备了 {len(real_images)} 张真实图片")

    # 加载预训练模型
    print(f"\n加载预训练模型...")
    unet = UNet2DModel.from_pretrained(PRETRAINED_MODEL_PATH).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='linear',
        clip_sample=True,
        prediction_type='epsilon'
    )
    print("✓ 模型加载完成")

    # 生成图片
    print(f"\n生成2000张图片...")
    gen_images_list = []
    for i in range(40):  # 40批，每批50张
        print(f"\n批次 {i+1}/40:")
        gen_batch = ddpm_sample(unet, scheduler, (50, 3, 32, 32), device, num_inference_steps=1000)
        gen_images_list.append(gen_batch)

        # 保存第一批的样本图片
        if i == 0:
            from torchvision.utils import save_image
            os.makedirs("test_samples", exist_ok=True)
            for j in range(min(8, gen_batch.shape[0])):
                save_image(gen_batch[j], f"test_samples/sample_{j}.png", normalize=True, value_range=(-1, 1))
            print("✓ 保存了8张样本图片到 test_samples/")

    gen_images = torch.cat(gen_images_list, dim=0)[:2000]
    print(f"\n✓ 生成了 {len(gen_images)} 张图片")

    # 计算FID
    print("\n" + "="*80)
    print("计算FID...")
    print("="*80)
    fid = calculate_fid(real_images, gen_images, device)

    # 打印结果
    print("\n" + "="*80)
    print("                    测试结果")
    print("="*80)
    print(f"  预训练模型FID: {fid:.2f}")
    print("="*80)

    if fid < 50:
        print("\n✅ FID < 50: 预训练模型质量很好！")
        print("   → 问题可能在Fine-tuning破坏了生成质量")
    elif fid < 150:
        print("\n⚠️ FID在50-150之间: 预训练模型质量一般")
        print("   → 可能需要更好的预训练模型或更多采样步数")
    else:
        print("\n❌ FID > 150: 预训练模型质量较差")
        print("   → 建议：")
        print("     1. 检查预训练模型是否正确加载")
        print("     2. 尝试其他预训练模型")
        print("     3. 检查数据预处理是否正确")

    print("\n样本图片保存在: test_samples/")
    print("请查看生成的图片质量！\n")


if __name__ == "__main__":
    main()
