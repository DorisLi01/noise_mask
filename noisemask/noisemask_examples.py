"""
Noise Mask Defense - Advanced Examples and Use Cases
噪声掩码防御 - 高级示例与使用案例

本文件展示如何在不同场景下使用Noise Mask防御成员推断攻击：
1. 图像分类任务
2. 大语言模型(LLM)
3. 扩散模型
4. 对抗训练集成
5. 多策略组合

Author: AI Assistant
Date: 2026-03-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# 导入基础模块
from noisemask_pytorch import (
    NoiseMaskConfig, MaskStrategy, NoiseMaskWrapper,
    MembershipInferenceEvaluator, compare_defense_effectiveness,
    create_noisemask_model
)


# =============================================================================
# 示例1: 图像分类任务 (CIFAR-10)
# =============================================================================

def example_image_classification():
    """
    图像分类任务中的Noise Mask应用
    
    使用ResNet18在CIFAR-10上展示如何防御成员推断攻击。
    """
    print("=" * 60)
    print("Example 1: Image Classification with Noise Mask")
    print("=" * 60)
    
    # 1. 数据准备
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 分割成员/非成员数据
    # 成员数据：训练集的前50%
    # 非成员数据：测试集的前50%
    member_indices = list(range(len(trainset) // 2))
    non_member_indices = list(range(len(testset) // 2))
    
    member_dataset = Subset(trainset, member_indices)
    non_member_dataset = Subset(testset, non_member_indices)
    
    member_loader = DataLoader(member_dataset, batch_size=128, shuffle=False)
    non_member_loader = DataLoader(non_member_dataset, batch_size=128, shuffle=False)
    
    # 2. 创建ResNet18模型
    model = torchvision.models.resnet18(num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nModel: ResNet18")
    print(f"Device: {device}")
    print(f"Member samples: {len(member_dataset)}")
    print(f"Non-member samples: {len(non_member_dataset)}")
    
    # 3. 训练模型（简化版）
    print("\nTraining base model...")
    train_model(model, member_loader, device, epochs=5)
    
    # 4. 创建带噪声掩码的模型
    config = NoiseMaskConfig(
        strategy=MaskStrategy.ADAPTIVE,
        noise_scale=0.1,
        noise_ratio=0.25,
        adaptive_factor=1.2,
        temperature=1.5,
        target_layers=[4, 5, 6, 7],  # ResNet的layer1-4
        preserve_accuracy=True
    )
    
    protected_model = NoiseMaskWrapper(model, config).to(device)
    print(f"\nProtected model created with {config.strategy.value} strategy")
    
    # 5. 评估防御效果
    print("\nEvaluating defense effectiveness...")
    comparison = compare_defense_effectiveness(
        model, protected_model, member_loader, non_member_loader
    )
    
    print_results(comparison)
    
    return model, protected_model, comparison


def train_model(model, dataloader, device, epochs=5, lr=0.001):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(dataloader):.4f}, Acc={acc:.2f}%")


def print_results(comparison):
    """打印评估结果"""
    print("\n" + "-" * 60)
    print("Defense Effectiveness Results")
    print("-" * 60)
    
    for attack_type, metrics in comparison.items():
        print(f"\nAttack Type: {attack_type.upper()}")
        print(f"  Original Model:")
        print(f"    Accuracy: {metrics['original']['accuracy']:.4f}")
        print(f"    AUC:      {metrics['original']['auc']:.4f}")
        print(f"    F1:       {metrics['original']['f1']:.4f}")
        print(f"  Protected Model:")
        print(f"    Accuracy: {metrics['protected']['accuracy']:.4f}")
        print(f"    AUC:      {metrics['protected']['auc']:.4f}")
        print(f"    F1:       {metrics['protected']['f1']:.4f}")
        print(f"  Improvement:")
        print(f"    Acc Drop: {metrics['improvement']['accuracy_drop']:.4f} (higher is better)")
        print(f"    AUC Drop: {metrics['improvement']['auc_drop']:.4f} (higher is better)")


# =============================================================================
# 示例2: 大语言模型(LLM)中的噪声掩码
# =============================================================================

class LLMNoiseMask(nn.Module):
    """
    大语言模型的噪声掩码模块
    
    在Transformer的注意力层或FFN层添加噪声，
    防御针对LLM的成员推断攻击。
    """
    
    def __init__(self, hidden_size: int, config: NoiseMaskConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对隐藏状态应用噪声掩码
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码
            
        Returns:
            加噪后的隐藏状态
        """
        if self.config.strategy == MaskStrategy.STATIC:
            return self._static_noise(hidden_states)
        elif self.config.strategy == MaskStrategy.ADAPTIVE:
            return self._adaptive_noise(hidden_states)
        else:
            return self._static_noise(hidden_states)
    
    def _static_noise(self, x: torch.Tensor) -> torch.Tensor:
        """静态噪声"""
        mask = torch.rand_like(x) < self.config.noise_ratio
        noise = torch.randn_like(x) * self.config.noise_scale
        return torch.where(mask, x + noise, x)
    
    def _adaptive_noise(self, x: torch.Tensor) -> torch.Tensor:
        """自适应噪声 - 基于token的L2范数"""
        # 计算每个token的范数作为敏感度指标
        token_norms = torch.norm(x, dim=-1, keepdim=True)  # [batch, seq, 1]
        
        # 范数越大（越"确定"），噪声越大
        normalized = token_norms / (token_norms.max() + 1e-8)
        scale = self.config.noise_scale * (1 + self.config.adaptive_factor * normalized)
        
        mask = torch.rand_like(x) < self.config.noise_ratio
        noise = torch.randn_like(x) * scale
        
        return torch.where(mask, x + noise, x)


def integrate_with_transformer():
    """
    展示如何将噪声掩码集成到Transformer模型
    """
    print("\n" + "=" * 60)
    print("Example 2: LLM Noise Mask Integration")
    print("=" * 60)
    
    # 使用Hugging Face Transformers示例
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # 加载预训练模型
        model_name = "bert-base-uncased"  # 可以替换为其他模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print(f"\nLoaded model: {model_name}")
        
        # 创建噪声掩码配置
        config = NoiseMaskConfig(
            strategy=MaskStrategy.ADAPTIVE,
            noise_scale=0.05,  # LLM通常需要更小的噪声
            noise_ratio=0.2,
            adaptive_factor=1.0,
            preserve_accuracy=True
        )
        
        # 包装模型（简化示例）
        # 实际使用时需要修改模型内部层
        noise_mask = LLMNoiseMask(hidden_size=768, config=config)
        
        print(f"\nNoise mask created:")
        print(f"  Strategy: {config.strategy.value}")
        print(f"  Noise scale: {config.noise_scale}")
        print(f"  Noise ratio: {config.noise_ratio}")
        
        # 测试
        text = "This is a test sentence for noise mask."
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            # 应用噪声掩码
            masked_states = noise_mask(hidden_states)
            
            print(f"\nOriginal shape: {hidden_states.shape}")
            print(f"Noise applied: {not torch.allclose(hidden_states, masked_states)}")
        
    except ImportError:
        print("\nNote: transformers library not installed.")
        print("Install with: pip install transformers")
        print("Showing conceptual example...")
        
        # 概念示例
        config = NoiseMaskConfig(
            strategy=MaskStrategy.ADAPTIVE,
            noise_scale=0.05,
            noise_ratio=0.2
        )
        noise_mask = LLMNoiseMask(hidden_size=768, config=config)
        
        # 模拟隐藏状态
        hidden_states = torch.randn(2, 10, 768)  # [batch, seq, hidden]
        masked_states = noise_mask(hidden_states)
        
        print(f"\nSimulated LLM hidden states: {hidden_states.shape}")
        print(f"After noise mask: {masked_states.shape}")
        print(f"Noise magnitude: {(masked_states - hidden_states).abs().mean():.6f}")


# =============================================================================
# 示例3: 扩散模型中的噪声掩码
# =============================================================================

class DiffusionNoiseMask:
    """
    扩散模型的噪声掩码
    
    在扩散模型的去噪过程中添加噪声掩码，
    防御针对扩散模型的成员推断攻击。
    """
    
    def __init__(self, config: NoiseMaskConfig):
        self.config = config
    
    def apply_to_latent(self, latent: torch.Tensor, 
                        timestep: int,
                        total_timesteps: int = 1000) -> torch.Tensor:
        """
        对扩散模型的潜在表示应用噪声掩码
        
        Args:
            latent: 潜在表示 [batch, channels, height, width]
            timestep: 当前时间步
            total_timesteps: 总时间步数
            
        Returns:
            加噪后的潜在表示
        """
        # 随着时间步增加，逐渐减少噪声（因为后期需要更精确的生成）
        progress = timestep / total_timesteps
        adaptive_scale = self.config.noise_scale * (1 - progress * 0.5)
        
        # 生成掩码
        mask = torch.rand_like(latent) < self.config.noise_ratio
        noise = torch.randn_like(latent) * adaptive_scale
        
        return torch.where(mask, latent + noise, latent)
    
    def apply_to_attention(self, attention_map: torch.Tensor) -> torch.Tensor:
        """
        对注意力图应用噪声掩码
        
        扩散模型中的交叉注意力可能泄露训练数据信息
        """
        # 注意力图加噪
        mask = torch.rand_like(attention_map) < self.config.noise_ratio
        noise = torch.randn_like(attention_map) * self.config.noise_scale
        
        # 确保注意力权重仍为正且归一化
        noisy_attention = torch.where(mask, attention_map + noise, attention_map)
        noisy_attention = F.softmax(noisy_attention, dim=-1)
        
        return noisy_attention


def example_diffusion_model():
    """
    扩散模型噪声掩码示例
    """
    print("\n" + "=" * 60)
    print("Example 3: Diffusion Model Noise Mask")
    print("=" * 60)
    
    config = NoiseMaskConfig(
        strategy=MaskStrategy.STATIC,
        noise_scale=0.1,
        noise_ratio=0.15,
        preserve_accuracy=True
    )
    
    noise_mask = DiffusionNoiseMask(config)
    
    # 模拟扩散模型的潜在表示
    batch_size = 4
    channels = 4
    height, width = 64, 64
    
    latent = torch.randn(batch_size, channels, height, width)
    
    print(f"\nSimulated diffusion latent: {latent.shape}")
    
    # 在不同时间步应用噪声
    for timestep in [0, 250, 500, 750, 999]:
        masked_latent = noise_mask.apply_to_latent(latent, timestep)
        noise_mag = (masked_latent - latent).abs().mean()
        print(f"  Timestep {timestep:4d}: noise magnitude = {noise_mag:.6f}")
    
    # 注意力图示例
    attention_map = torch.randn(4, 64, 77)  # [batch, spatial, text_tokens]
    masked_attention = noise_mask.apply_to_attention(attention_map)
    
    print(f"\nAttention map: {attention_map.shape}")
    print(f"After masking: {masked_attention.shape}")
    print(f"Attention sum (should be ~1): {masked_attention.sum(dim=-1).mean():.4f}")


# =============================================================================
# 示例4: 多策略组合噪声掩码
# =============================================================================

class CombinedNoiseMask(nn.Module):
    """
    组合多种噪声掩码策略
    
    结合静态噪声和自适应噪声的优势，
    在不同层使用不同策略。
    """
    
    def __init__(self, layer_configs: dict):
        """
        Args:
            layer_configs: 层配置字典 {layer_name: NoiseMaskConfig}
        """
        super().__init__()
        self.layer_configs = layer_configs
        self.masks = nn.ModuleDict()
        
        for name, config in layer_configs.items():
            if config.strategy == MaskStrategy.STATIC:
                from noisemask_pytorch import StaticNoiseMask
                self.masks[name] = StaticNoiseMask(config)
            elif config.strategy == MaskStrategy.ADAPTIVE:
                from noisemask_pytorch import AdaptiveNoiseMask
                self.masks[name] = AdaptiveNoiseMask(config)
            elif config.strategy == MaskStrategy.CONFIDENCE_BASED:
                from noisemask_pytorch import ConfidenceBasedNoiseMask
                self.masks[name] = ConfidenceBasedNoiseMask(config)
    
    def forward(self, x: torch.Tensor, layer_name: str,
                model_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """对指定层应用噪声掩码"""
        if layer_name in self.masks:
            return self.masks[layer_name](x, model_output)
        return x


def example_combined_strategy():
    """
    展示如何组合多种噪声掩码策略
    """
    print("\n" + "=" * 60)
    print("Example 4: Combined Noise Mask Strategies")
    print("=" * 60)
    
    # 为不同层配置不同策略
    layer_configs = {
        'layer1': NoiseMaskConfig(
            strategy=MaskStrategy.STATIC,
            noise_scale=0.05,
            noise_ratio=0.2
        ),
        'layer2': NoiseMaskConfig(
            strategy=MaskStrategy.ADAPTIVE,
            noise_scale=0.1,
            noise_ratio=0.25,
            adaptive_factor=1.5
        ),
        'layer3': NoiseMaskConfig(
            strategy=MaskStrategy.CONFIDENCE_BASED,
            noise_scale=0.08,
            noise_ratio=0.3,
            temperature=2.0
        ),
    }
    
    combined_mask = CombinedNoiseMask(layer_configs)
    
    print("\nCombined strategy configuration:")
    for name, config in layer_configs.items():
        print(f"  {name}: {config.strategy.value}, scale={config.noise_scale}")
    
    # 测试
    x = torch.randn(4, 128, 16, 16)  # 模拟特征图
    model_output = torch.randn(4, 10)  # 模拟模型输出
    
    for name in layer_configs.keys():
        masked = combined_mask(x, name, model_output)
        noise_mag = (masked - x).abs().mean()
        print(f"\n  {name}: noise magnitude = {noise_mag:.6f}")


# =============================================================================
# 示例5: 与对抗训练结合
# =============================================================================

class AdversarialNoiseMaskTrainer:
    """
    对抗训练 + 噪声掩码
    
    结合对抗训练和噪声掩码，提供更强大的隐私保护。
    """
    
    def __init__(self, model: nn.Module, config: NoiseMaskConfig):
        self.model = model
        self.config = config
        self.noise_mask_wrapper = NoiseMaskWrapper(model, config)
        
    def adversarial_step(self, inputs: torch.Tensor, labels: torch.Tensor,
                         epsilon: float = 0.03) -> torch.Tensor:
        """
        对抗训练步骤
        
        1. 生成对抗样本
        2. 应用噪声掩码
        3. 联合训练
        """
        inputs.requires_grad = True
        
        # 前向传播
        outputs = self.noise_mask_wrapper(inputs)
        loss = F.cross_entropy(outputs, labels)
        
        # 计算对抗梯度
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗样本
        data_grad = inputs.grad.data
        perturbed_inputs = inputs + epsilon * data_grad.sign()
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
        # 在对抗样本上应用噪声掩码
        outputs_adv = self.noise_mask_wrapper(perturbed_inputs)
        
        return outputs_adv
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            
            # 正常前向传播
            outputs = self.noise_mask_wrapper(inputs)
            loss_clean = F.cross_entropy(outputs, labels)
            
            # 对抗训练
            outputs_adv = self.adversarial_step(inputs, labels)
            loss_adv = F.cross_entropy(outputs_adv, labels)
            
            # 联合损失
            loss = loss_clean + 0.5 * loss_adv
            
            loss.backward()
            optimizer.step()
            
            # 更新噪声掩码步数
            self.noise_mask_wrapper.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)


def example_adversarial_training():
    """
    对抗训练 + 噪声掩码示例
    """
    print("\n" + "=" * 60)
    print("Example 5: Adversarial Training + Noise Mask")
    print("=" * 60)
    
    # 创建简单模型
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleNet().cuda()
    
    config = NoiseMaskConfig(
        strategy=MaskStrategy.ADAPTIVE,
        noise_scale=0.1,
        noise_ratio=0.2,
        adaptive_factor=1.0
    )
    
    trainer = AdversarialNoiseMaskTrainer(model, config)
    
    print("\nAdversarial trainer created:")
    print(f"  Strategy: {config.strategy.value}")
    print(f"  Noise scale: {config.noise_scale}")
    print(f"  Adversarial epsilon: 0.03")
    
    # 模拟训练
    dummy_data = torch.randn(32, 3, 32, 32).cuda()
    dummy_labels = torch.randint(0, 10, (32,)).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nRunning training step...")
    outputs = trainer.adversarial_step(dummy_data, dummy_labels)
    print(f"  Output shape: {outputs.shape}")
    print(f"  Training ready!")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("Noise Mask Defense - Advanced Examples")
    print("=" * 70)
    
    # 示例1: 图像分类
    try:
        example_image_classification()
    except Exception as e:
        print(f"\nExample 1 skipped: {e}")
    
    # 示例2: LLM
    integrate_with_transformer()
    
    # 示例3: 扩散模型
    example_diffusion_model()
    
    # 示例4: 组合策略
    example_combined_strategy()
    
    # 示例5: 对抗训练
    example_adversarial_training()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
