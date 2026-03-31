"""
Noise Mask Defense Against Membership Inference Attacks
噪声掩码防御成员推断攻击

本模块实现了多种噪声掩码策略，用于防御针对深度学习模型的成员推断攻击(MIA)。
核心思想：通过在模型推理过程中添加精心设计的噪声掩码，扰乱攻击者利用的
统计信号（如置信度、损失值、梯度等），同时保持模型正常功能。

Author: AI Assistant
Date: 2026-03-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Union, Tuple, List
from dataclasses import dataclass
from enum import Enum
import warnings


# =============================================================================
# 配置类与枚举
# =============================================================================

class MaskStrategy(Enum):
    """噪声掩码策略枚举"""
    STATIC = "static"           # 静态噪声掩码
    ADAPTIVE = "adaptive"       # 自适应噪声掩码
    GRADIENT_BASED = "gradient" # 基于梯度的噪声掩码
    CONFIDENCE_BASED = "confidence"  # 基于置信度的噪声掩码
    FEATURE_BASED = "feature"   # 基于特征的噪声掩码


@dataclass
class NoiseMaskConfig:
    """
    噪声掩码配置类
    
    Args:
        strategy: 掩码策略类型
        noise_scale: 噪声强度/尺度 (默认: 0.1)
        noise_ratio: 被掩码的特征/神经元比例 (默认: 0.3)
        temperature: 温度参数，用于softmax平滑 (默认: 1.0)
        adaptive_factor: 自适应调整因子 (默认: 1.0)
        min_noise: 最小噪声强度 (默认: 0.01)
        max_noise: 最大噪声强度 (默认: 1.0)
        decay_rate: 噪声衰减率，用于训练时逐步降低噪声 (默认: 0.99)
        target_layers: 目标层索引列表，None表示所有层 (默认: None)
        preserve_accuracy: 是否优先保持模型准确率 (默认: True)
    """
    strategy: MaskStrategy = MaskStrategy.ADAPTIVE
    noise_scale: float = 0.1
    noise_ratio: float = 0.3
    temperature: float = 1.0
    adaptive_factor: float = 1.0
    min_noise: float = 0.01
    max_noise: float = 1.0
    decay_rate: float = 0.99
    target_layers: Optional[List[int]] = None
    preserve_accuracy: bool = True


# =============================================================================
# 基础噪声掩码模块
# =============================================================================

class BaseNoiseMask(nn.Module):
    """
    噪声掩码基类
    
    所有具体噪声掩码策略的基类，定义了统一的接口。
    """
    
    def __init__(self, config: NoiseMaskConfig):
        super().__init__()
        self.config = config
        self.current_step = 0
        
    def forward(self, x: torch.Tensor, 
                model_output: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，应用噪声掩码
        
        Args:
            x: 输入特征/激活值 [batch_size, ...]
            model_output: 模型输出（用于自适应策略）
            labels: 真实标签（用于梯度计算）
            
        Returns:
            加噪后的特征
        """
        raise NotImplementedError
        
    def get_current_noise_scale(self) -> float:
        """获取当前噪声强度（考虑衰减）"""
        decayed = self.config.noise_scale * (self.config.decay_rate ** self.current_step)
        return max(decaied, self.config.min_noise)
    
    def step(self):
        """更新步数（用于噪声衰减）"""
        self.current_step += 1


class StaticNoiseMask(BaseNoiseMask):
    """
    静态噪声掩码
    
    在指定特征上添加固定强度的高斯噪声。
    简单有效，但可能对所有样本一视同仁。
    """
    
    def __init__(self, config: NoiseMaskConfig):
        super().__init__(config)
        if config.strategy != MaskStrategy.STATIC:
            warnings.warn(f"StaticNoiseMask initialized with {config.strategy}, using STATIC")
    
    def forward(self, x: torch.Tensor,
                model_output: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        应用静态高斯噪声掩码
        
        策略：
        1. 随机选择一定比例的特征位置
        2. 在这些位置添加高斯噪声 N(0, noise_scale)
        """
        if not self.training and self.config.preserve_accuracy:
            # 推理时可选降低噪声以保护准确率
            noise_scale = self.get_current_noise_scale() * 0.5
        else:
            noise_scale = self.get_current_noise_scale()
        
        # 生成掩码：随机选择要加噪的位置
        mask = torch.rand_like(x) < self.config.noise_ratio
        
        # 生成高斯噪声
        noise = torch.randn_like(x) * noise_scale
        
        # 应用掩码噪声
        masked_x = x.clone()
        masked_x[mask] = masked_x[mask] + noise[mask]
        
        return masked_x


class AdaptiveNoiseMask(BaseNoiseMask):
    """
    自适应噪声掩码
    
    根据样本的置信度动态调整噪声强度：
    - 高置信度样本：添加更强噪声（可能是成员）
    - 低置信度样本：添加较弱噪声（可能是非成员）
    
    这样可以针对性地混淆成员推断攻击。
    """
    
    def __init__(self, config: NoiseMaskConfig):
        super().__init__(config)
    
    def forward(self, x: torch.Tensor,
                model_output: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        应用自适应噪声掩码
        
        策略：
        1. 计算每个样本的置信度（softmax后的最大概率）
        2. 置信度越高，添加的噪声越强
        3. 使用温度缩放平滑噪声强度分布
        """
        if model_output is None:
            # 如果没有模型输出，退化为静态噪声
            warnings.warn("No model_output provided, falling back to uniform noise")
            noise_scale = self.get_current_noise_scale()
            noise = torch.randn_like(x) * noise_scale
            return x + noise * self.config.noise_ratio
        
        # 计算置信度 [batch_size]
        probs = F.softmax(model_output / self.config.temperature, dim=1)
        confidence, _ = torch.max(probs, dim=1)  # [batch_size]
        
        # 自适应噪声强度：置信度越高，噪声越大
        # 形状: [batch_size, 1, 1, ...] 以便广播
        adaptive_scales = []
        for conf in confidence:
            # 高置信度 -> 高噪声，低置信度 -> 低噪声
            scale = self.config.noise_scale * (1 + self.config.adaptive_factor * conf.item())
            scale = min(scale, self.config.max_noise)
            adaptive_scales.append(scale)
        
        # 为每个样本生成不同强度的噪声
        batch_size = x.shape[0]
        masked_x = x.clone()
        
        for i in range(batch_size):
            # 生成该样本的掩码
            mask = torch.rand_like(x[i]) < self.config.noise_ratio
            noise = torch.randn_like(x[i]) * adaptive_scales[i]
            masked_x[i][mask] = masked_x[i][mask] + noise[mask]
        
        return masked_x


class GradientBasedNoiseMask(BaseNoiseMask):
    """
    基于梯度的噪声掩码
    
    利用梯度信息识别对成员推断敏感的特征：
    - 梯度大的特征往往携带更多成员信息
    - 在这些特征上添加更强的噪声
    
    需要labels来计算梯度。
    """
    
    def __init__(self, config: NoiseMaskConfig):
        super().__init__(config)
        self.gradient_buffer = {}
    
    def forward(self, x: torch.Tensor,
                model_output: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        应用基于梯度的噪声掩码
        
        策略：
        1. 计算损失对输入特征的梯度
        2. 梯度绝对值越大，说明该特征越敏感
        3. 在敏感特征上添加更强的噪声
        """
        if labels is None or model_output is None:
            warnings.warn("Gradient-based mask requires labels and model_output")
            return StaticNoiseMask(self.config).forward(x, model_output, labels)
        
        # 计算损失
        loss = F.cross_entropy(model_output, labels, reduction='none')  # [batch_size]
        
        # 为简化，我们基于损失大小调整噪声
        # 损失小的样本（可能是成员）需要更多保护
        with torch.no_grad():
            # 归一化损失到[0,1]范围
            loss_normalized = (loss - loss.min()) / (loss.max() - loss.min() + 1e-8)
            
            # 损失越小 -> 噪声越大（成员推断攻击通常关注低损失样本）
            noise_weights = 1 - loss_normalized  # [batch_size]
            
            batch_size = x.shape[0]
            masked_x = x.clone()
            
            for i in range(batch_size):
                # 根据损失调整噪声强度
                scale = self.config.noise_scale * (1 + self.config.adaptive_factor * noise_weights[i].item())
                scale = min(scale, self.config.max_noise)
                
                # 生成掩码和噪声
                mask = torch.rand_like(x[i]) < self.config.noise_ratio
                noise = torch.randn_like(x[i]) * scale
                masked_x[i][mask] = masked_x[i][mask] + noise[mask]
        
        return masked_x


class ConfidenceBasedNoiseMask(BaseNoiseMask):
    """
    基于置信度的噪声掩码（高级版）
    
    不仅考虑最大置信度，还考虑置信度分布的熵：
    - 低熵（确定性高）：可能是成员，加强噪声
    - 高熵（不确定性高）：可能是非成员，减少噪声
    """
    
    def __init__(self, config: NoiseMaskConfig):
        super().__init__(config)
    
    def forward(self, x: torch.Tensor,
                model_output: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        应用基于置信度分布的噪声掩码
        
        策略：
        1. 计算预测概率分布
        2. 计算分布熵：熵越低，模型越"确定"
        3. 确定性高的样本添加更强噪声
        """
        if model_output is None:
            return StaticNoiseMask(self.config).forward(x, model_output, labels)
        
        with torch.no_grad():
            # 计算概率分布
            probs = F.softmax(model_output / self.config.temperature, dim=1)
            
            # 计算熵: H = -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # [batch_size]
            
            # 归一化熵到[0,1]（假设最大熵为log(num_classes)）
            max_entropy = np.log(probs.shape[1])
            entropy_normalized = entropy / max_entropy
            
            # 低熵（高确定性）-> 高噪声
            certainty = 1 - entropy_normalized  # [batch_size]
            
            batch_size = x.shape[0]
            masked_x = x.clone()
            
            for i in range(batch_size):
                scale = self.config.noise_scale * (1 + self.config.adaptive_factor * certainty[i].item())
                scale = min(scale, self.config.max_noise)
                
                mask = torch.rand_like(x[i]) < self.config.noise_ratio
                noise = torch.randn_like(x[i]) * scale
                masked_x[i][mask] = masked_x[i][mask] + noise[mask]
        
        return masked_x


# =============================================================================
# 噪声掩码包装器（用于集成到现有模型）
# =============================================================================

class NoiseMaskWrapper(nn.Module):
    """
    噪声掩码模型包装器
    
    将噪声掩码无缝集成到现有PyTorch模型中，支持：
    - 在指定层插入噪声掩码
    - 灵活配置不同层的掩码策略
    - 训练和推理模式自动切换
    
    Usage:
        model = YourCNN()
        config = NoiseMaskConfig(strategy=MaskStrategy.ADAPTIVE, noise_scale=0.1)
        protected_model = NoiseMaskWrapper(model, config, target_layers=[2, 4])
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: NoiseMaskConfig,
                 target_layers: Optional[List[int]] = None,
                 mask_positions: Optional[List[str]] = None):
        """
        Args:
            model: 原始模型
            config: 噪声掩码配置
            target_layers: 要添加掩码的层索引（用于Sequential模型）
            mask_positions: 掩码位置标识（用于复杂模型，如['layer1', 'layer2']）
        """
        super().__init__()
        self.model = model
        self.config = config
        self.target_layers = target_layers or config.target_layers
        self.mask_positions = mask_positions
        
        # 初始化噪声掩码模块
        self.noise_masks = self._create_noise_masks()
        
        # 注册前向钩子
        self.hooks = []
        self.activations = {}
        self._register_hooks()
    
    def _create_noise_masks(self) -> nn.ModuleDict:
        """根据配置创建噪声掩码模块"""
        mask_map = {
            MaskStrategy.STATIC: StaticNoiseMask,
            MaskStrategy.ADAPTIVE: AdaptiveNoiseMask,
            MaskStrategy.GRADIENT_BASED: GradientBasedNoiseMask,
            MaskStrategy.CONFIDENCE_BASED: ConfidenceBasedNoiseMask,
        }
        
        mask_class = mask_map.get(self.config.strategy, StaticNoiseMask)
        
        # 为每个目标层创建一个掩码实例
        masks = nn.ModuleDict()
        if self.target_layers:
            for layer_idx in self.target_layers:
                masks[str(layer_idx)] = mask_class(self.config)
        else:
            # 默认在最后一层前添加
            masks['default'] = mask_class(self.config)
        
        return masks
    
    def _register_hooks(self):
        """注册前向传播钩子以插入噪声掩码"""
        if not self.target_layers:
            return
        
        # 获取所有子模块
        children = list(self.model.children())
        
        for layer_idx in self.target_layers:
            if layer_idx < len(children):
                layer = children[layer_idx]
                hook = layer.register_forward_hook(self._create_hook_fn(str(layer_idx)))
                self.hooks.append(hook)
    
    def _create_hook_fn(self, mask_key: str):
        """创建钩子函数"""
        def hook_fn(module, input, output):
            # 保存激活值供后续使用
            self.activations[mask_key] = output
            # 应用噪声掩码
            if mask_key in self.noise_masks:
                return self.noise_masks[mask_key](output)
            return output
        return hook_fn
    
    def forward(self, x: torch.Tensor, 
                return_clean: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入数据
            return_clean: 是否同时返回无噪声的输出（用于对比）
            
        Returns:
            加噪后的输出，或(加噪输出, 干净输出)元组
        """
        # 如果需要对比，先计算干净输出
        if return_clean:
            with torch.no_grad():
                self.eval()
                clean_output = self.model(x)
                self.train(self.training)
        
        # 正常前向传播（带噪声掩码）
        output = self.model(x)
        
        if return_clean:
            return output, clean_output
        return output
    
    def step(self):
        """更新所有噪声掩码的步数"""
        for mask in self.noise_masks.values():
            mask.step()
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# =============================================================================
# 成员推断攻击评估
# =============================================================================

class MembershipInferenceEvaluator:
    """
    成员推断攻击评估器
    
    实现多种成员推断攻击方法，用于评估噪声掩码的防御效果。
    
    支持的攻击类型：
    1. 基于置信度的攻击 (Confidence-based)
    2. 基于损失值的攻击 (Loss-based)
    3. 基于修正置信度的攻击 (Modified entropy)
    """
    
    def __init__(self, attack_type: str = "confidence"):
        """
        Args:
            attack_type: 攻击类型 ('confidence', 'loss', 'entropy')
        """
        self.attack_type = attack_type
        self.attack_model = None
    
    def compute_attack_score(self, 
                            model: nn.Module,
                            data_loader: torch.utils.data.DataLoader,
                            is_member: bool = True) -> np.ndarray:
        """
        计算攻击分数（成员推断的统计信号）
        
        Args:
            model: 目标模型
            data_loader: 数据加载器
            is_member: 是否为成员数据（用于标签）
            
        Returns:
            攻击分数数组
        """
        model.eval()
        scores = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.cuda() if torch.cuda.is_available() else inputs, labels.cuda() if torch.cuda.is_available() else labels
                
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                
                if self.attack_type == "confidence":
                    # 使用最大置信度作为攻击信号
                    score, _ = torch.max(probs, dim=1)
                    scores.extend(score.cpu().numpy())
                    
                elif self.attack_type == "loss":
                    # 使用负损失作为攻击信号（损失越小，越可能是成员）
                    loss = F.cross_entropy(outputs, labels, reduction='none')
                    score = -loss  # 负损失，越大越可能是成员
                    scores.extend(score.cpu().numpy())
                    
                elif self.attack_type == "entropy":
                    # 使用负熵作为攻击信号（熵越小，越可能是成员）
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    score = -entropy
                    scores.extend(score.cpu().numpy())
        
        return np.array(scores)
    
    def evaluate_defense(self,
                        model: nn.Module,
                        member_loader: torch.utils.data.DataLoader,
                        non_member_loader: torch.utils.data.DataLoader,
                        threshold: Optional[float] = None) -> dict:
        """
        评估防御效果
        
        Args:
            model: 被评估的模型（可能带有噪声掩码）
            member_loader: 成员数据加载器
            non_member_loader: 非成员数据加载器
            threshold: 分类阈值，None时自动计算
            
        Returns:
            评估指标字典
        """
        # 计算成员和非成员的攻击分数
        member_scores = self.compute_attack_score(model, member_loader, is_member=True)
        non_member_scores = self.compute_attack_score(model, non_member_loader, is_member=False)
        
        # 创建标签（1=成员，0=非成员）
        y_true = np.concatenate([
            np.ones(len(member_scores)),
            np.zeros(len(non_member_scores))
        ])
        y_scores = np.concatenate([member_scores, non_member_scores])
        
        # 自动确定阈值（最大化准确率）
        if threshold is None:
            from sklearn.metrics import accuracy_score
            best_threshold = 0
            best_acc = 0
            for t in np.linspace(y_scores.min(), y_scores.max(), 100):
                y_pred = (y_scores >= t).astype(int)
                acc = accuracy_score(y_true, y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_threshold = t
            threshold = best_threshold
        
        # 预测
        y_pred = (y_scores >= threshold).astype(int)
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.5,
            'threshold': threshold,
            'member_score_mean': member_scores.mean(),
            'member_score_std': member_scores.std(),
            'non_member_score_mean': non_member_scores.mean(),
            'non_member_score_std': non_member_scores.std(),
        }
        
        return results


# =============================================================================
# 辅助函数与工具
# =============================================================================

def create_noisemask_model(model: nn.Module,
                           strategy: str = "adaptive",
                           noise_scale: float = 0.5,
                           noise_ratio: float = 0.3,
                           **kwargs) -> NoiseMaskWrapper:
    """
    便捷函数：快速创建带噪声掩码保护的模型
    
    Args:
        model: 原始模型
        strategy: 掩码策略 ('static', 'adaptive', 'gradient', 'confidence')
        noise_scale: 噪声强度
        noise_ratio: 噪声掩码比例
        **kwargs: 其他配置参数
        
    Returns:
        包装后的模型
        
    Example:
        >>> model = torchvision.models.resnet18(num_classes=10)
        >>> protected_model = create_noisemask_model(
        ...     model, 
        ...     strategy='adaptive',
        ...     noise_scale=0.1,
        ...     target_layers=[4, 5, 6]
        ... )
    """
    strategy_map = {
        'static': MaskStrategy.STATIC,
        'adaptive': MaskStrategy.ADAPTIVE,
        'gradient': MaskStrategy.GRADIENT_BASED,
        'confidence': MaskStrategy.CONFIDENCE_BASED,
    }
    
    config = NoiseMaskConfig(
        strategy=strategy_map.get(strategy, MaskStrategy.ADAPTIVE),
        noise_scale=noise_scale,
        noise_ratio=noise_ratio,
        **kwargs
    )
    
    return NoiseMaskWrapper(model, config)


def compare_defense_effectiveness(model: nn.Module,
                                  protected_model: nn.Module,
                                  member_loader: torch.utils.data.DataLoader,
                                  non_member_loader: torch.utils.data.DataLoader,
                                  attack_types: List[str] = None) -> dict:
    """
    对比原始模型和受保护模型的防御效果
    
    Args:
        model: 原始模型
        protected_model: 带噪声掩码的模型
        member_loader: 成员数据
        non_member_loader: 非成员数据
        attack_types: 攻击类型列表
        
    Returns:
        对比结果字典
    """
    attack_types = attack_types or ['confidence', 'loss', 'entropy']
    results = {}
    
    for attack_type in attack_types:
        evaluator = MembershipInferenceEvaluator(attack_type)
        
        # 评估原始模型
        original_metrics = evaluator.evaluate_defense(
            model, member_loader, non_member_loader
        )
        
        # 评估受保护模型
        protected_metrics = evaluator.evaluate_defense(
            protected_model, member_loader, non_member_loader
        )
        
        results[attack_type] = {
            'original': original_metrics,
            'protected': protected_metrics,
            'improvement': {
                'accuracy_drop': original_metrics['accuracy'] - protected_metrics['accuracy'],
                'auc_drop': original_metrics['auc'] - protected_metrics['auc'],
                'f1_drop': original_metrics['f1'] - protected_metrics['f1'],
            }
        }
    
    return results


# =============================================================================
# 使用示例与测试
# =============================================================================

def demo():
    """
    完整演示：噪声掩码防御成员推断攻击
    
    本演示展示如何：
    1. 创建带噪声掩码的模型
    2. 训练/使用模型
    3. 评估防御效果
    """
    print("=" * 60)
    print("Noise Mask Defense Demo")
    print("=" * 60)
    
    # 1. 创建简单的CNN模型
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    # 2. 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleCNN(num_classes=10).to(device)
    print(f"\nOriginal model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 3. 创建带噪声掩码的模型
    config = NoiseMaskConfig(
        strategy=MaskStrategy.ADAPTIVE,
        noise_scale=0.15,
        noise_ratio=0.3,
        adaptive_factor=1.5,
        target_layers=[2, 4],  # 在第2和第4层后添加噪声
        preserve_accuracy=True
    )
    
    protected_model = NoiseMaskWrapper(model, config).to(device)
    print(f"Protected model created with noise mask strategy: {config.strategy.value}")
    print(f"  - Noise scale: {config.noise_scale}")
    print(f"  - Noise ratio: {config.noise_ratio}")
    print(f"  - Target layers: {config.target_layers}")
    
    # 4. 模拟数据（实际使用时替换为真实数据）
    from torch.utils.data import TensorDataset, DataLoader
    
    # 模拟成员数据（模型训练时见过的数据）
    member_data = torch.randn(100, 3, 32, 32)
    member_labels = torch.randint(0, 10, (100,))
    member_dataset = TensorDataset(member_data, member_labels)
    member_loader = DataLoader(member_dataset, batch_size=16, shuffle=False)
    
    # 模拟非成员数据（模型未见过的数据）
    non_member_data = torch.randn(100, 3, 32, 32)
    non_member_labels = torch.randint(0, 10, (100,))
    non_member_dataset = TensorDataset(non_member_data, non_member_labels)
    non_member_loader = DataLoader(non_member_dataset, batch_size=16, shuffle=False)
    
    # 5. 评估防御效果
    print("\n" + "=" * 60)
    print("Evaluating Defense Effectiveness")
    print("=" * 60)
    
    # 先训练一下模型（简化版）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("\nTraining model for 2 epochs...")
    model.train()
    for epoch in range(2):
        total_loss = 0
        for inputs, labels in member_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}, Loss: {total_loss/len(member_loader):.4f}")
    
    # 6. 评估成员推断攻击
    print("\nEvaluating membership inference attacks...")
    
    comparison = compare_defense_effectiveness(
        model, protected_model, member_loader, non_member_loader
    )
    
    for attack_type, metrics in comparison.items():
        print(f"\n  Attack Type: {attack_type}")
        print(f"    Original Model:")
        print(f"      - Accuracy: {metrics['original']['accuracy']:.4f}")
        print(f"      - AUC: {metrics['original']['auc']:.4f}")
        print(f"      - F1: {metrics['original']['f1']:.4f}")
        print(f"    Protected Model:")
        print(f"      - Accuracy: {metrics['protected']['accuracy']:.4f}")
        print(f"      - AUC: {metrics['protected']['auc']:.4f}")
        print(f"      - F1: {metrics['protected']['f1']:.4f}")
        print(f"    Improvement:")
        print(f"      - Accuracy drop: {metrics['improvement']['accuracy_drop']:.4f}")
        print(f"      - AUC drop: {metrics['improvement']['auc_drop']:.4f}")
    
    # 7. 测试不同噪声掩码策略
    print("\n" + "=" * 60)
    print("Comparing Different Mask Strategies")
    print("=" * 60)
    
    strategies = ['static', 'adaptive', 'confidence']
    for strategy in strategies:
        protected = create_noisemask_model(
            SimpleCNN(num_classes=10).to(device),
            strategy=strategy,
            noise_scale=0.15,
            noise_ratio=0.3
        )
        print(f"\n  Strategy: {strategy}")
        print(f"    Model ready for evaluation")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    
    return protected_model, comparison


if __name__ == "__main__":
    # 运行演示
    protected_model, comparison = demo()
