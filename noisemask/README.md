# Noise Mask 噪声掩码防御成员推断攻击

本项目实现了多种噪声掩码策略，用于防御针对深度学习模型的**成员推断攻击 (Membership Inference Attack, MIA)**。

## 📋 项目概述

### 什么是成员推断攻击？

成员推断攻击是一种隐私攻击，攻击者试图判断某个特定样本是否被用于训练目标模型。这类攻击利用了模型在训练数据（成员）和非训练数据（非成员）上的统计差异（如置信度、损失值等）。

### 噪声掩码防御原理

通过在模型推理过程中添加精心设计的噪声掩码，扰乱攻击者利用的统计信号，同时尽量保持模型的正常功能。

核心策略：
- **高置信度样本**（可能是成员）→ 添加更强噪声
- **低置信度样本**（可能是非成员）→ 添加较弱噪声

## 🗂️ 文件结构

```
noisemask/
├── noisemask_pytorch.py      # PyTorch实现
├── noisemask_tensorflow.py   # TensorFlow实现
├── noisemask_examples.py     # 高级示例与使用案例
└── README.md                 # 本文件
```

## 🚀 快速开始

### 安装依赖

```bash
# PyTorch版本
pip install torch torchvision numpy scikit-learn

# TensorFlow版本
pip install tensorflow numpy scikit-learn

# 可选：用于LLM示例
pip install transformers
```

### 基础用法

#### PyTorch

```python
import torchvision
from noisemask_pytorch import (
    NoiseMaskConfig, MaskStrategy, NoiseMaskWrapper,
    create_noisemask_model
)

# 1. 创建基础模型
model = torchvision.models.resnet18(num_classes=10)

# 2. 配置噪声掩码
config = NoiseMaskConfig(
    strategy=MaskStrategy.ADAPTIVE,  # 自适应噪声
    noise_scale=0.1,                  # 噪声强度
    noise_ratio=0.3,                  # 掩码比例
    target_layers=[4, 5, 6]          # 目标层
)

# 3. 包装模型
protected_model = NoiseMaskWrapper(model, config)

# 4. 正常使用
output = protected_model(input_data)
```

#### TensorFlow

```python
import tensorflow as tf
from noisemask_tensorflow import create_noisemask_model_tf

# 1. 创建基础模型
base_model = tf.keras.applications.ResNet50(weights=None, classes=10)

# 2. 创建带噪声掩码的模型
protected_model = create_noisemask_model_tf(
    base_model,
    strategy='adaptive',
    noise_scale=0.1,
    noise_ratio=0.3
)

# 3. 编译和训练
protected_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## 🛡️ 噪声掩码策略

### 1. 静态噪声掩码 (Static)

在指定特征上添加固定强度的高斯噪声。

```python
config = NoiseMaskConfig(
    strategy=MaskStrategy.STATIC,
    noise_scale=0.1,
    noise_ratio=0.3
)
```

### 2. 自适应噪声掩码 (Adaptive)

根据样本置信度动态调整噪声强度。

```python
config = NoiseMaskConfig(
    strategy=MaskStrategy.ADAPTIVE,
    noise_scale=0.1,
    noise_ratio=0.3,
    adaptive_factor=1.5,  # 自适应调整因子
    temperature=1.0        # Softmax温度
)
```

### 3. 基于梯度的噪声掩码 (Gradient-based)

利用梯度信息识别敏感特征，在敏感特征上添加更强噪声。

```python
config = NoiseMaskConfig(
    strategy=MaskStrategy.GRADIENT_BASED,
    noise_scale=0.1,
    noise_ratio=0.3
)
```

### 4. 基于置信度的噪声掩码 (Confidence-based)

考虑置信度分布的熵，低熵（高确定性）样本添加更强噪声。

```python
config = NoiseMaskConfig(
    strategy=MaskStrategy.CONFIDENCE_BASED,
    noise_scale=0.1,
    noise_ratio=0.3,
    temperature=2.0
)
```

## 📊 评估防御效果

```python
from noisemask_pytorch import (
    MembershipInferenceEvaluator,
    compare_defense_effectiveness
)

# 评估单一攻击类型
evaluator = MembershipInferenceEvaluator(attack_type='confidence')
results = evaluator.evaluate_defense(
    protected_model,
    member_loader,      # 成员数据
    non_member_loader   # 非成员数据
)

print(f"Attack Accuracy: {results['accuracy']:.4f}")
print(f"AUC: {results['auc']:.4f}")
print(f"F1 Score: {results['f1']:.4f}")

# 对比原始模型和受保护模型
comparison = compare_defense_effectiveness(
    original_model,
    protected_model,
    member_loader,
    non_member_loader,
    attack_types=['confidence', 'loss', 'entropy']
)
```

## 🎯 高级用法

### 图像分类任务

```python
# 见 noisemask_examples.py 中的 example_image_classification()
```

### 大语言模型 (LLM)

```python
# 见 noisemask_examples.py 中的 integrate_with_transformer()
```

### 扩散模型

```python
# 见 noisemask_examples.py 中的 example_diffusion_model()
```

### 组合多种策略

```python
# 见 noisemask_examples.py 中的 example_combined_strategy()
```

### 对抗训练集成

```python
# 见 noisemask_examples.py 中的 example_adversarial_training()
```

## ⚙️ 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy` | MaskStrategy | ADAPTIVE | 噪声掩码策略 |
| `noise_scale` | float | 0.1 | 噪声强度/尺度 |
| `noise_ratio` | float | 0.3 | 被掩码的特征比例 |
| `temperature` | float | 1.0 | Softmax温度参数 |
| `adaptive_factor` | float | 1.0 | 自适应调整因子 |
| `min_noise` | float | 0.01 | 最小噪声强度 |
| `max_noise` | float | 1.0 | 最大噪声强度 |
| `decay_rate` | float | 0.99 | 噪声衰减率 |
| `target_layers` | List[int] | None | 目标层索引 |
| `preserve_accuracy` | bool | True | 优先保持准确率 |

## 📈 实验建议

### 1. 噪声强度调参

- 从 `noise_scale=0.05` 开始
- 逐步增加直到攻击准确率显著下降
- 监控模型正常任务的准确率

### 2. 策略选择

- **追求简单**：使用 `STATIC`
- **平衡效果**：使用 `ADAPTIVE`
- **最强防御**：使用 `CONFIDENCE_BASED`

### 3. 层选择

- 在较深层添加噪声通常效果更好
- 避免在输入层添加过强噪声
- 可以同时在多层添加噪声

## 🔬 评估指标

- **Attack Accuracy**: 攻击者判断成员/非成员的准确率（越低越好）
- **AUC**: ROC曲线下面积（越接近0.5越好）
- **F1 Score**: 攻击的F1分数（越低越好）
- **Model Accuracy**: 模型在正常任务上的准确率（越高越好）

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{noisemask_defense,
  title={Noise Mask Defense Against Membership Inference Attacks},
  author={AI Assistant},
  year={2026},
  note={PyTorch and TensorFlow implementations}
}
```

## 📚 相关论文

1. Shokri et al. "Membership Inference Attacks Against Machine Learning Models"
2. Salem et al. "ML-Leaks: Model and Data Independent Membership Inference Attacks"
3. Nasr et al. "Comprehensive Privacy Analysis of Deep Learning"

## 🤝 贡献

欢迎提交Issue和PR！

## 📄 许可证

MIT License
