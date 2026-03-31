# Noise Mask 噪声掩码防御成员推断攻击 - 实验脚本
# 用于快速验证防御效果

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys
sys.path.insert(0, '.')

from noisemask_pytorch import (
    NoiseMaskConfig, MaskStrategy, NoiseMaskWrapper,
    MembershipInferenceEvaluator, compare_defense_effectiveness,
    create_noisemask_model, StaticNoiseMask, AdaptiveNoiseMask,
    ConfidenceBasedNoiseMask, GradientBasedNoiseMask
)


# =============================================================================
# 配置
# =============================================================================

class Config:
    """实验配置"""
    # 数据配置
    NUM_SAMPLES = 1000
    BATCH_SIZE = 64
    NUM_CLASSES = 10
    INPUT_DIM = 100
    
    # 模型配置
    HIDDEN_DIMS = [256, 128, 64]
    
    # 训练配置
    EPOCHS = 10
    LR = 0.001
    
    # 噪声掩码配置
    NOISE_SCALE = 0.15
    NOISE_RATIO = 0.3
    ADAPTIVE_FACTOR = 1.5
    
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 数据生成
# =============================================================================

def generate_synthetic_data(config: Config):
    """
    生成合成数据用于实验
    
    成员数据：从特定分布采样（模型会学习到的分布）
    非成员数据：从不同分布采样
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 成员数据 - 有特定模式
    member_X = []
    member_y = []
    for class_id in range(config.NUM_CLASSES):
        # 每个类有特定的均值
        mean = np.random.randn(config.INPUT_DIM) * 0.5 + class_id
        cov = np.eye(config.INPUT_DIM) * 0.1
        class_samples = np.random.multivariate_normal(
            mean, cov, config.NUM_SAMPLES // config.NUM_CLASSES
        )
        member_X.append(class_samples)
        member_y.extend([class_id] * len(class_samples))
    
    member_X = np.vstack(member_X).astype(np.float32)
    member_y = np.array(member_y)
    
    # 非成员数据 - 不同分布
    non_member_X = []
    non_member_y = []
    for class_id in range(config.NUM_CLASSES):
        # 偏移的均值
        mean = np.random.randn(config.INPUT_DIM) * 0.5 + class_id + 0.5
        cov = np.eye(config.INPUT_DIM) * 0.15
        class_samples = np.random.multivariate_normal(
            mean, cov, config.NUM_SAMPLES // config.NUM_CLASSES
        )
        non_member_X.append(class_samples)
        non_member_y.extend([class_id] * len(class_samples))
    
    non_member_X = np.vstack(non_member_X).astype(np.float32)
    non_member_y = np.array(non_member_y)
    
    # 创建DataLoader
    member_dataset = TensorDataset(
        torch.from_numpy(member_X),
        torch.from_numpy(member_y)
    )
    non_member_dataset = TensorDataset(
        torch.from_numpy(non_member_X),
        torch.from_numpy(non_member_y)
    )
    
    member_loader = DataLoader(
        member_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    non_member_loader = DataLoader(
        non_member_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    
    # 测试数据
    test_X = torch.from_numpy(member_X[:200])
    test_y = torch.from_numpy(member_y[:200])
    test_loader = DataLoader(
        TensorDataset(test_X, test_y),
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    return member_loader, non_member_loader, test_loader


# =============================================================================
# 模型定义
# =============================================================================

class MLPClassifier(nn.Module):
    """简单的MLP分类器"""
    
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# =============================================================================
# 训练函数
# =============================================================================

def train_model(model, train_loader, config: Config, verbose=True):
    """训练模型"""
    model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    
    model.train()
    for epoch in range(config.EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
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
        if verbose and (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{config.EPOCHS}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
    
    return model


def evaluate_accuracy(model, data_loader, config: Config):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


# =============================================================================
# 可视化函数
# =============================================================================

def plot_roc_curves(results_dict, save_path='roc_curves.png'):
    """绘制ROC曲线对比"""
    plt.figure(figsize=(10, 8))
    
    colors = {
        'original': 'red',
        'static': 'blue',
        'adaptive': 'green',
        'confidence': 'purple',
        'gradient': 'orange'
    }
    
    for name, results in results_dict.items():
        # 重构攻击分数和标签
        member_scores = np.random.normal(
            results['member_score_mean'],
            results['member_score_std'],
            500
        )
        non_member_scores = np.random.normal(
            results['non_member_score_mean'],
            results['non_member_score_std'],
            500
        )
        
        y_true = np.concatenate([np.ones(500), np.zeros(500)])
        y_scores = np.concatenate([member_scores, non_member_scores])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors.get(name, 'gray'),
                label=f'{name} (AUC = {roc_auc:.3f})',
                linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Membership Inference Attack', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nROC curves saved to {save_path}")
    plt.close()


def plot_defense_comparison(comparison_results, save_path='defense_comparison.png'):
    """绘制防御效果对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['accuracy', 'auc', 'f1']
    titles = ['Attack Accuracy', 'AUC', 'F1 Score']
    
    strategies = list(comparison_results.keys())
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        original_vals = [comparison_results[s]['original'][metric] for s in strategies]
        protected_vals = [comparison_results[s]['protected'][metric] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        ax.bar(x - width/2, original_vals, width, label='Original', color='red', alpha=0.7)
        ax.bar(x + width/2, protected_vals, width, label='Protected', color='green', alpha=0.7)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Defense comparison saved to {save_path}")
    plt.close()


# =============================================================================
# 主实验
# =============================================================================

def run_experiment():
    """运行完整实验"""
    print("=" * 70)
    print("Noise Mask Defense - Membership Inference Attack Experiment")
    print("=" * 70)
    
    config = Config()
    print(f"\nDevice: {config.DEVICE}")
    
    # 1. 生成数据
    print("\n" + "-" * 70)
    print("Step 1: Generating synthetic data...")
    print("-" * 70)
    member_loader, non_member_loader, test_loader = generate_synthetic_data(config)
    print(f"  Member samples: {len(member_loader.dataset)}")
    print(f"  Non-member samples: {len(non_member_loader.dataset)}")
    
    # 2. 训练原始模型
    print("\n" + "-" * 70)
    print("Step 2: Training original model...")
    print("-" * 70)
    original_model = MLPClassifier(
        config.INPUT_DIM, config.HIDDEN_DIMS, config.NUM_CLASSES
    )
    original_model = train_model(original_model, member_loader, config)
    
    acc = evaluate_accuracy(original_model, test_loader, config)
    print(f"  Test Accuracy: {acc:.2f}%")
    
    # 3. 评估原始模型的MIA脆弱性
    print("\n" + "-" * 70)
    print("Step 3: Evaluating MIA vulnerability (Original Model)...")
    print("-" * 70)
    
    evaluator = MembershipInferenceEvaluator(attack_type='confidence')
    original_results = evaluator.evaluate_defense(
        original_model, member_loader, non_member_loader
    )
    
    print(f"  Attack Accuracy: {original_results['accuracy']:.4f}")
    print(f"  AUC: {original_results['auc']:.4f}")
    print(f"  F1 Score: {original_results['f1']:.4f}")
    print(f"  Member score: {original_results['member_score_mean']:.4f} ± {original_results['member_score_std']:.4f}")
    print(f"  Non-member score: {original_results['non_member_score_mean']:.4f} ± {original_results['non_member_score_std']:.4f}")
    
    # 4. 测试不同噪声掩码策略
    print("\n" + "-" * 70)
    print("Step 4: Testing different noise mask strategies...")
    print("-" * 70)
    
    strategies = [
        ('static', MaskStrategy.STATIC),
        ('adaptive', MaskStrategy.ADAPTIVE),
        ('confidence', MaskStrategy.CONFIDENCE_BASED),
    ]
    
    all_results = {'original': original_results}
    comparison_results = {}
    
    for name, strategy in strategies:
        print(f"\n  Strategy: {name.upper()}")
        
        # 创建带噪声掩码的模型
        noise_config = NoiseMaskConfig(
            strategy=strategy,
            noise_scale=config.NOISE_SCALE,
            noise_ratio=config.NOISE_RATIO,
            adaptive_factor=config.ADAPTIVE_FACTOR,
            preserve_accuracy=True
        )
        
        # 重新初始化模型
        protected_model = MLPClassifier(
            config.INPUT_DIM, config.HIDDEN_DIMS, config.NUM_CLASSES
        )
        protected_model.load_state_dict(original_model.state_dict())
        protected_model = NoiseMaskWrapper(protected_model, noise_config).to(config.DEVICE)
        
        # 评估防御效果
        protected_results = evaluator.evaluate_defense(
            protected_model, member_loader, non_member_loader
        )
        
        all_results[name] = protected_results
        
        print(f"    Attack Accuracy: {protected_results['accuracy']:.4f} (↓{original_results['accuracy']-protected_results['accuracy']:.4f})")
        print(f"    AUC: {protected_results['auc']:.4f} (↓{original_results['auc']-protected_results['auc']:.4f})")
        print(f"    F1 Score: {protected_results['f1']:.4f} (↓{original_results['f1']-protected_results['f1']:.4f})")
        
        # 测试准确率
        acc_protected = evaluate_accuracy(protected_model, test_loader, config)
        print(f"    Test Accuracy: {acc_protected:.2f}% (vs {acc:.2f}%)")
        
        # 保存对比结果
        comparison_results[name] = {
            'original': original_results,
            'protected': protected_results,
            'improvement': {
                'accuracy_drop': original_results['accuracy'] - protected_results['accuracy'],
                'auc_drop': original_results['auc'] - protected_results['auc'],
                'f1_drop': original_results['f1'] - protected_results['f1'],
            }
        }
    
    # 5. 可视化结果
    print("\n" + "-" * 70)
    print("Step 5: Generating visualizations...")
    print("-" * 70)
    
    try:
        plot_roc_curves(all_results, 'roc_curves.png')
        plot_defense_comparison(comparison_results, 'defense_comparison.png')
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")
    
    # 6. 总结
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print("\nDefense Effectiveness (Attack Accuracy Reduction):")
    for name, results in comparison_results.items():
        drop = results['improvement']['accuracy_drop']
        auc_drop = results['improvement']['auc_drop']
        print(f"  {name:12s}: Acc ↓{drop:.4f}, AUC ↓{auc_drop:.4f}")
    
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print("=" * 70)
    
    return comparison_results


# =============================================================================
# 参数扫描实验
# =============================================================================

def noise_scale_sweep():
    """
    噪声强度参数扫描实验
    
    测试不同噪声强度对防御效果和模型准确率的影响
    """
    print("\n" + "=" * 70)
    print("Noise Scale Sweep Experiment")
    print("=" * 70)
    
    config = Config()
    config.EPOCHS = 5  # 减少训练轮数
    
    # 生成数据
    member_loader, non_member_loader, test_loader = generate_synthetic_data(config)
    
    # 训练基础模型
    print("\nTraining base model...")
    base_model = MLPClassifier(config.INPUT_DIM, config.HIDDEN_DIMS, config.NUM_CLASSES)
    base_model = train_model(base_model, member_loader, config, verbose=False)
    base_acc = evaluate_accuracy(base_model, test_loader, config)
    
    # 测试不同噪声强度
    noise_scales = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    results = []
    
    evaluator = MembershipInferenceEvaluator(attack_type='confidence')
    
    for scale in noise_scales:
        print(f"\nTesting noise_scale={scale}...")
        
        # 创建带噪声的模型
        noise_config = NoiseMaskConfig(
            strategy=MaskStrategy.ADAPTIVE,
            noise_scale=scale,
            noise_ratio=0.3,
            adaptive_factor=1.5
        )
        
        protected_model = MLPClassifier(
            config.INPUT_DIM, config.HIDDEN_DIMS, config.NUM_CLASSES
        )
        protected_model.load_state_dict(base_model.state_dict())
        protected_model = NoiseMaskWrapper(protected_model, noise_config).to(config.DEVICE)
        
        # 评估
        mia_results = evaluator.evaluate_defense(
            protected_model, member_loader, non_member_loader
        )
        acc = evaluate_accuracy(protected_model, test_loader, config)
        
        results.append({
            'noise_scale': scale,
            'attack_acc': mia_results['accuracy'],
            'auc': mia_results['auc'],
            'model_acc': acc
        })
        
        print(f"  Attack Acc: {mia_results['accuracy']:.4f}, Model Acc: {acc:.2f}%")
    
    # 打印结果表格
    print("\n" + "-" * 70)
    print("Results Summary:")
    print("-" * 70)
    print(f"{'Noise Scale':<12} {'Attack Acc':<12} {'AUC':<12} {'Model Acc':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['noise_scale']:<12.2f} {r['attack_acc']:<12.4f} {r['auc']:<12.4f} {r['model_acc']:<12.2f}")
    
    # 绘制结果
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        scales = [r['noise_scale'] for r in results]
        attack_accs = [r['attack_acc'] for r in results]
        model_accs = [r['model_acc'] for r in results]
        
        axes[0].plot(scales, attack_accs, 'o-', color='red', linewidth=2, markersize=8)
        axes[0].set_xlabel('Noise Scale', fontsize=12)
        axes[0].set_ylabel('Attack Accuracy', fontsize=12)
        axes[0].set_title('MIA Accuracy vs Noise Scale', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0.5, color='gray', linestyle='--', label='Random Guess')
        axes[0].legend()
        
        axes[1].plot(scales, model_accs, 'o-', color='blue', linewidth=2, markersize=8)
        axes[1].axhline(y=base_acc, color='gray', linestyle='--', label=f'Baseline ({base_acc:.1f}%)')
        axes[1].set_xlabel('Noise Scale', fontsize=12)
        axes[1].set_ylabel('Model Accuracy (%)', fontsize=12)
        axes[1].set_title('Model Accuracy vs Noise Scale', fontsize=13)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('noise_scale_sweep.png', dpi=150)
        print("\nPlot saved to noise_scale_sweep.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate plot: {e}")
    
    return results


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Noise Mask Defense Experiment')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'sweep'],
                       help='Experiment mode')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_experiment()
    elif args.mode == 'sweep':
        noise_scale_sweep()
