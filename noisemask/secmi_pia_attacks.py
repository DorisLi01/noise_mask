"""
SecMI and PIA Attack Defense Evaluation
SecMI与PIA攻击防御效果评估

本模块实现了：
1. SecMI (Secret Membership Inference) 攻击
2. PIA (Population Inference Attack) 攻击
3. 评估Noise Mask对这两种攻击的防御效果

References:
- SecMI: "Membership Inference Attacks Against Machine Learning Models" (Shokri et al.)
- PIA: "Demystifying Membership Inference Attacks in Machine Learning as a Service"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings


# =============================================================================
# SecMI (Secret Membership Inference) 攻击实现
# =============================================================================

class SecMIAttack:
    """
    SecMI攻击实现
    
    SecMI是一种基于影子模型（Shadow Models）的成员推断攻击，
    攻击者通过训练多个影子模型来学习成员与非成员的统计差异。
    
    攻击信号：
    - 预测置信度 (Prediction Confidence)
    - 损失值 (Loss Value)
    - 预测修正 (Prediction Correctness)
    - 熵 (Entropy)
    """
    
    def __init__(self, num_classes: int, attack_feature: str = 'confidence'):
        """
        Args:
            num_classes: 类别数量
            attack_feature: 攻击特征类型 ('confidence', 'loss', 'entropy', 'correctness', 'all')
        """
        self.num_classes = num_classes
        self.attack_feature = attack_feature
        self.attack_model = None
        
    def extract_features(self, model: nn.Module, 
                         x: torch.Tensor, 
                         y: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        提取攻击特征
        
        Args:
            model: 目标模型
            x: 输入数据
            y: 真实标签（用于计算损失和正确性）
            
        Returns:
            特征数组 [N, feature_dim]
        """
        model.eval()
        features_list = []
        
        with torch.no_grad():
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)
            
            if self.attack_feature == 'confidence':
                # 最大置信度
                conf, _ = torch.max(probs, dim=1)
                features_list.append(conf.cpu().numpy().reshape(-1, 1))
                
            elif self.attack_feature == 'loss':
                # 负损失（需要标签）
                if y is None:
                    raise ValueError("Loss feature requires labels")
                loss = F.cross_entropy(outputs, y, reduction='none')
                features_list.append((-loss).cpu().numpy().reshape(-1, 1))
                
            elif self.attack_feature == 'entropy':
                # 预测分布的熵
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                features_list.append((-entropy).cpu().numpy().reshape(-1, 1))
                
            elif self.attack_feature == 'correctness':
                # 预测正确性
                if y is None:
                    raise ValueError("Correctness feature requires labels")
                pred = outputs.argmax(dim=1)
                correct = (pred == y).float()
                features_list.append(correct.cpu().numpy().reshape(-1, 1))
                
            elif self.attack_feature == 'all':
                # 组合所有特征
                conf, _ = torch.max(probs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                
                features = torch.stack([conf, -entropy], dim=1)
                
                if y is not None:
                    loss = F.cross_entropy(outputs, y, reduction='none')
                    pred = outputs.argmax(dim=1)
                    correct = (pred == y).float()
                    features = torch.cat([
                        features, 
                        (-loss).unsqueeze(1),
                        correct.unsqueeze(1)
                    ], dim=1)
                
                features_list.append(features.cpu().numpy())
        
        return np.concatenate(features_list, axis=1) if len(features_list) == 1 else features_list[0]
    
    def train_attack_model(self, 
                          shadow_model: nn.Module,
                          member_data: torch.utils.data.DataLoader,
                          non_member_data: torch.utils.data.DataLoader) -> nn.Module:
        """
        训练攻击模型（基于影子模型）
        
        Args:
            shadow_model: 影子模型（结构与目标模型相同）
            member_data: 成员数据
            non_member_data: 非成员数据
            
        Returns:
            训练好的攻击模型
        """
        # 提取特征
        X_member, y_member = [], []
        X_non_member, y_non_member = [], []
        
        for x, y in member_data:
            x, y = x.cuda() if torch.cuda.is_available() else x, y.cuda() if torch.cuda.is_available() else y
            feats = self.extract_features(shadow_model, x, y)
            X_member.append(feats)
            y_member.extend([1] * len(feats))
        
        for x, y in non_member_data:
            x, y = x.cuda() if torch.cuda.is_available() else x, y.cuda() if torch.cuda.is_available() else y
            feats = self.extract_features(shadow_model, x, y)
            X_non_member.append(feats)
            y_non_member.extend([0] * len(feats))
        
        X = np.vstack(X_member + X_non_member)
        y = np.array(y_member + y_non_member)
        
        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练简单的攻击模型（MLP或逻辑回归）
        from sklearn.neural_network import MLPClassifier
        
        self.attack_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        self.attack_model.fit(X_train, y_train)
        
        # 评估攻击模型
        train_acc = self.attack_model.score(X_train, y_train)
        test_acc = self.attack_model.score(X_test, y_test)
        
        print(f"  Attack Model - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        return self.attack_model
    
    def attack(self, 
               target_model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               true_labels: Optional[np.ndarray] = None) -> Dict:
        """
        执行SecMI攻击
        
        Args:
            target_model: 目标模型
            data_loader: 数据加载器
            true_labels: 真实成员标签（用于评估）
            
        Returns:
            攻击结果字典
        """
        if self.attack_model is None:
            raise ValueError("Attack model not trained. Call train_attack_model first.")
        
        # 提取特征
        X_test = []
        for x, y in data_loader:
            x, y = x.cuda() if torch.cuda.is_available() else x, y.cuda() if torch.cuda.is_available() else y
            feats = self.extract_features(target_model, x, y)
            X_test.append(feats)
        
        X_test = np.vstack(X_test)
        
        # 预测
        predictions = self.attack_model.predict(X_test)
        probabilities = self.attack_model.predict_proba(X_test)[:, 1]
        
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
        }
        
        if true_labels is not None:
            results['accuracy'] = accuracy_score(true_labels, predictions)
            results['precision'] = precision_score(true_labels, predictions, zero_division=0)
            results['recall'] = recall_score(true_labels, predictions, zero_division=0)
            results['f1'] = f1_score(true_labels, predictions, zero_division=0)
            results['auc'] = roc_auc_score(true_labels, probabilities) if len(set(true_labels)) > 1 else 0.5
        
        return results


# =============================================================================
# PIA (Population Inference Attack) 攻击实现
# =============================================================================

class PIAAttack:
    """
    PIA攻击实现
    
    PIA是一种基于群体统计的成员推断攻击，
    攻击者通过分析模型在目标数据集和参考数据集上的统计差异来推断成员关系。
    
    核心思想：
    - 成员数据：模型表现更"自信"（高置信度、低损失）
    - 非成员数据：模型表现相对"不确定"
    
    攻击信号：
    - 置信度分布差异
    - 损失分布差异
    """
    
    def __init__(self, num_classes: int, reference_size: int = 1000):
        """
        Args:
            num_classes: 类别数量
            reference_size: 参考数据集大小
        """
        self.num_classes = num_classes
        self.reference_size = reference_size
        self.threshold = None
        
    def compute_statistics(self, model: nn.Module, 
                          data_loader: torch.utils.data.DataLoader) -> Dict:
        """
        计算数据集上的统计信息
        
        Args:
            model: 目标模型
            data_loader: 数据加载器
            
        Returns:
            统计信息字典
        """
        model.eval()
        
        confidences = []
        losses = []
        entropies = []
        
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.cuda() if torch.cuda.is_available() else x, y.cuda() if torch.cuda.is_available() else y
                outputs = model(x)
                probs = F.softmax(outputs, dim=1)
                
                # 置信度
                conf, _ = torch.max(probs, dim=1)
                confidences.extend(conf.cpu().numpy())
                
                # 损失
                loss = F.cross_entropy(outputs, y, reduction='none')
                losses.extend(loss.cpu().numpy())
                
                # 熵
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                entropies.extend(entropy.cpu().numpy())
        
        return {
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'loss_mean': np.mean(losses),
            'loss_std': np.std(losses),
            'entropy_mean': np.mean(entropies),
            'entropy_std': np.std(entropies),
            'confidences': np.array(confidences),
            'losses': np.array(losses),
            'entropies': np.array(entropies)
        }
    
    def calibrate(self, 
                  model: nn.Module,
                  member_loader: torch.utils.data.DataLoader,
                  non_member_loader: torch.utils.data.DataLoader,
                  metric: str = 'confidence'):
        """
        校准攻击阈值
        
        Args:
            model: 目标模型
            member_loader: 成员数据
            non_member_loader: 非成员数据
            metric: 使用的指标 ('confidence', 'loss', 'entropy')
        """
        member_stats = self.compute_statistics(model, member_loader)
        non_member_stats = self.compute_statistics(model, non_member_loader)
        
        if metric == 'confidence':
            # 高置信度 -> 成员
            member_vals = member_stats['confidences']
            non_member_vals = non_member_stats['confidences']
            # 阈值设为两者均值的中点
            self.threshold = (member_vals.mean() + non_member_vals.mean()) / 2
            self.metric = 'confidence'
            self.higher_is_member = True
            
        elif metric == 'loss':
            # 低损失 -> 成员
            member_vals = member_stats['losses']
            non_member_vals = non_member_stats['losses']
            self.threshold = (member_vals.mean() + non_member_vals.mean()) / 2
            self.metric = 'loss'
            self.higher_is_member = False
            
        elif metric == 'entropy':
            # 低熵 -> 成员
            member_vals = member_stats['entropies']
            non_member_vals = non_member_stats['entropies']
            self.threshold = (member_vals.mean() + non_member_vals.mean()) / 2
            self.metric = 'entropy'
            self.higher_is_member = False
        
        print(f"  Calibrated threshold: {self.threshold:.4f}")
        print(f"  Member {self.metric}: {member_vals.mean():.4f} ± {member_vals.std():.4f}")
        print(f"  Non-member {self.metric}: {non_member_vals.mean():.4f} ± {non_member_vals.std():.4f}")
    
    def attack(self, 
               model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               true_labels: Optional[np.ndarray] = None) -> Dict:
        """
        执行PIA攻击
        
        Args:
            target_model: 目标模型
            data_loader: 数据加载器
            true_labels: 真实成员标签
            
        Returns:
            攻击结果字典
        """
        if self.threshold is None:
            raise ValueError("Threshold not calibrated. Call calibrate first.")
        
        # 计算统计信息
        stats = self.compute_statistics(model, data_loader)
        
        if self.metric == 'confidence':
            values = stats['confidences']
        elif self.metric == 'loss':
            values = stats['losses']
        else:
            values = stats['entropies']
        
        # 预测
        if self.higher_is_member:
            predictions = (values >= self.threshold).astype(int)
        else:
            predictions = (values <= self.threshold).astype(int)
        
        # 计算分数（归一化到0-1）
        if self.higher_is_member:
            scores = (values - values.min()) / (values.max() - values.min() + 1e-8)
        else:
            scores = 1 - (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        results = {
            'predictions': predictions,
            'scores': scores,
            'values': values,
            'threshold': self.threshold
        }
        
        if true_labels is not None:
            results['accuracy'] = accuracy_score(true_labels, predictions)
            results['precision'] = precision_score(true_labels, predictions, zero_division=0)
            results['recall'] = recall_score(true_labels, predictions, zero_division=0)
            results['f1'] = f1_score(true_labels, predictions, zero_division=0)
            results['auc'] = roc_auc_score(true_labels, scores) if len(set(true_labels)) > 1 else 0.5
        
        return results


# =============================================================================
# 防御效果评估器
# =============================================================================

class DefenseEvaluator:
    """
    Noise Mask防御效果评估器
    
    评估Noise Mask对SecMI和PIA攻击的防御效果
    """
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.results = {}
    
    def evaluate_secmi_defense(self,
                               original_model: nn.Module,
                               protected_model: nn.Module,
                               shadow_model: nn.Module,
                               member_loader: torch.utils.data.DataLoader,
                               non_member_loader: torch.utils.data.DataLoader,
                               attack_features: List[str] = None) -> Dict:
        """
        评估Noise Mask对SecMI攻击的防御效果
        
        Args:
            original_model: 原始模型
            protected_model: 带Noise Mask的模型
            shadow_model: 影子模型
            member_loader: 成员数据
            non_member_loader: 非成员数据
            attack_features: 攻击特征列表
            
        Returns:
            评估结果字典
        """
        attack_features = attack_features or ['confidence', 'loss', 'entropy', 'all']
        results = {}
        
        print("\n" + "=" * 70)
        print("SecMI Attack Defense Evaluation")
        print("=" * 70)
        
        for feature in attack_features:
            print(f"\n--- Attack Feature: {feature.upper()} ---")
            
            # 原始模型
            print("  [Original Model]")
            secmi_orig = SecMIAttack(self.num_classes, feature)
            secmi_orig.train_attack_model(shadow_model, member_loader, non_member_loader)
            
            # 创建测试数据
            test_data = []
            test_labels = []
            for x, y in member_loader:
                test_data.append((x, y))
                test_labels.extend([1] * len(x))
            for x, y in non_member_loader:
                test_data.append((x, y))
                test_labels.extend([0] * len(x))
            
            # 创建DataLoader
            all_x = torch.cat([x for x, y in test_data])
            all_y = torch.cat([y for x, y in test_data])
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(all_x, all_y),
                batch_size=64, shuffle=False
            )
            
            attack_results_orig = secmi_orig.attack(
                original_model, test_loader, np.array(test_labels)
            )
            
            print(f"    Attack Acc: {attack_results_orig['accuracy']:.4f}")
            print(f"    AUC: {attack_results_orig['auc']:.4f}")
            print(f"    F1: {attack_results_orig['f1']:.4f}")
            
            # 受保护模型
            print("  [Protected Model]")
            secmi_prot = SecMIAttack(self.num_classes, feature)
            secmi_prot.train_attack_model(shadow_model, member_loader, non_member_loader)
            
            attack_results_prot = secmi_prot.attack(
                protected_model, test_loader, np.array(test_labels)
            )
            
            print(f"    Attack Acc: {attack_results_prot['accuracy']:.4f}")
            print(f"    AUC: {attack_results_prot['auc']:.4f}")
            print(f"    F1: {attack_results_prot['f1']:.4f}")
            
            # 计算改进
            improvement = {
                'accuracy_drop': attack_results_orig['accuracy'] - attack_results_prot['accuracy'],
                'auc_drop': attack_results_orig['auc'] - attack_results_prot['auc'],
                'f1_drop': attack_results_orig['f1'] - attack_results_prot['f1'],
            }
            
            print(f"  [Improvement]")
            print(f"    Acc Drop: {improvement['accuracy_drop']:.4f}")
            print(f"    AUC Drop: {improvement['auc_drop']:.4f}")
            print(f"    F1 Drop: {improvement['f1_drop']:.4f}")
            
            results[feature] = {
                'original': attack_results_orig,
                'protected': attack_results_prot,
                'improvement': improvement
            }
        
        self.results['secmi'] = results
        return results
    
    def evaluate_pia_defense(self,
                            original_model: nn.Module,
                            protected_model: nn.Module,
                            member_loader: torch.utils.data.DataLoader,
                            non_member_loader: torch.utils.data.DataLoader,
                            metrics: List[str] = None) -> Dict:
        """
        评估Noise Mask对PIA攻击的防御效果
        
        Args:
            original_model: 原始模型
            protected_model: 带Noise Mask的模型
            member_loader: 成员数据
            non_member_loader: 非成员数据
            metrics: 攻击指标列表
            
        Returns:
            评估结果字典
        """
        metrics = metrics or ['confidence', 'loss', 'entropy']
        results = {}
        
        print("\n" + "=" * 70)
        print("PIA Attack Defense Evaluation")
        print("=" * 70)
        
        for metric in metrics:
            print(f"\n--- Attack Metric: {metric.upper()} ---")
            
            # 原始模型
            print("  [Original Model]")
            pia_orig = PIAAttack(self.num_classes)
            pia_orig.calibrate(original_model, member_loader, non_member_loader, metric)
            
            # 测试数据
            test_data = []
            test_labels = []
            for x, y in member_loader:
                test_data.append((x, y))
                test_labels.extend([1] * len(x))
            for x, y in non_member_loader:
                test_data.append((x, y))
                test_labels.extend([0] * len(x))
            
            all_x = torch.cat([x for x, y in test_data])
            all_y = torch.cat([y for x, y in test_data])
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(all_x, all_y),
                batch_size=64, shuffle=False
            )
            
            attack_results_orig = pia_orig.attack(
                original_model, test_loader, np.array(test_labels)
            )
            
            print(f"    Attack Acc: {attack_results_orig['accuracy']:.4f}")
            print(f"    AUC: {attack_results_orig['auc']:.4f}")
            print(f"    F1: {attack_results_orig['f1']:.4f}")
            
            # 受保护模型
            print("  [Protected Model]")
            pia_prot = PIAAttack(self.num_classes)
            pia_prot.calibrate(protected_model, member_loader, non_member_loader, metric)
            
            attack_results_prot = pia_prot.attack(
                protected_model, test_loader, np.array(test_labels)
            )
            
            print(f"    Attack Acc: {attack_results_prot['accuracy']:.4f}")
            print(f"    AUC: {attack_results_prot['auc']:.4f}")
            print(f"    F1: {attack_results_prot['f1']:.4f}")
            
            # 计算改进
            improvement = {
                'accuracy_drop': attack_results_orig['accuracy'] - attack_results_prot['accuracy'],
                'auc_drop': attack_results_orig['auc'] - attack_results_prot['auc'],
                'f1_drop': attack_results_orig['f1'] - attack_results_prot['f1'],
            }
            
            print(f"  [Improvement]")
            print(f"    Acc Drop: {improvement['accuracy_drop']:.4f}")
            print(f"    AUC Drop: {improvement['auc_drop']:.4f}")
            print(f"    F1 Drop: {improvement['f1_drop']:.4f}")
            
            results[metric] = {
                'original': attack_results_orig,
                'protected': attack_results_prot,
                'improvement': improvement
            }
        
        self.results['pia'] = results
        return results
    
    def generate_report(self, save_path: str = 'defense_report.txt'):
        """生成评估报告"""
        with open(save_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Noise Mask Defense Evaluation Report\n")
            f.write("=" * 70 + "\n\n")
            
            if 'secmi' in self.results:
                f.write("SecMI Attack Results:\n")
                f.write("-" * 70 + "\n")
                for feature, res in self.results['secmi'].items():
                    f.write(f"\nFeature: {feature}\n")
                    f.write(f"  Original  - Acc: {res['original']['accuracy']:.4f}, "
                           f"AUC: {res['original']['auc']:.4f}\n")
                    f.write(f"  Protected - Acc: {res['protected']['accuracy']:.4f}, "
                           f"AUC: {res['protected']['auc']:.4f}\n")
                    f.write(f"  Improvement - Acc↓: {res['improvement']['accuracy_drop']:.4f}, "
                           f"AUC↓: {res['improvement']['auc_drop']:.4f}\n")
            
            if 'pia' in self.results:
                f.write("\n\nPIA Attack Results:\n")
                f.write("-" * 70 + "\n")
                for metric, res in self.results['pia'].items():
                    f.write(f"\nMetric: {metric}\n")
                    f.write(f"  Original  - Acc: {res['original']['accuracy']:.4f}, "
                           f"AUC: {res['original']['auc']:.4f}\n")
                    f.write(f"  Protected - Acc: {res['protected']['accuracy']:.4f}, "
                           f"AUC: {res['protected']['auc']:.4f}\n")
                    f.write(f"  Improvement - Acc↓: {res['improvement']['accuracy_drop']:.4f}, "
                           f"AUC↓: {res['improvement']['auc_drop']:.4f}\n")
        
        print(f"\nReport saved to {save_path}")


# =============================================================================
# 快速测试函数
# =============================================================================

def quick_test():
    """快速测试SecMI和PIA攻击"""
    print("=" * 70)
    print("SecMI & PIA Attack Quick Test")
    print("=" * 70)
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(50, 100)
            self.fc2 = nn.Linear(100, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    
    # 生成测试数据
    member_x = torch.randn(200, 50)
    member_y = torch.randint(0, 10, (200,))
    non_member_x = torch.randn(200, 50)
    non_member_y = torch.randint(0, 10, (200,))
    
    member_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(member_x, member_y),
        batch_size=32, shuffle=True
    )
    non_member_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(non_member_x, non_member_y),
        batch_size=32, shuffle=False
    )
    
    # 创建模型
    model = SimpleModel()
    shadow_model = SimpleModel()
    
    print("\nModels created successfully!")
    print(f"  Member samples: {len(member_x)}")
    print(f"  Non-member samples: {len(non_member_x)}")
    
    # 测试SecMI
    print("\n" + "-" * 70)
    print("Testing SecMI Attack...")
    print("-" * 70)
    
    secmi = SecMIAttack(num_classes=10, attack_feature='confidence')
    secmi.train_attack_model(shadow_model, member_loader, non_member_loader)
    
    # 测试PIA
    print("\n" + "-" * 70)
    print("Testing PIA Attack...")
    print("-" * 70)
    
    pia = PIAAttack(num_classes=10)
    pia.calibrate(model, member_loader, non_member_loader, 'confidence')
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    quick_test()
