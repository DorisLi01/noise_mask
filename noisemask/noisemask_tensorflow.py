"""
Noise Mask Defense Against Membership Inference Attacks (TensorFlow Version)
噪声掩码防御成员推断攻击 - TensorFlow实现

本模块实现了多种噪声掩码策略的TensorFlow版本，用于防御针对深度学习模型的成员推断攻击(MIA)。

Author: AI Assistant
Date: 2026-03-24
"""

import tensorflow as tf
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
        target_layers: 目标层名称列表，None表示自动选择 (默认: None)
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
    target_layers: Optional[List[str]] = None
    preserve_accuracy: bool = True


# =============================================================================
# TensorFlow噪声掩码层
# =============================================================================

class StaticNoiseMaskLayer(tf.keras.layers.Layer):
    """
    静态噪声掩码层 (TensorFlow)
    
    在指定特征上添加固定强度的高斯噪声。
    """
    
    def __init__(self, config: NoiseMaskConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.current_step = tf.Variable(0, trainable=False, dtype=tf.int32)
    
    def call(self, inputs, training=None):
        """
        应用静态高斯噪声掩码
        
        Args:
            inputs: 输入特征
            training: 是否为训练模式
            
        Returns:
            加噪后的特征
        """
        x = inputs
        
        # 推理时可选降低噪声
        if not training and self.config.preserve_accuracy:
            noise_scale = self.config.noise_scale * 0.5
        else:
            noise_scale = self.config.noise_scale
        
        # 生成掩码：随机选择要加噪的位置
        mask = tf.random.uniform(tf.shape(x)) < self.config.noise_ratio
        
        # 生成高斯噪声
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_scale)
        
        # 应用掩码噪声
        masked_x = tf.where(mask, x + noise, x)
        
        return masked_x
    
    def step(self):
        """更新步数"""
        self.current_step.assign_add(1)


class AdaptiveNoiseMaskLayer(tf.keras.layers.Layer):
    """
    自适应噪声掩码层 (TensorFlow)
    
    根据样本的置信度动态调整噪声强度。
    """
    
    def __init__(self, config: NoiseMaskConfig, num_classes: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_classes = num_classes
        self.current_step = tf.Variable(0, trainable=False, dtype=tf.int32)
    
    def call(self, inputs, training=None):
        """
        应用自适应噪声掩码
        
        注意：此层需要额外的置信度信息，通过compute_adaptive_noise方法使用
        """
        x = inputs
        
        # 基础噪声（当没有置信度信息时）
        noise_scale = self.config.noise_scale
        mask = tf.random.uniform(tf.shape(x)) < self.config.noise_ratio
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_scale)
        
        return tf.where(mask, x + noise, x)
    
    def compute_adaptive_noise(self, x, predictions):
        """
        计算自适应噪声（基于预测置信度）
        
        Args:
            x: 输入特征
            predictions: 模型预测输出 (logits或probabilities)
            
        Returns:
            加噪后的特征
        """
        # 计算概率和置信度
        probs = tf.nn.softmax(predictions / self.config.temperature, axis=1)
        confidence = tf.reduce_max(probs, axis=1)  # [batch_size]
        
        # 扩展维度以便广播
        confidence = tf.expand_dims(confidence, axis=-1)
        for _ in range(len(x.shape) - 2):
            confidence = tf.expand_dims(confidence, axis=-1)
        
        # 自适应噪声强度
        scale = self.config.noise_scale * (1 + self.config.adaptive_factor * confidence)
        scale = tf.minimum(scale, self.config.max_noise)
        
        # 生成噪声
        mask = tf.random.uniform(tf.shape(x)) < self.config.noise_ratio
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=scale)
        
        return tf.where(mask, x + noise, x)


class ConfidenceBasedNoiseMaskLayer(tf.keras.layers.Layer):
    """
    基于置信度分布熵的噪声掩码层 (TensorFlow)
    """
    
    def __init__(self, config: NoiseMaskConfig, num_classes: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_classes = num_classes
    
    def call(self, inputs, training=None):
        """基础调用（需要配合predictions使用compute_entropy_noise）"""
        return inputs
    
    def compute_entropy_noise(self, x, predictions):
        """
        基于熵计算噪声强度
        
        低熵（高确定性）-> 高噪声
        高熵（低确定性）-> 低噪声
        """
        # 计算概率分布
        probs = tf.nn.softmax(predictions / self.config.temperature, axis=1)
        
        # 计算熵: H = -sum(p * log(p))
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
        
        # 归一化熵
        max_entropy = np.log(self.num_classes)
        entropy_normalized = entropy / max_entropy
        
        # 确定性 = 1 - 归一化熵
        certainty = 1 - entropy_normalized
        
        # 扩展维度
        certainty = tf.expand_dims(certainty, axis=-1)
        for _ in range(len(x.shape) - 2):
            certainty = tf.expand_dims(certainty, axis=-1)
        
        # 噪声强度
        scale = self.config.noise_scale * (1 + self.config.adaptive_factor * certainty)
        scale = tf.minimum(scale, self.config.max_noise)
        
        # 生成噪声
        mask = tf.random.uniform(tf.shape(x)) < self.config.noise_ratio
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=scale)
        
        return tf.where(mask, x + noise, x)


# =============================================================================
# 噪声掩码模型包装器
# =============================================================================

class NoiseMaskModel(tf.keras.Model):
    """
    带噪声掩码保护的Keras模型
    
    将噪声掩码集成到现有Keras模型中。
    
    Usage:
        base_model = create_base_model()
        config = NoiseMaskConfig(strategy=MaskStrategy.ADAPTIVE, noise_scale=0.1)
        protected_model = NoiseMaskModel(base_model, config)
    """
    
    def __init__(self, 
                 base_model: tf.keras.Model,
                 config: NoiseMaskConfig,
                 num_classes: int = 10,
                 mask_positions: Optional[List[str]] = None):
        """
        Args:
            base_model: 基础模型
            config: 噪声掩码配置
            num_classes: 类别数量
            mask_positions: 要插入掩码的位置
        """
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.num_classes = num_classes
        self.mask_positions = mask_positions or []
        
        # 创建噪声掩码层
        self.noise_mask_layers = self._create_noise_layers()
        
        # 用于存储中间激活
        self.intermediate_activations = {}
    
    def _create_noise_layers(self) -> dict:
        """创建噪声掩码层字典"""
        layers = {}
        
        if self.config.strategy == MaskStrategy.STATIC:
            layer_class = StaticNoiseMaskLayer
        elif self.config.strategy == MaskStrategy.ADAPTIVE:
            layer_class = AdaptiveNoiseMaskLayer
        elif self.config.strategy == MaskStrategy.CONFIDENCE_BASED:
            layer_class = ConfidenceBasedNoiseMaskLayer
        else:
            layer_class = StaticNoiseMaskLayer
        
        # 为每个目标位置创建层
        if self.mask_positions:
            for pos in self.mask_positions:
                layers[pos] = layer_class(self.config, num_classes=self.num_classes)
        else:
            # 默认创建一个
            layers['default'] = layer_class(self.config, num_classes=self.num_classes)
        
        return layers
    
    def call(self, inputs, training=None, return_clean=False):
        """
        前向传播
        
        Args:
            inputs: 输入数据
            training: 是否为训练模式
            return_clean: 是否同时返回干净输出
            
        Returns:
            模型输出
        """
        # 获取干净输出（用于对比）
        if return_clean:
            clean_output = self.base_model(inputs, training=False)
        
        # 正常前向传播
        x = inputs
        
        # 简单实现：在输入层添加噪声（更复杂的实现需要在模型架构中插入层）
        if 'input' in self.noise_mask_layers:
            x = self.noise_mask_layers['input'](x, training=training)
        
        # 通过基础模型
        output = self.base_model(x, training=training)
        
        # 如果配置了自适应噪声，基于输出调整
        if self.config.strategy in [MaskStrategy.ADAPTIVE, MaskStrategy.CONFIDENCE_BASED]:
            # 这里简化处理，实际应该在中间层应用
            pass
        
        if return_clean:
            return output, clean_output
        return output
    
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=None, **kwargs):
        """编译模型"""
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss_fn = loss
        self.train_metrics = metrics or ['accuracy']
    
    def train_step(self, data):
        """自定义训练步骤"""
        x, y = data
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # 更新指标
        self.compiled_metrics.update_state(y, predictions)
        
        return {m.name: m.result() for m in self.metrics}


# =============================================================================
# 成员推断攻击评估 (TensorFlow版)
# =============================================================================

class MembershipInferenceEvaluatorTF:
    """
    成员推断攻击评估器 (TensorFlow版本)
    """
    
    def __init__(self, attack_type: str = "confidence"):
        """
        Args:
            attack_type: 攻击类型 ('confidence', 'loss', 'entropy')
        """
        self.attack_type = attack_type
    
    def compute_attack_score(self, 
                            model: tf.keras.Model,
                            x: np.ndarray,
                            y: np.ndarray,
                            batch_size: int = 32) -> np.ndarray:
        """
        计算攻击分数
        
        Args:
            model: 目标模型
            x: 输入数据
            y: 标签
            batch_size: 批次大小
            
        Returns:
            攻击分数数组
        """
        scores = []
        
        # 批量预测
        predictions = model.predict(x, batch_size=batch_size, verbose=0)
        probs = tf.nn.softmax(predictions, axis=1).numpy()
        
        if self.attack_type == "confidence":
            # 最大置信度
            scores = np.max(probs, axis=1)
            
        elif self.attack_type == "loss":
            # 负交叉熵损失
            for i in range(len(x)):
                loss = -tf.keras.losses.sparse_categorical_crossentropy(
                    [y[i]], [predictions[i]]
                ).numpy()[0]
                scores.append(loss)
                
        elif self.attack_type == "entropy":
            # 负熵
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            scores = -entropy
        
        return np.array(scores)
    
    def evaluate_defense(self,
                        model: tf.keras.Model,
                        x_member: np.ndarray,
                        y_member: np.ndarray,
                        x_non_member: np.ndarray,
                        y_non_member: np.ndarray,
                        threshold: Optional[float] = None) -> dict:
        """
        评估防御效果
        
        Args:
            model: 被评估模型
            x_member: 成员数据
            y_member: 成员标签
            x_non_member: 非成员数据
            y_non_member: 非成员标签
            threshold: 分类阈值
            
        Returns:
            评估指标字典
        """
        # 计算分数
        member_scores = self.compute_attack_score(model, x_member, y_member)
        non_member_scores = self.compute_attack_score(model, x_non_member, y_non_member)
        
        # 创建标签
        y_true = np.concatenate([
            np.ones(len(member_scores)),
            np.zeros(len(non_member_scores))
        ])
        y_scores = np.concatenate([member_scores, non_member_scores])
        
        # 自动确定阈值
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
# 辅助函数
# =============================================================================

def create_noisemask_model_tf(base_model: tf.keras.Model,
                               strategy: str = "adaptive",
                               noise_scale: float = 0.1,
                               noise_ratio: float = 0.3,
                               num_classes: int = 10,
                               **kwargs) -> NoiseMaskModel:
    """
    便捷函数：快速创建带噪声掩码保护的TensorFlow模型
    
    Args:
        base_model: 原始模型
        strategy: 掩码策略
        noise_scale: 噪声强度
        noise_ratio: 噪声比例
        num_classes: 类别数
        **kwargs: 其他配置
        
    Returns:
        包装后的模型
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
    
    return NoiseMaskModel(base_model, config, num_classes=num_classes)


def insert_noise_layers_tf(model: tf.keras.Model,
                            config: NoiseMaskConfig,
                            layer_indices: List[int] = None) -> tf.keras.Model:
    """
    在现有Keras模型中插入噪声掩码层
    
    Args:
        model: 原始模型
        config: 噪声掩码配置
        layer_indices: 要插入噪声的层索引
        
    Returns:
        新模型
    """
    # 获取所有层
    layers = model.layers.copy()
    
    # 创建新层列表
    new_layers = []
    
    for i, layer in enumerate(layers):
        new_layers.append(layer)
        
        # 在指定层后插入噪声层
        if layer_indices and i in layer_indices:
            if config.strategy == MaskStrategy.STATIC:
                noise_layer = StaticNoiseMaskLayer(config, name=f'noise_mask_{i}')
            else:
                noise_layer = StaticNoiseMaskLayer(config, name=f'noise_mask_{i}')
            new_layers.append(noise_layer)
    
    # 构建新模型
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs
    
    for layer in new_layers:
        x = layer(x)
    
    new_model = tf.keras.Model(inputs=inputs, outputs=x)
    
    return new_model


# =============================================================================
# 使用示例
# =============================================================================

def demo_tf():
    """
    TensorFlow版本演示
    """
    print("=" * 60)
    print("Noise Mask Defense Demo (TensorFlow)")
    print("=" * 60)
    
    # 1. 创建简单的CNN模型
    def create_simple_cnn(num_classes=10, input_shape=(32, 32, 3)):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes)
        ])
        return model
    
    # 2. 创建模型
    model = create_simple_cnn(num_classes=10)
    print(f"\nOriginal model created")
    model.summary()
    
    # 3. 创建带噪声掩码的模型
    config = NoiseMaskConfig(
        strategy=MaskStrategy.ADAPTIVE,
        noise_scale=0.15,
        noise_ratio=0.3,
        adaptive_factor=1.5,
        preserve_accuracy=True
    )
    
    protected_model = NoiseMaskModel(model, config, num_classes=10)
    print(f"\nProtected model created with noise mask strategy: {config.strategy.value}")
    
    # 4. 模拟数据
    x_member = np.random.randn(100, 32, 32, 3).astype(np.float32)
    y_member = np.random.randint(0, 10, 100)
    x_non_member = np.random.randn(100, 32, 32, 3).astype(np.float32)
    y_non_member = np.random.randint(0, 10, 100)
    
    # 5. 编译和训练
    print("\nTraining model...")
    protected_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    protected_model.fit(
        x_member, y_member,
        epochs=2,
        batch_size=16,
        verbose=1
    )
    
    # 6. 评估成员推断攻击
    print("\nEvaluating membership inference attacks...")
    
    evaluator = MembershipInferenceEvaluatorTF(attack_type='confidence')
    results = evaluator.evaluate_defense(
        protected_model, x_member, y_member, x_non_member, y_non_member
    )
    
    print(f"\n  Attack Accuracy: {results['accuracy']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  Member score mean: {results['member_score_mean']:.4f}")
    print(f"  Non-member score mean: {results['non_member_score_mean']:.4f}")
    
    # 7. 对比不同策略
    print("\n" + "=" * 60)
    print("Comparing Different Strategies")
    print("=" * 60)
    
    strategies = ['static', 'adaptive', 'confidence']
    for strategy in strategies:
        base_model = create_simple_cnn()
        protected = create_noisemask_model_tf(
            base_model, strategy=strategy, noise_scale=0.15
        )
        print(f"\n  Strategy: {strategy} - Model ready")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    
    return protected_model, results


if __name__ == "__main__":
    # 运行演示
    protected_model, results = demo_tf()
