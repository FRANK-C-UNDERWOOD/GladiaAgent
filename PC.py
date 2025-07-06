# 预测编码系统完整导入库列表
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Normal, kl_divergence

# 用于高级可视化
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

# 用于序列处理
from torch.nn.utils.rnn import pad_sequence

# 用于模型分析
from torch.utils.tensorboard import SummaryWriter

# 用于高级数学运算
import scipy.signal
import pywt  # 小波变换库

# 用于神经科学启发的组件
import snntorch as snn  # 脉冲神经网络库
from snntorch import surrogate, spikegen

# 用于不确定性量化
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# 用于多尺度处理
import torchvision.transforms as transforms

# 用于数据增强
import kornia.augmentation as K

# 用于高级优化
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR

# 用于模型保存和加载
import pickle
import json

# 用于性能分析
import time
from torch.profiler import profile, record_function, ProfilerActivity

# 用于分布式训练
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 用于内存优化
from torch.cuda.amp import autocast, GradScaler

# 用于模型解释
import shap
import lime
import captum

# 用于信号处理
import librosa

# 用于生物启发的组件
import nengo  # 神经模拟框架
#import nengo_dl

#测试
import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 确保可复现性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


class BayesianPrecisionNetwork(nn.Module):
    """
    贝叶斯精度网络 - 实现完整的不确定性量化
    输入: prediction_error [batch, input_dim]
    输出: precision, mu, log_var
    """
    def __init__(self, input_dim, hidden_dim=16, min_precision=1e-6):
        super().__init__()
        self.min_precision = min_precision
        
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ELU(),
            nn.Linear(32, hidden_dim * 2)  # 同时输出均值和方差
        )
        
        # 输出参数投影
        self.mu_proj = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 确保正值
        )
        self.log_var_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, error):
        # 编码误差信息
        h = self.encoder(error)
        mu = h[:, :h.size(1)//2]
        log_var = h[:, h.size(1)//2:]
        
        # 通过重参数化技巧采样
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        precision = 1.0 / (mu + std * eps + self.min_precision)
        
        return {
            'precision': precision,
            'mu': mu,
            'log_var': log_var,
            'sampled_precision': precision
        }

class TemporalPredictiveLayer(nn.Module):
    """
    时间感知预测层 - 整合时间维度处理
    输入: x [batch, seq_len, features]
    输出: temporal_context [batch, features], predictions
    """
    def __init__(self, input_dim, hidden_dim, mem_size=5):
        super().__init__()
        # 时间记忆缓存
        self.memory_buffer = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 时间相关的预测编码层
        self.temporal_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 时间注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=2)
    
    def forward(self, x, n_iter=3):
        # 初始时间上下文
        temporal_context, (h, c) = self.memory_buffer(x)
        
        # 时间步预测
        predictions = []
        for i in range(n_iter):
            # 注意力加权上下文
            attn_out, _ = self.attention(
                temporal_context, temporal_context, temporal_context
            )
            
            # 合并当前输入和上下文
            concat_input = torch.cat([x[:, -1, :], attn_out[:, -1, :]], dim=-1)
            
            # 生成预测
            prediction = self.temporal_predictor(concat_input)
            predictions.append(prediction)
            
            # 更新内存缓存
            if i < n_iter - 1:
                # 将预测添加到序列中
                updated_seq = torch.cat([
                    x[:, 1:, :],
                    prediction.unsqueeze(1)
                ], dim=1)
                
                # 更新LSTM状态
                temporal_context, (h, c) = self.memory_buffer(
                    updated_seq, 
                    (h.detach(), c.detach())  # 防止梯度爆炸
                )
                
        return {
            'predictions': predictions,
            'final_prediction': predictions[-1],
            'temporal_context': temporal_context[:, -1, :]
        }

class MultiScaleProcessor(nn.Module):
    """
    多尺度时间处理器 - 捕捉不同时间尺度的模式
    输入: x [batch, features, time]
    输出: fused_representation [batch, features]
    """
    def __init__(self, input_dim, scales=[1, 2, 4], scale_hidden=32):
        super().__init__()
        self.scale_processors = nn.ModuleDict()
        
        # 为每个尺度创建处理器
        for scale in scales:
            kernel_size = scale * 3
            padding = kernel_size // 2
            processor = nn.Sequential(
                nn.Conv1d(input_dim, scale_hidden, kernel_size, padding=padding),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)  # 全局池化
            )
            self.scale_processors[f'scale_{scale}'] = processor
            
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(scale_hidden * len(scales), 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
        # 尺度注意力权重
        self.scale_attention = nn.Parameter(torch.ones(len(scales)))
        
    def forward(self, x):
        outputs = []
        scales = list(self.scale_processors.keys())
        
        for i, scale_name in enumerate(scales):
            processor = self.scale_processors[scale_name]
            # 处理并压缩时间维度
            scale_output = processor(x).squeeze(-1)  # [batch, scale_hidden]
            # 应用尺度注意力权重
            weighted_output = scale_output * self.scale_attention[i]
            outputs.append(weighted_output)
            
        # 融合多尺度特征
        fused = torch.cat(outputs, dim=-1)
        return self.fusion(fused)

class AttentivePredictionFusion(nn.Module):
    """
    注意力引导预测融合 - 平衡输入和预测表示
    输入: x [batch, features], prediction [batch, features]
    输出: refined_prediction [batch, features]
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # 注意力计算参数
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # 信息融合门控
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, prediction):
        # 计算注意力权重
        q = self.query(prediction)  # 基于当前预测查询
        k = self.key(x)            # 输入特征作为键
        v = self.value(x)           # 输入特征作为值
        
        # 点积注意力
        attn_weights = F.softmax(torch.bmm(q.unsqueeze(1), k.unsqueeze(2)), dim=2)
        attn_weights = attn_weights.squeeze(1)
        
        # 注意力加权的值
        attended = torch.sum(attn_weights * v, dim=1)
        
        # 门控融合
        gate_value = self.gate(torch.cat([prediction, attended], dim=1))
        
        # 残差连接融合
        fused_prediction = (1 - gate_value) * prediction + gate_value * attended
        
        return {
            'fused_prediction': fused_prediction,
            'attn_weights': attn_weights,
            'gate_value': gate_value
        }

class DynamicIterationController(nn.Module):
    """
    动态迭代控制器 - 自适应调整迭代次数
    输入: initial_error [batch, features], min_iter, max_iter
    输出: iteration_count [scalar]
    """
    def __init__(self, input_dim, hidden_dim=16, min_iter=1, max_iter=10):
        super().__init__()
        self.min_iter = min_iter
        self.max_iter = max_iter
        
        # 误差强度评估器
        self.error_assessor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1的强度值
        )
    
    def forward(self, initial_error):
        # 评估误差强度
        error_norm = torch.norm(initial_error, dim=1, keepdim=True)
        normalized_error = error_norm / (error_norm.max(dim=0, keepdim=True)[0] + 1e-8)
        
        # 计算迭代次数
        intensity = self.error_assessor(initial_error).mean()
        iteration_count = self.min_iter + int((self.max_iter - self.min_iter) * intensity)
        
        return torch.clamp(iteration_count, self.min_iter, self.max_iter)

class AdaptiveFreeEnergyCalculator(nn.Module):
    """
    自适应自由能计算器 - 动态平衡自由能项
    输入: error, precision, representation
    输出: free_energy [scalar]
    """
    def __init__(self, initial_alpha=1.0, initial_beta=0.5):
        super().__init__()
        # 可学习的平衡参数
        self.log_alpha = nn.Parameter(torch.tensor(np.log(initial_alpha)))
        self.log_beta = nn.Parameter(torch.tensor(np.log(initial_beta)))
        
        # 防止零除的常数
        self.eps = 1e-6
    
    def forward(self, error, precision, representation):
        # 确保数值稳定
        precision = torch.clamp(precision, min=self.eps)
        
        # 计算基本项
        accuracy_term = 0.5 * torch.sum(precision * (error ** 2), dim=-1)
        complexity_term = 0.5 * torch.sum(representation ** 2, dim=-1)
        
        # 应用可学习权重
        alpha = torch.exp(self.log_alpha) + self.eps
        beta = torch.exp(self.log_beta) + self.eps
        
        return alpha * accuracy_term + beta * complexity_term

# 替代 NeuroModulationSystem 的纯 PyTorch 实现
class NeuroModulationSystem(nn.Module):
    """
    神经调制系统 - 动态调整各层学习率
    输入: layer_errors (列表，包含各层误差张量)
    输出: modulation_factors (各层调制系数)
    """
    def __init__(self, num_layers, hidden_dim=8):
        super().__init__()
        self.num_layers = num_layers
        
        # 调制参数生成器
        self.modulator_generator = nn.Sequential(
            nn.Linear(num_layers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_layers),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, layer_errors):
        # 计算每层平均误差
        error_metrics = torch.stack([
            torch.mean(torch.abs(e)) for e in layer_errors
        ]).unsqueeze(0)  # [1, num_layers]
        
        # 生成调制参数
        modulation_factors = self.modulator_generator(error_metrics)
        return modulation_factors.squeeze(0)

class EnhancedIterativePredictiveLayer(nn.Module):
    """
    增强版迭代预测层 - 整合所有改进
    输入: x [batch, input_dim], 可选memory_vector [batch, memory_dim]
    输出: predictions, free_energy, 及其他诊断信息
    """
    def __init__(self, input_dim, memory_dim=None, hidden_dim=64, 
                 min_iter=2, max_iter=8):
        super().__init__()
        # 1. 记忆适配器（如果提供记忆维度）
        if memory_dim:
            self.memory_adapter = nn.Linear(memory_dim, input_dim)
        else:
            self.memory_adapter = None
            
        # 2. 生成模型
        self.generative_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 3. 使用贝叶斯精度网络
        self.precision_network = BayesianPrecisionNetwork(input_dim)
        
        # 4. 注意力预测融合
        self.attention_fusion = AttentivePredictionFusion(input_dim)
        
        # 5. 动态迭代控制器
        self.iter_controller = DynamicIterationController(input_dim, min_iter=min_iter, max_iter=max_iter)
        
        # 6. 自适应自由能计算
        self.free_energy_calc = AdaptiveFreeEnergyCalculator()
        
        # 7. 内部表示更新率
        self.internal_lr = nn.Parameter(torch.tensor(0.1))
        
        # 8. 收敛检测
        self.convergence_detector = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, memory_vector=None):
        # 合并记忆信息
        if memory_vector is not None and self.memory_adapter:
            mem_info = self.memory_adapter(memory_vector)
            adapted_x = x + mem_info
        else:
            adapted_x = x
            
        # 初始误差
        initial_error = adapted_x - self.generative_model(adapted_x)
        
        # 动态确定迭代次数
        iterations = self.iter_controller(initial_error)
        
        # 迭代推理
        current_belief = adapted_x.clone()
        predictions = []
        precision_info = []
        
        for i in range(iterations):
            # 生成预测
            prediction = self.generative_model(current_belief)
            
            # 计算预测误差
            prediction_error = adapted_x - prediction
            
            # 估计精度
            precision_data = self.precision_network(prediction_error)
            precision = precision_data['precision']
            
            # 注意力融合改进预测
            fused_result = self.attention_fusion(adapted_x, prediction)
            enhanced_prediction = fused_result['fused_prediction']
            
            # 计算自由能
            free_energy = self.free_energy_calc(
                prediction_error, precision, current_belief
            )
            
            # 检查收敛性
            convergence_prob = self.convergence_detector(prediction_error)
            converged = convergence_prob > 0.85
            
            # 更新belief
            current_belief = current_belief + self.internal_lr * (enhanced_prediction - current_belief)
            
            # 存储迭代结果
            iter_result = {
                'prediction': prediction,
                'enhanced_prediction': enhanced_prediction,
                'error': prediction_error,
                'precision': precision,
                'free_energy': free_energy,
                'converged': converged
            }
            iter_result.update(precision_data)
            predictions.append(iter_result)
            
            # 如果所有样本都收敛，提前终止
            if converged.all():
                break
        
        return {
            'iterations': iterations,
            'predictions': predictions,
            'final_prediction': predictions[-1]['enhanced_prediction'] if predictions else None
        }

class PredictiveCodingAnalyzer:
    """
    预测编码分析工具 - 提供可视化和诊断功能
    """
    def __init__(self, model):
        self.model = model
        self.activation_history = {}
        
    def register_hooks(self):
        """注册前向钩子以捕获激活状态"""
        hooks = []
        def save_activation(name):
            def hook(module, input, output):
                self.activation_history[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.LSTM)):
                hook = module.register_forward_hook(save_activation(name))
                hooks.append(hook)
                
        return hooks
    
    def visualize_prediction_flow(self, inputs):
        """可视化预测流和误差传播"""
        # 注册钩子并运行前向传播
        hooks = self.register_hooks()
        with torch.no_grad():
            outputs = self.model(inputs)
        # 移除钩子
        for hook in hooks:
            hook.remove()
            
        # 创建可视化
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. 误差分布图
        errors = []
        for i, pred in enumerate(outputs['layer_results'][0]):
            errors.append(pred['error'].cpu().numpy())
            
        axes[0].boxplot(errors)
        axes[0].set_title("Prediction Error Distribution Across Layers")
        axes[0].set_xlabel("Layer Index")
        axes[0].set_ylabel("Error Magnitude")
        
        # 2. 自由能变化图
        energies = []
        for i, pred in enumerate(outputs['layer_results'][0]):
            energies.append(pred['free_energy'].mean().item())
            
        axes[1].plot(energies, marker='o')
        axes[1].set_title("Free Energy Across Layers")
        axes[1].set_xlabel("Layer Index")
        axes[1].set_ylabel("Free Energy")
        axes[1].grid(True)
        
        return fig
    
    def uncertainty_heatmap(self, inputs, num_samples=20):
        """生成预测不确定性热力图"""
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model(inputs)
                predictions.append(outputs['final_prediction'])
        
        pred_tensor = torch.stack(predictions)  # [samples, batch, features]
        variance = torch.var(pred_tensor, dim=0)
        
        # 可视化
        plt.figure(figsize=(10, 8))
        sns.heatmap(variance.cpu().numpy(), cmap='viridis')
        plt.title("Predictive Uncertainty Heatmap")
        plt.xlabel("Feature Dimension")
        plt.ylabel("Batch Index")
        return plt

class AdvancedPredictiveCodingSystem(nn.Module):
    """
    高级预测编码系统 - 整合所有改进
    """
    def __init__(self, input_dim, layer_dims=[128, 64, 32], 
                 memory_dim=None, scales=[1, 2, 4]):
        super().__init__()
        # 创建层级结构
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        # 如果提供内存维度，创建第一个内存适配器
        self.memory_encoder = None
        if memory_dim:
            self.memory_encoder = nn.Linear(input_dim, memory_dim)
            memory_vect = memory_dim
        else:
            memory_vect = None
            
        # 创建多层预测系统
        for i, dim in enumerate(layer_dims):
            # 时间处理器 (仅第一层)
            temporal_processor = None
            if i == 0:
                temporal_processor = TemporalPredictiveLayer(
                    input_dim=prev_dim,
                    hidden_dim=dim
                )
            
            # 多尺度处理器
            scale_processor = MultiScaleProcessor(
                input_dim=prev_dim,
                scales=scales
            ) if i < len(layer_dims) - 1 else None
            
            # 预测层
            layer = EnhancedIterativePredictiveLayer(
                input_dim=prev_dim,
                memory_dim=memory_vect,
                hidden_dim=dim
            )
            
            self.layers.append(nn.ModuleDict({
                'temporal_processor': temporal_processor,
                'scale_processor': scale_processor,
                'predictive_layer': layer
            }))
            prev_dim = dim
            
        # 神经调制系统
        self.neuromodulation = NeuroModulationSystem(len(layer_dims))
        
        # 输出层
        self.output_layer = nn.Linear(layer_dims[-1], input_dim)
        
    def forward(self, x):
        # 如果使用内存系统
        if self.memory_encoder:
            memory_vector = self.memory_encoder(x)
        else:
            memory_vector = None
            
        # 分层处理
        all_results = []
        layer_errors = []
        current = x
        
        for i, layer_block in enumerate(self.layers):
            # 时间处理 (仅第一层)
            if layer_block['temporal_processor']:
                temporal_result = layer_block['temporal_processor'](current.unsqueeze(1))
                current = temporal_result['final_prediction']
            
            # 多尺度处理
            if layer_block['scale_processor']:
                # 添加时间维度 (batch, channels, time=1)
                scale_output = layer_block['scale_processor'](current.unsqueeze(2))
                current = scale_output
                
            # 预测编码层
            prediction_result = layer_block['predictive_layer'](
                current, 
                memory_vector=memory_vector
            )
            all_results.append(prediction_result)
            layer_errors.append(prediction_result['predictions'][-1]['error'])
            
            # 更新当前表示
            current = prediction_result['final_prediction']
            
            # 提前退出条件
            if prediction_result.get('early_exit', False):
                break
                
        # 应用神经调制系统
        modulation = self.neuromodulation(layer_errors)
        
        # 最终预测
        output = self.output_layer(current)
        
        return {
            'output': output,
            'all_results': all_results,
            'modulation': modulation,
            'memory_vector': memory_vector,
            'layer_errors': layer_errors
        }
    
class UnifiedTrainer:
    """预测编码系统统一训练器"""
    
    def __init__(self, model, 
                 learning_rate=1e-3, 
                 memory_lr=5e-4, 
                 clip_value=1.0,
                 free_energy_weight=0.5):
        self.model = model
        self.clip_value = clip_value
        self.free_energy_weight = free_energy_weight
        
        # 分离内存编码器参数和其他参数
        memory_params = list(self.model.memory_encoder.parameters()) if hasattr(self.model, 'memory_encoder') else []
        predictive_params = [p for n, p in self.model.named_parameters() if not n.startswith('memory_encoder')]
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            predictive_params, 
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.memory_optimizer = optim.AdamW(
            memory_params, 
            lr=memory_lr
        ) if memory_params else None
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
    def compute_total_free_energy(self, results):
        """计算系统的总自由能"""
        total_free_energy = 0.0
        count = 0
        
        # 遍历所有时间步
        for timestep in results.get('layer_results', []):
            # 遍历所有层
            for layer_result in timestep:
                # 获取该层的预测结果
                predictions = layer_result.get('predictions', [])
                if predictions:
                    # 使用最后一次迭代的自由能
                    last_iter = predictions[-1]
                    free_energy = last_iter.get('free_energy', 0.0)
                    if isinstance(free_energy, torch.Tensor):
                        total_free_energy += free_energy.sum().item()
                        count += free_energy.numel()
        
        # 计算平均自由能
        return total_free_energy / count if count > 0 else 0.0
    
    def compute_prediction_loss(self, predictions, targets):
        """计算预测损失"""
        # 确保预测和目标形状匹配
        predictions = predictions.view_as(targets)
        return F.mse_loss(predictions, targets)
    
    def compute_kl_divergence(self, precision_data):
        """计算KL散度正则化项"""
        kl_loss = 0.0
        for mu, log_var in precision_data.get('precision_params', []):
            # 标准正态分布先验
            prior = torch.distributions.Normal(0, 1)
            # 变分后验
            posterior = torch.distributions.Normal(mu, torch.exp(0.5 * log_var))
            kl_loss += torch.distributions.kl_divergence(posterior, prior).sum()
        return kl_loss
    
    def train_step(self, inputs, targets):
        """执行单次训练步骤"""
        # 前向传播
        outputs = self.model(inputs)
        
        # 计算各种损失
        prediction_loss = self.compute_prediction_loss(outputs['output'], targets)
        free_energy = self.compute_total_free_energy(outputs)
        kl_divergence = self.compute_kl_divergence(outputs.get('precision_data', {}))
        
        # 组合损失
        total_loss = prediction_loss + self.free_energy_weight * free_energy + kl_divergence * 0.1
        
        # 优化器清零梯度
        self.optimizer.zero_grad()
        if self.memory_optimizer:
            self.memory_optimizer.zero_grad()
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
        
        # 参数更新
        self.optimizer.step()
        if self.memory_optimizer:
            self.memory_optimizer.step()
        
        # 更新学习率
        self.scheduler.step(total_loss)
        
        return {
            'total_loss': total_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'free_energy': free_energy,
            'kl_divergence': kl_divergence.item() if isinstance(kl_divergence, torch.Tensor) else kl_divergence
        }
    
    def evaluate(self, inputs, targets):
        """评估模型性能"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            prediction_loss = self.compute_prediction_loss(outputs['output'], targets).item()
            free_energy = self.compute_total_free_energy(outputs)
            
            # 计算准确率（根据应用场景）
            accuracy = 0.0
            if hasattr(self.model, 'compute_accuracy'):
                accuracy = self.model.compute_accuracy(outputs, targets)
            
            return {
                'prediction_loss': prediction_loss,
                'free_energy': free_energy,
                'accuracy': accuracy,
                'memory_vector': outputs.get('memory_vector')
            }

class TestPredictiveCodingSystem(unittest.TestCase):
    """预测编码系统测试套件"""
    
    def setUp(self):
        """初始化测试环境"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 创建测试数据
        self.batch_size = 8
        self.seq_len = 20
        self.input_dim = 64
        
        # 生成正弦波序列数据
        t = np.linspace(0, 4*np.pi, self.seq_len)
        self.data = np.zeros((self.batch_size, self.seq_len, self.input_dim))
        for i in range(self.batch_size):
            for j in range(self.input_dim):
                freq = 0.5 + 0.1 * j
                phase = np.random.uniform(0, 2*np.pi)
                self.data[i, :, j] = np.sin(freq * t + phase)
        
        # 转换为PyTorch张量
        self.inputs = torch.tensor(self.data, dtype=torch.float32)
        self.targets = torch.roll(self.inputs, shifts=-1, dims=1)  # 下一个时间步作为目标
        
        # 创建数据加载器
        dataset = TensorDataset(self.inputs, self.targets)
        self.dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    def test_memory_encoder(self):
        """测试记忆编码器"""
        from PC import MemoryEncoder  # 替换为实际模块路径
        
        # 创建编码器
        encoder = MemoryEncoder(
            input_size=self.input_dim,
            hidden_size=128,
            memory_dim=64
        )
        
        # 前向传播
        memory_vector = encoder(self.inputs)
        
        # 验证输出
        self.assertEqual(memory_vector.shape, (self.batch_size, 64))
        self.assertFalse(torch.isnan(memory_vector).any())
        self.assertFalse(torch.isinf(memory_vector).any())
        
        # 验证反向传播
        loss = memory_vector.sum()
        loss.backward()
        for name, param in encoder.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
    
    def test_iterative_predictive_layer(self):
        """测试迭代预测层"""
        from PC import IterativePredictiveLayer  # 替换为实际模块路径
        
        # 创建预测层
        layer = IterativePredictiveLayer(
            input_dim=self.input_dim,
            hidden_dim=32,
            min_iter=2,
            max_iter=5
        )
        
        # 测试单样本
        sample = self.inputs[0, 0]  # 第一个样本的第一个时间步
        results = layer(sample)
        
        # 验证输出结构
        self.assertIn('iterations', results)
        self.assertIn('predictions', results)
        self.assertIn('final_prediction', results)
        
        # 验证迭代次数
        self.assertGreaterEqual(results['iterations'], 2)
        self.assertLessEqual(results['iterations'], 5)
        
        # 验证预测结果
        self.assertEqual(len(results['predictions']), results['iterations'])
        self.assertEqual(results['final_prediction'].shape, sample.shape)
        
        # 测试批量处理
        batch = self.inputs[:, 0]  # 所有样本的第一个时间步
        batch_results = layer(batch)
        self.assertEqual(batch_results['final_prediction'].shape, batch.shape)
    
    def test_multi_scale_processor(self):
        """测试多尺度处理器"""
        from PC import MultiScaleProcessor  # 替换为实际模块路径
        
        # 创建处理器
        processor = MultiScaleProcessor(
            input_dim=self.input_dim,
            scales=[1, 2, 4]
        )
        
        # 测试处理
        sample = self.inputs[0].permute(1, 0)  # [features, time]
        output = processor(sample.unsqueeze(0))  # 添加批次维度
        
        # 验证输出
        self.assertEqual(output.shape, (1, self.input_dim))
        
        # 测试不同时间长度
        short_input = torch.randn(1, self.input_dim, 5)  # 短序列
        output_short = processor(short_input)
        self.assertEqual(output_short.shape, (1, self.input_dim))
    
    def test_dynamic_iteration_controller(self):
        """测试动态迭代控制器"""
        from PC import DynamicIterationController  # 替换为实际模块路径
        
        # 创建控制器
        controller = DynamicIterationController(
            input_dim=self.input_dim,
            min_iter=2,
            max_iter=8
        )
        
        # 测试不同误差水平
        small_error = torch.randn(4, self.input_dim) * 0.1
        large_error = torch.randn(4, self.input_dim) * 1.0
        
        iter_small = controller(small_error)
        iter_large = controller(large_error)
        
        # 验证迭代次数范围
        self.assertGreaterEqual(iter_small, 2)
        self.assertLessEqual(iter_small, 8)
        self.assertGreaterEqual(iter_large, 2)
        self.assertLessEqual(iter_large, 8)
        
        # 验证大误差需要更多迭代
        self.assertGreater(iter_large, iter_small)
    
    def test_full_system_forward(self):
        """测试完整系统前向传播"""
        from PC import AdvancedPredictiveCodingSystem  # 替换为实际模块路径
        
        # 创建系统
        system = AdvancedPredictiveCodingSystem(
            input_dim=self.input_dim,
            layer_dims=[48, 32, 16],
            memory_dim=32
        )
        
        # 前向传播
        outputs = system(self.inputs)
        
        # 验证输出结构
        self.assertIn('output', outputs)
        self.assertIn('all_results', outputs)
        self.assertIn('modulation', outputs)
        self.assertIn('memory_vector', outputs)
        self.assertIn('layer_errors', outputs)
        
        # 验证输出形状
        self.assertEqual(outputs['output'].shape, self.inputs.shape)
        self.assertEqual(len(outputs['all_results']), 3)  # 3层
        self.assertEqual(outputs['modulation'].shape, (3,))  # 3层调制因子
        self.assertEqual(outputs['memory_vector'].shape, (self.batch_size, 32))
        self.assertEqual(len(outputs['layer_errors']), 3)  # 3层误差
    
    def test_training_loop(self):
        """测试训练循环"""
        from PC import AdvancedPredictiveCodingSystem, UnifiedTrainer  # 替换为实际模块路径
        
        # 创建系统和训练器
        system = AdvancedPredictiveCodingSystem(
            input_dim=self.input_dim,
            layer_dims=[48, 32],
            memory_dim=32
        )
        trainer = UnifiedTrainer(system, learning_rate=1e-3)
        
        # 初始损失
        initial_loss = trainer.evaluate(self.inputs, self.targets)['total_loss']
        
        # 训练一个epoch
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            stats = trainer.train_step(inputs, targets)
            if batch_idx > 2:  # 只运行几个批次
                break
        
        # 训练后损失
        trained_loss = trainer.evaluate(self.inputs, self.targets)['total_loss']
        
        # 验证损失下降
        self.assertLess(trained_loss, initial_loss)
    
    def test_convergence_detection(self):
        """测试收敛检测机制"""
        from PC import IterativePredictiveLayer  # 替换为实际模块路径
        
        # 创建预测层
        layer = IterativePredictiveLayer(
            input_dim=self.input_dim,
            hidden_dim=32,
            min_iter=2,
            max_iter=10
        )
        
        # 测试收敛情况
        convergent_input = torch.zeros(1, self.input_dim)  # 零输入应快速收敛
        results = layer(convergent_input)
        
        # 验证提前终止
        self.assertLess(results['iterations'], 10)
        
        # 检查收敛标志
        last_pred = results['predictions'][-1]
        self.assertTrue(last_pred['converged'].all())
    
    def test_uncertainty_quantification(self):
        """测试不确定性量化"""
        from PC import BayesianPrecisionNetwork  # 替换为实际模块路径
        
        # 创建精度网络
        precision_net = BayesianPrecisionNetwork(
            input_dim=self.input_dim,
            hidden_dim=16
        )
        
        # 生成测试误差
        error = torch.randn(4, self.input_dim)
        
        # 前向传播
        outputs = precision_net(error)
        
        # 验证输出结构
        self.assertIn('precision', outputs)
        self.assertIn('mu', outputs)
        self.assertIn('log_var', outputs)
        self.assertIn('sampled_precision', outputs)
        
        # 验证精度值范围
        precision = outputs['precision']
        self.assertTrue(torch.all(precision > 0))
        self.assertTrue(torch.all(precision < 100))  # 合理范围
    
    def test_analyzer_tool(self):
        """测试分析工具"""
        from PC import AdvancedPredictiveCodingSystem, PredictiveCodingAnalyzer  # 替换为实际模块路径
        
        # 创建系统和分析器
        system = AdvancedPredictiveCodingSystem(
            input_dim=self.input_dim,
            layer_dims=[48, 32]
        )
        analyzer = PredictiveCodingAnalyzer(system)
        
        # 运行前向传播以捕获激活
        with torch.no_grad():
            system(self.inputs)
        
        # 测试可视化功能
        fig = analyzer.visualize_prediction_flow(self.inputs)
        self.assertIsInstance(fig, plt.Figure)
        
        # 测试不确定性热力图
        plt_obj = analyzer.uncertainty_heatmap(self.inputs[:2])
        self.assertIsInstance(plt_obj, plt.Axes)
    
    def test_performance_benchmark(self):
        """性能基准测试"""
        from PC import AdvancedPredictiveCodingSystem  # 替换为实际模块路径
        import time
        
        # 创建系统
        system = AdvancedPredictiveCodingSystem(
            input_dim=self.input_dim,
            layer_dims=[64, 48, 32]
        )
        
        # 预热
        with torch.no_grad():
            _ = system(self.inputs[:1])
        
        # 测试推理速度
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = system(self.inputs[:4])
        inference_time = (time.time() - start_time) / 10
        
        print(f"平均推理时间 (batch=4): {inference_time:.4f}s")
        
        # 测试训练速度
        trainer = UnifiedTrainer(system)
        start_time = time.time()
        for inputs, targets in self.dataloader:
            trainer.train_step(inputs, targets)
        training_time = time.time() - start_time
        
        print(f"单epoch训练时间: {training_time:.4f}s")
        
        # 内存占用测试
        from pympler import asizeof
        model_size = asizeof.asizeof(system) / (1024 * 1024)  # MB
        print(f"模型内存占用: {model_size:.2f} MB")
        
        # 验证性能在可接受范围内
        self.assertLess(inference_time, 0.5)  # 500ms内完成推理
        self.assertLess(training_time, 5.0)  # 5秒内完成一个epoch
        self.assertLess(model_size, 50)  # 小于50MB

if __name__ == '__main__':
    unittest.main(verbosity=2)