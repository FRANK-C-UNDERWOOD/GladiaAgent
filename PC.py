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
from config import input_dim as D
# 核心维度参数
#B = batch_size  # 批次大小
#T = seq_len     # 序列长度
#D = 384         # 特征维度

from config import input_dim as D  



# 确保可复现性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class PredictiveEncoder(nn.Module):
    def __init__(self, input_dim: int = 384):
        super().__init__()
        self.input_dim = input_dim
        D = input_dim

        # 使用固定维度 D 构建所有线性层
        self.linear1 = nn.Linear(D, D)
        self.linear2 = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
    def register_safe_hook(self, name: str, module: nn.Module):
        """
        为指定模块注册一个安全的 forward hook，自动保存激活值
        """
       
    def forward(self, x: torch.Tensor):
        """
        x:  [B, T, D]
        B = batch size 
        T = sequence length 
        D = input_dim       
        """
        assert x.shape[-1] == self.input_dim, \
            f"Expected last dim {self.input_dim}, got {x.shape[-1]}"
        
        residual = x

        # BT自动广播
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)

        x = self.norm(x + residual) 

        return x 
    
class BayesianPrecisionNetwork(nn.Module):
    """
    贝叶斯精度网络 - 实现完整的不确定性量化
    输入: prediction_error [B, T, D]
    输出: 
        precision [B, T, 1]
        mu [B, T, 1]
        log_var [B, T, 1]
        sampled_precision [B, T, 1]
    """
    def __init__(self, input_dim, hidden_dim=16, min_precision=1e-6):
        super().__init__()
        self.min_precision = min_precision
        
        # 编码器网络 - 保持时间维度不变
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ELU(),
            nn.Linear(32, hidden_dim * 2)  # 同时输出均值和方差
        )
        
        # 输出参数投影 - 保持时间维度不变
        self.mu_proj = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 确保正值
        )
        self.log_var_proj = nn.Linear(hidden_dim, 1)
    

    def forward(self, error):
        """
        error: [B, T, D] 预测误差
        输出: 字典包含:
            precision: [B, T, 1]
            mu: [B, T, 1]
            log_var: [B, T, 1]
            sampled_precision: [B, T, 1]
        """
        # 1. 维度验证
        assert error.dim() == 3, f"输入应为三维 [B, T, D]，实际为 {error.shape}"
        
        # 2. 编码误差信息
        h = self.encoder(error)  # [B, T, hidden_dim*2]
        
        # 3. 分割隐状态为均值和方差部分
        # 沿特征维度分割为两半
        split_point = h.size(-1) // 2
        mu_part = h[..., :split_point]  # [B, T, hidden_dim]
        log_var_part = h[..., split_point:]  # [B, T, hidden_dim]
        
        # 4. 投影到最终参数
        mu = self.mu_proj(mu_part)  # [B, T, 1]
        log_var = self.log_var_proj(log_var_part)  # [B, T, 1]
        
        # 5. 通过重参数化技巧采样
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
    输入: x [B, T, D]
    输出: 
        predictions: 列表包含 n_iter 个预测，每个 [B, T, D]
        final_prediction: [B, T, D]
        temporal_context: [B, T, hidden_dim]
    """
    def __init__(self, input_dim=D, hidden_dim=256):
        super().__init__()
        # 时间记忆缓存 (使用全局D)
        self.memory_buffer = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 时间相关的预测编码层 (保持输出维度D)
        self.temporal_predictor = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # 输出维度匹配全局D
        )
        
        # 时间注意力机制 (添加batch_first支持)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # 隐藏状态适配器
        self.context_adapter = nn.Linear(hidden_dim, input_dim) if hidden_dim != input_dim else nn.Identity()
        
        

    def forward(self, x, n_iter=3):
        """
        输入: x [B, T, D]
        输出: 
            predictions: 列表包含 n_iter 个预测，每个 [B, T, D]
            final_prediction: [B, T, D]
            temporal_context: [B, T, hidden_dim]
        """
        # 验证输入维度
        assert x.dim() == 3 and x.shape[-1] == D, f"输入应为 [B, T, D] 形状, 实际为 {x.shape}"
        
        # 初始时间上下文
        temporal_context, (h, c) = self.memory_buffer(x)  # [B, T, hidden_dim]
        
        # 时间步预测
        predictions = []
        for i in range(n_iter):
            # 注意力加权上下文
            attn_out, _ = self.attention(
                temporal_context, temporal_context, temporal_context
            )  # [B, T, hidden_dim]
            
            # 合并当前输入和上下文
            concat_input = torch.cat([
                x,  # 原始输入 [B, T, D]
                self.context_adapter(attn_out)  # 适配为 [B, T, D]
            ], dim=-1)  # [B, T, 2D]
            
            # 生成预测
            prediction = self.temporal_predictor(concat_input)  # [B, T, D]
            predictions.append(prediction)
            
            # 更新内存缓存
            if i < n_iter - 1:
                # 将预测添加到序列中 (作为下一个时间步输入)
                updated_seq = torch.cat([
                    x[:, :-1, :],  # 保留前T-1个时间步
                    prediction[:, -1:, :]  # 用预测替换最后一个时间步
                ], dim=1)  # [B, T, D]
                
                # 更新LSTM状态
                temporal_context, (h, c) = self.memory_buffer(
                    updated_seq, 
                    (h.detach(), c.detach())  # 防止梯度爆炸
                )
                
        return {
            'predictions': predictions,  # 列表包含 n_iter 个 [B, T, D]
            'final_prediction': predictions[-1],  # [B, T, D]
            'temporal_context': temporal_context  # [B, T, hidden_dim]
        }

class MultiScaleProcessor(nn.Module):
    """
    多尺度时间处理器 - 捕捉不同时间尺度的模式
    输入: x [B, T, D]
    输出: fused_representation [B, T, D]
    """
    def __init__(self, input_dim=D, scales=[1, 2, 4], scale_hidden=128):
        super().__init__()
        self.scale_processors = nn.ModuleDict()
        
        # 为每个尺度创建处理器
        for scale in scales:
            kernel_size = scale * 3
            padding = kernel_size // 2
            
            processor = nn.Sequential(
                # 时间维度卷积 (T维度)
                nn.Conv1d(
                    in_channels=input_dim, 
                    out_channels=scale_hidden,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode='replicate'
                ),
                nn.ReLU(),
                # 保持时间维度不变
                nn.Identity()
            )
            self.scale_processors[f'scale_{scale}'] = processor
            
        # 特征融合 (保持时间维度)
        self.fusion = nn.Sequential(
            nn.Linear(scale_hidden * len(scales), input_dim),
            nn.ReLU()
        )
        
        # 尺度注意力权重 (每个尺度一个权重)
        self.scale_attention = nn.Parameter(torch.ones(len(scales)))
        
        

    def forward(self, x):
        """
        输入: x [B, T, D]
        输出: [B, T, D]
        """
        # 验证输入维度
        assert x.dim() == 3 and x.shape[-1] == D, f"输入应为 [B, T, D] 形状, 实际为 {x.shape}"
        
        # 转换为卷积友好的格式 [B, D, T]
        x = x.permute(0, 2, 1)  # [B, D, T]
        
        outputs = []
        scales = list(self.scale_processors.keys())
        
        for i, scale_name in enumerate(scales):
            processor = self.scale_processors[scale_name]
            # 处理尺度特征 [B, scale_hidden, T]
            scale_output = processor(x)
            
            # 应用尺度注意力权重
            weighted_output = scale_output * self.scale_attention[i]
            
            # 转换回原始格式 [B, T, scale_hidden]
            weighted_output = weighted_output.permute(0, 2, 1)
            outputs.append(weighted_output)
            
        # 融合多尺度特征 [B, T, scale_hidden * num_scales]
        fused = torch.cat(outputs, dim=-1)
        
        # 融合特征 [B, T, D]
        fused_rep = self.fusion(fused)
        return fused_rep

class AttentivePredictionFusion(nn.Module):
    """
    注意力引导预测融合 - 平衡输入和预测表示
    输入: x [B, T, D], prediction [B, T, D]
    输出: refined_prediction [B, T, D]
    """
    def __init__(self, input_dim=D, hidden_dim=128):
        super().__init__()
        # 注意力计算参数
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, input_dim)  # 输出维度匹配输入D
        
        # 信息融合门控
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, 1),  # 输入是2*D
            nn.Sigmoid()
        )

       

    def forward(self, x, prediction):
        """
        输入: 
            x [B, T, D] - 输入特征
            prediction [B, T, D] - 预测特征
        输出: 
            fused_prediction [B, T, D]
            attn_weights [B, T, T]
            gate_value [B, T, 1]
        """
        # 验证输入维度
        assert x.shape == prediction.shape, "输入和预测维度不一致"
        assert x.dim() == 3 and x.shape[-1] == D, f"输入应为 [B, T, D] 形状, 实际为 {x.shape}"
        
        # 1. 计算注意力权重
        q = self.query(prediction)  # [B, T, hidden_dim]
        k = self.key(x)             # [B, T, hidden_dim]
        
        # 点积注意力 (批次内处理)
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # [B, T, T]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T, T]
        
        # 2. 注意力加权的值
        v = self.value(x)  # [B, T, D]
        attended = torch.bmm(attn_weights, v)  # [B, T, D]
        
        # 3. 门控融合
        gate_input = torch.cat([prediction, attended], dim=-1)  # [B, T, 2*D]
        gate_value = self.gate(gate_input)  # [B, T, 1]
        
        # 4. 残差连接融合
        fused_prediction = (1 - gate_value) * prediction + gate_value * attended  # [B, T, D]
        
        return {
            'fused_prediction': fused_prediction,
            'attn_weights': attn_weights,
            'gate_value': gate_value
        }

class DynamicIterationController(nn.Module):
    """
    动态迭代控制器 - 自适应调整迭代次数
    输入: initial_error [B, T, D]
    输出: iteration_count [B] (每个样本的迭代次数)
    """
    def __init__(self, input_dim=D, hidden_dim=64, min_iter=1, max_iter=10):
        super().__init__()
        self.min_iter = min_iter
        self.max_iter = max_iter
        
        # 误差强度评估器 - 处理时间序列
        self.error_assessor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1的强度值
        )

       

    def forward(self, initial_error):
        """
        输入: initial_error [B, T, D] - 初始预测误差
        输出: iteration_counts [B] - 每个样本的迭代次数
        """
        # 验证输入维度
        assert initial_error.dim() == 3 and initial_error.shape[-1] == D, \
            f"输入应为 [B, T, D] 形状, 实际为 {initial_error.shape}"
        
        # 1. 计算每个时间步的误差范数
        error_norm = torch.norm(initial_error, dim=2)  # [B, T]
        
        # 2. 取最大时间步误差作为样本误差
        max_error, _ = torch.max(error_norm, dim=1, keepdim=True)  # [B, 1]
        
        # 3. 评估误差强度 (使用最大误差)
        intensity = self.error_assessor(max_error).squeeze(1)  # [B]
        
        # 4. 计算每个样本的迭代次数
        iteration_counts = self.min_iter + (self.max_iter - self.min_iter) * intensity
        
        # 5. 四舍五入为整数并限制范围
        iteration_counts = torch.round(iteration_counts).long()
        iteration_counts = torch.clamp(iteration_counts, self.min_iter, self.max_iter)
        
        return iteration_counts
    
class AdaptiveFreeEnergyCalculator(nn.Module):
    """
    自适应自由能计算器 - 动态平衡自由能项
    输入: 
        error [B, T, D]
        precision [B, T, 1]  # 来自BayesianPrecisionNetwork
        representation [B, T, D]
    输出: free_energy [B, T]  # 每个时间步的自由能
    """
    def __init__(self, initial_alpha=1.0, initial_beta=0.5):
        super().__init__()
        # 可学习的平衡参数
        self.log_alpha = nn.Parameter(torch.tensor(np.log(initial_alpha)))
        self.log_beta = nn.Parameter(torch.tensor(np.log(initial_beta)))
        
        # 防止零除的常数
        self.eps = 1e-6
        
        

    def forward(self, error, precision, representation):
        """
        输入:
            error: [B, T, D]
            precision: [B, T, 1]
            representation: [B, T, D]
        输出:
            free_energy: [B, T] (每个时间步的自由能)
        """
        # 验证输入维度
        assert error.shape == representation.shape, "误差和表示维度不一致"
        assert precision.dim() == 3 and precision.shape[-1] == 1, f"精度应为 [B, T, 1] 形状, 实际为 {precision.shape}"
        
        # 确保数值稳定
        precision = torch.clamp(precision, min=self.eps)
        
        # 扩展精度维度以匹配误差维度
        expanded_precision = precision.expand_as(error)  # [B, T, D]
        
        # 计算基本项 (按特征维度求和)
        accuracy_term = 0.5 * torch.sum(expanded_precision * (error ** 2), dim=-1)  # [B, T]
        complexity_term = 0.5 * torch.sum(representation ** 2, dim=-1)  # [B, T]
        
        # 应用可学习权重
        alpha = torch.exp(self.log_alpha) + self.eps
        beta = torch.exp(self.log_beta) + self.eps
        
        # 计算自由能
        free_energy = alpha * accuracy_term + beta * complexity_term
        
        return free_energy

class NeuroModulationSystem(nn.Module):
    """
    神经调制系统 - 动态调整各层学习率
    输入: layer_errors (列表，包含各层误差张量)
    输出: modulation_factors (各层调制系数)
    """
    def __init__(self, num_layers, hidden_dim=64):
        super().__init__()
        self.num_layers = num_layers
        
        # 调制参数生成器
        self.modulator_generator = nn.Sequential(
            nn.Linear(num_layers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_layers),
            nn.Softmax(dim=-1)  # 确保调制系数和为1
        )

         

    def forward(self, layer_errors):
        """
        输入: layer_errors - 列表包含 num_layers 个张量，每个形状为 [B, T] (来自AdaptiveFreeEnergyCalculator)
        输出: modulation_factors [B, num_layers] - 每层的调制系数
        """
        # 1. 计算每层平均误差 (按时间维度平均)
        error_metrics = []
        for error in layer_errors:
            # 确保误差形状为 [B, T]
            assert error.dim() == 2, f"误差张量应为二维 [B, T]，实际为 {error.shape}"
            
            # 计算每个样本的平均误差 [B]
            mean_error = torch.mean(error, dim=1)  # [B]
            error_metrics.append(mean_error)
        
        # 2. 堆叠误差指标 [num_layers, B] -> 转置为 [B, num_layers]
        error_matrix = torch.stack(error_metrics, dim=1)  # [B, num_layers]
        
        # 3. 生成调制参数
        modulation_factors = self.modulator_generator(error_matrix)  # [B, num_layers]
        
        return modulation_factors
        
class EnhancedIterativePredictiveLayer(nn.Module):
    """
    增强版迭代预测层 - 整合所有改进
    输入: 
        x [B, T, D]
        memory_vector [B, M] (可选)
    输出: 
        final_prediction [B, T, D]
        iterations [B] (每个样本的实际迭代次数)
        predictions (迭代历史信息)
    """
    def __init__(self, input_dim=D, memory_dim=None, hidden_dim=256, 
                 min_iter=2, max_iter=8):
        super().__init__()
        self.input_dim = input_dim
        
        # 1. 记忆适配器（如果提供记忆维度）
        if memory_dim:
            self.memory_adapter = nn.Linear(memory_dim, input_dim)
        else:
            self.memory_adapter = None
            
        # 2. 生成模型 - 处理整个时间序列
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
        self.iter_controller = DynamicIterationController(
            input_dim=input_dim,
            min_iter=min_iter,
            max_iter=max_iter
        )
        
        # 6. 自适应自由能计算
        self.free_energy_calc = AdaptiveFreeEnergyCalculator()
        
        # 7. 内部表示更新率
        self.internal_lr = nn.Parameter(torch.tensor(0.1))
        
        # 8. 收敛检测 (处理时间序列)
        self.convergence_detector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        

    def forward(self, x, memory_vector=None):
        """
        输入: 
            x [B, T, D]
            memory_vector [B, M] (可选)
        输出: 
            final_prediction [B, T, D]
            iterations [B]
            predictions (迭代历史信息列表)
        """
        # 验证输入维度
        assert x.dim() == 3 and x.shape[-1] == self.input_dim, \
            f"输入应为 [B, T, D] 形状, 实际为 {x.shape}"
        
        # 合并记忆信息
        if memory_vector is not None and self.memory_adapter:
            mem_info = self.memory_adapter(memory_vector)  # [B, M] -> [B, D]
            mem_info = mem_info.unsqueeze(1)  # [B, 1, D]
            adapted_x = x + mem_info
        else:
            adapted_x = x
        
        # 初始预测
        initial_prediction = self.generative_model(adapted_x)
        
        # 初始误差
        initial_error = adapted_x - initial_prediction
        
        # 动态确定迭代次数 (每个样本单独)
        iterations = self.iter_controller(initial_error)  # [B]
        
        # 迭代推理
        current_belief = adapted_x.clone()
        all_predictions = []  # 存储每次迭代的结果
        
        # 最大迭代次数用于循环
        max_iter = torch.max(iterations).item()
        
        # 创建掩码，标记已完成迭代的样本
        completed_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        
        for i in range(max_iter):
            # 只处理未完成的样本
            active_idx = torch.where(~completed_mask)[0]
            if len(active_idx) == 0:
                break
                
            # 生成预测 (所有样本)
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
            
            # 检查收敛性 (所有时间步的平均收敛概率)
            convergence_prob = self.convergence_detector(prediction_error)  # [B, T, 1]
            mean_convergence = torch.mean(convergence_prob, dim=1)  # [B, 1]
            converged = mean_convergence.squeeze() > 0.85  # [B]
            
            # 更新belief (只更新未完成样本)
            delta = enhanced_prediction[active_idx] - current_belief[active_idx]
            current_belief[active_idx] = current_belief[active_idx] + self.internal_lr * delta
            
            # 存储迭代结果 (只存储当前迭代的样本)
            iter_result = {
                'prediction': prediction[active_idx],
                'enhanced_prediction': enhanced_prediction[active_idx],
                'error': prediction_error[active_idx],
                'precision': precision[active_idx],
                'free_energy': free_energy[active_idx],
                'converged': converged[active_idx]
            }
            iter_result.update({k: v[active_idx] for k, v in precision_data.items()})
            
            all_predictions.append(iter_result)
            
            # 更新完成状态
            # 条件1: 达到分配的迭代次数
            reached_iter = (i + 1) >= iterations[active_idx]
            # 条件2: 所有时间步收敛
            all_converged = converged[active_idx]
            # 完成条件: 达到迭代次数或完全收敛
            done = reached_iter | all_converged
            
            # 更新完成掩码
            completed_mask[active_idx] = done
            
            # 提前退出检查
            if completed_mask.all():
                break
        
        # 获取最终预测
        final_prediction = current_belief
        
        return {
            'iterations': iterations,
            'predictions': all_predictions,
            'final_prediction': final_prediction
        }
    
class PredictiveCodingAnalyzer:
    """
    预测编码分析工具 - 提供可视化和诊断功能
    适配全局维度约定: B=batch_size, T=seq_len, D=input_dim=384
    """
    def __init__(self, model):
        self.model = model
        self.activation_history = {}
    
    def register_safe_hook(self, name: str, module: nn.Module):
        """
        为指定模块注册一个安全的 forward hook，自动保存激活值
        """
        def hook(module, inputs, output):
            try:
                if isinstance(output, torch.Tensor):
                    self.activation_history[name] = output.detach().cpu()
                elif isinstance(output, (tuple, list)):
                    self.activation_history[name] = [
                        o.detach().cpu() if isinstance(o, torch.Tensor) else o
                        for o in output
                    ]
                elif isinstance(output, dict):
                    self.activation_history[name] = {
                        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in output.items()
                    }
                else:
                    self.activation_history[name] = output
            except Exception as e:
                print(f"[Hook Error @ {name}] {e}")
                self.activation_history[name] = None
        
        module.register_forward_hook(hook)
    
    def register_hooks(self):
        """注册前向钩子以捕获激活状态"""
        # 为关键模块注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.LSTM, nn.MultiheadAttention)):
                self.register_safe_hook(name, module)
                
    def visualize_prediction_flow(self, inputs):
        """
        可视化预测流和误差传播
        输入: inputs [B, T, D]
        输出: matplotlib Figure 对象
        """
        # 验证输入维度
        assert inputs.dim() == 3, f"输入应为三维 [B, T, D], 实际为 {inputs.shape}"
        
        # 注册钩子并运行前向传播
        self.register_hooks()
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. 误差分布图 (跨时间步和特征)
        if 'all_results' in outputs:
            errors = []
            for layer_result in outputs['all_results']:
                # 取最后一层最后一个时间步的误差
                if 'predictions' in layer_result and layer_result['predictions']:
                    last_pred = layer_result['predictions'][-1]
                    # 沿特征维度平均
                    mean_error = torch.mean(last_pred['error'], dim=-1)  # [B, T]
                    errors.append(mean_error.cpu().numpy())
            
            # 计算每层的平均误差
            layer_errors = [np.mean(e) for e in errors]
            
            axes[0].plot(layer_errors, marker='o')
            axes[0].set_title("Mean Prediction Error Across Layers")
            axes[0].set_xlabel("Layer Index")
            axes[0].set_ylabel("Average Error Magnitude")
            axes[0].grid(True)
        
        # 2. 自由能变化图 (随时间步变化)
        if 'all_results' in outputs and outputs['all_results']:
            # 取第一层的结果
            first_layer = outputs['all_results'][0]
            if 'predictions' in first_layer and first_layer['predictions']:
                # 获取最后一次迭代的自由能
                free_energy = first_layer['predictions'][-1]['free_energy']  # [B, T]
                # 计算批次平均
                mean_free_energy = torch.mean(free_energy, dim=0).cpu().numpy()  # [T]
                
                axes[1].plot(mean_free_energy, marker='o')
                axes[1].set_title("Free Energy Over Time Steps")
                axes[1].set_xlabel("Time Step")
                axes[1].set_ylabel("Average Free Energy")
                axes[1].grid(True)
        
        return fig
    
    def uncertainty_heatmap(self, inputs, time_step=-1, num_samples=20):
        """
        生成预测不确定性热力图
        输入: 
            inputs [B, T, D]
            time_step: 要分析的时间步 (默认最后一个)
            num_samples: 采样次数
        输出: matplotlib Figure 对象
        """
        # 验证输入维度
        assert inputs.dim() == 3, f"输入应为三维 [B, T, D], 实际为 {inputs.shape}"
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model(inputs)
                # 获取最终预测 [B, T, D]
                final_pred = outputs['final_prediction']
                # 选择特定时间步
                pred_at_step = final_pred[:, time_step, :]  # [B, D]
                predictions.append(pred_at_step)
        
        # 堆叠预测结果 [num_samples, B, D]
        pred_tensor = torch.stack(predictions)
        # 计算方差 [B, D]
        variance = torch.var(pred_tensor, dim=0).cpu().numpy()
        
        # 可视化
        plt.figure(figsize=(12, 8))
        sns.heatmap(variance, cmap='viridis', 
                    xticklabels=50, yticklabels=10)
        plt.title(f"Predictive Uncertainty at Time Step {time_step}")
        plt.xlabel("Feature Dimension")
        plt.ylabel("Batch Index")
        plt.colorbar(label='Variance')
        return plt
    
    def activation_tsne(self, inputs, layer_name, time_step=-1):
        """
        使用t-SNE可视化激活状态
        输入: 
            inputs [B, T, D]
            layer_name: 要可视化的层名称
            time_step: 要分析的时间步 (默认最后一个)
        输出: matplotlib Figure 对象
        """
        # 注册钩子
        self.register_hooks()
        with torch.no_grad():
            self.model(inputs)
        
        # 获取指定层的激活状态
        if layer_name in self.activation_history:
            activations = self.activation_history[layer_name]  # [B, ...]
            
            # 处理不同模块类型的激活状态
            if activations.dim() == 3:  # 线性层/注意力输出
                # 选择特定时间步
                act_at_step = activations[:, time_step, :]  # [B, D]
            elif activations.dim() == 2:  # 池化后输出
                act_at_step = activations
            else:
                raise ValueError(f"不支持的激活维度: {activations.shape}")
            
            # 使用t-SNE降维
            tsne = TSNE(n_components=2, perplexity=min(30, act_at_step.size(0)-1))
            embeddings = tsne.fit_transform(act_at_step.cpu().numpy())
            
            # 可视化
            plt.figure(figsize=(10, 8))
            plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6)
            plt.title(f"t-SNE of Activations: {layer_name} at t={time_step}")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            return plt
        else:
            print(f"警告: 未找到层 {layer_name} 的激活记录")
            return None
    
    def temporal_attention_visualization(self, inputs, layer_name="attention"):
        """
        可视化时间注意力权重
        输入: 
            inputs [B, T, D]
            layer_name: 注意力层名称
        输出: matplotlib Figure 对象
        """
        # 注册钩子
        self.register_hooks()
        with torch.no_grad():
            self.model(inputs)
        
        # 获取注意力权重
        if layer_name in self.activation_history:
            # 假设注意力权重是元组的第一个元素
            attn_weights = self.activation_history[layer_name][0]  # [B, num_heads, T, T]
            
            # 取第一个样本和第一个注意力头
            sample_attn = attn_weights[0, 0].cpu().numpy()
            
            # 可视化
            plt.figure(figsize=(10, 8))
            sns.heatmap(sample_attn, cmap="YlGnBu")
            plt.title(f"Temporal Attention Weights: {layer_name}")
            plt.xlabel("Key Time Step")
            plt.ylabel("Query Time Step")
            plt.colorbar(label='Attention Weight')
            return plt
        else:
            print(f"警告: 未找到层 {layer_name} 的注意力权重")
            return None
    
    def temporal_attention_visualization(self, inputs, layer_name="attention"):
        """
        可视化时间注意力权重
        输入: 
            inputs [B, T, D]
            layer_name: 注意力层名称
        输出: matplotlib Figure 对象
        """
        # 注册钩子
        hooks = self.register_hooks()
        with torch.no_grad():
            self.model(inputs)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 获取注意力权重
        if layer_name in self.activation_history:
            # 假设注意力权重是元组的第一个元素
            attn_weights = self.activation_history[layer_name][0]  # [B, num_heads, T, T]
            
            # 取第一个样本和第一个注意力头
            sample_attn = attn_weights[0, 0].cpu().numpy()
            
            # 可视化
            plt.figure(figsize=(10, 8))
            sns.heatmap(sample_attn, cmap="YlGnBu")
            plt.title(f"Temporal Attention Weights: {layer_name}")
            plt.xlabel("Key Time Step")
            plt.ylabel("Query Time Step")
            plt.colorbar(label='Attention Weight')
            return plt
        else:
            print(f"警告: 未找到层 {layer_name} 的注意力权重")
            return None
        
class DimensionAdapter(nn.Module):
    """
    统一的维度适配层 - 根据目标模块自动调整维度
    全局约定: B=batch_size, T=seq_len, D=input_dim=384
    支持模块: 'temporal', 'scale', 'predictive', 'fusion', 'precision'
    """
    def __init__(self):
        super().__init__()
        # 维度记录器
        self.dim_history = {}
        
        self.register_safe_hook("temporal_lstm_output", self.lstm)

    def forward(self, x, target_module):
        """
        输入: x (任意维度)
        输出: 适配后的张量，符合目标模块的维度要求
        """
        # 保存原始维度和目标模块
        self.dim_history['original'] = x.shape
        self.dim_history['target'] = target_module
        
        # 根据目标模块处理维度
        if target_module == 'temporal':
            # TemporalPredictiveLayer 需要 [B, T, D]
            if x.dim() == 2:
                # 假设是 [B, D] -> 添加时间维度 [B, 1, D]
                x = x.unsqueeze(1)
            elif x.dim() == 3 and x.shape[1] == 1:
                # 已经是 [B, 1, D] 无需处理
                pass
            else:
                raise ValueError(f"无法适配到temporal模块的维度: 当前 {x.shape}, 需要 [B, T, D]")
            return x
        
        elif target_module == 'scale':
            # MultiScaleProcessor 需要 [B, D, T] (卷积友好格式)
            if x.dim() == 3:
                # [B, T, D] -> [B, D, T]
                x = x.permute(0, 2, 1)
            elif x.dim() == 2:
                # [B, D] -> [B, D, 1]
                x = x.unsqueeze(-1)
            else:
                raise ValueError(f"无法适配到scale模块的维度: 当前 {x.shape}, 需要 [B, D, T]")
            return x
        
        elif target_module == 'predictive':
            # EnhancedIterativePredictiveLayer 需要 [B, T, D]
            if x.dim() == 3:
                # 已经是 [B, T, D] 格式
                return x
            elif x.dim() == 2:
                # [B, D] -> [B, 1, D]
                return x.unsqueeze(1)
            else:
                raise ValueError(f"无法适配到predictive模块的维度: 当前 {x.shape}, 需要 [B, T, D]")
        
        elif target_module == 'fusion':
            # AttentivePredictionFusion 需要 [B, T, D]
            if x.dim() == 3:
                return x
            elif x.dim() == 2:
                return x.unsqueeze(1)  # [B, D] -> [B, 1, D]
            else:
                raise ValueError(f"无法适配到fusion模块的维度: 当前 {x.shape}, 需要 [B, T, D]")
        
        elif target_module == 'precision':
            # BayesianPrecisionNetwork 需要 [B, T, D]
            if x.dim() == 3:
                return x
            elif x.dim() == 2:
                return x.unsqueeze(1)  # [B, D] -> [B, 1, D]
            else:
                raise ValueError(f"无法适配到precision模块的维度: 当前 {x.shape}, 需要 [B, T, D]")
        
        elif target_module == 'free_energy':
            # AdaptiveFreeEnergyCalculator 需要 [B, T] (自由能输出)
            if x.dim() == 2:
                return x
            elif x.dim() == 1:
                return x.unsqueeze(1)  # [B] -> [B, 1]
            else:
                # 尝试减少维度
                if x.dim() > 2:
                    return x.mean(dim=-1)  # 取特征维度平均
                raise ValueError(f"无法适配到free_energy模块的维度: 当前 {x.shape}, 需要 [B, T]")
        
        else:
            raise ValueError(f"未知目标模块: {target_module}")

    def restore(self, x, original_shape=None):
        """
        恢复原始维度
        输入: 适配后的张量
        输出: 恢复原始维度的张量
        """
        if original_shape is None:
            original_shape = self.dim_history.get('original')
        
        if original_shape is None:
            raise RuntimeError("无法恢复维度: 没有保存原始维度信息")
        
        # 如果已经是目标形状，直接返回
        if tuple(x.shape) == original_shape:
            return x
        
        # 根据目标模块决定恢复策略
        target_module = self.dim_history.get('target')
        
        if target_module == 'scale':
            # 从 [B, D, T] 恢复为 [B, T, D]
            if x.dim() == 3:
                return x.permute(0, 2, 1)
        
        elif target_module == 'free_energy':
            # 从 [B, T] 恢复为原始形状
            if original_shape == (x.size(0),):
                return x.squeeze(1)  # [B, T] -> [B]
        
        # 默认恢复策略: 尝试重塑
        try:
            return x.view(original_shape)
        except:
            # 如果重塑失败，返回最接近的形状
            if x.dim() == 3 and original_shape == 2:
                return x.squeeze(1)
            elif x.dim() == 2 and original_shape == 3:
                return x.unsqueeze(1)
            else:
                raise RuntimeError(f"无法恢复原始维度: 适配后 {x.shape}, 原始 {original_shape}")

class AdvancedPredictiveCodingSystem(nn.Module):
    """
    高级预测编码系统 - 整合所有改进，统一维度接口
    全局维度约定: B=batch_size, T=seq_len, D=input_dim=384
    """
    def __init__(self, input_dim=D, layer_dims=[256, 192, 128], 
                 memory_dim=128, scales=[1, 2, 4]):
        super().__init__()
        self.input_dim = input_dim
        
        # 记忆编码器（如果启用）
        self.memory_encoder = None
        if memory_dim:
            self.memory_encoder = nn.Sequential(
                nn.Linear(input_dim, memory_dim),
                nn.ReLU(),
                nn.Linear(memory_dim, memory_dim)
            )
        
        # 创建层级结构
        self.layers = nn.ModuleList()
        current_dim = input_dim

       

        for i, hidden_dim in enumerate(layer_dims):
            # 第一层添加时间处理器
            temporal_processor = TemporalPredictiveLayer(
                input_dim=current_dim,
                hidden_dim=hidden_dim
            ) if i == 0 else None
            
            # 所有层添加多尺度处理器（最后一层除外）
            scale_processor = MultiScaleProcessor(
                input_dim=current_dim,
                scales=scales,
                scale_hidden=hidden_dim // 2
            ) if i < len(layer_dims) - 1 else None
            
            # 核心预测层
            predictive_layer = EnhancedIterativePredictiveLayer(
                input_dim=current_dim,
                memory_dim=memory_dim if memory_dim else None,
                hidden_dim=hidden_dim,
                min_iter=2,
                max_iter=6
            )
            
            # 构建层块
            layer_block = nn.ModuleDict({
                'temporal_processor': temporal_processor,
                'scale_processor': scale_processor,
                'predictive_layer': predictive_layer
            })
            self.layers.append(layer_block)
            current_dim = hidden_dim
        
        # 神经调制系统
        self.neuromodulation = NeuroModulationSystem(len(layer_dims))
        
        # 输出层 - 恢复到原始输入维度
        self.output_layer = nn.Sequential(
            nn.Linear(layer_dims[-1], input_dim),
            nn.Tanh()
        )
        
        # 残差连接
        self.residual = nn.Identity()
    
    def forward(self, x):
        """
        输入: x [B, T, D]
        输出: 
            output [B, T, D]
            all_results (各层结果)
            modulation [B, num_layers] 调制系数
        """
        # 验证输入维度
        assert x.dim() == 3 and x.shape[-1] == self.input_dim, \
            f"输入应为 [B, T, D] 形状, 实际为 {x.shape}"
        
        # 保存原始输入用于残差连接
        original_x = x
        
        # 生成记忆向量（如果启用）
        memory_vector = None
        if self.memory_encoder:
            # 沿时间维度平均作为记忆输入
            time_avg = torch.mean(x, dim=1)  # [B, D]
            memory_vector = self.memory_encoder(time_avg)  # [B, memory_dim]
        
        # 分层处理
        all_results = []
        layer_errors = []
        
        for i, layer_block in enumerate(self.layers):
            # 时间处理（仅第一层）
            if layer_block['temporal_processor']:
                x = layer_block['temporal_processor'](x)['final_prediction']  # [B, T, D]
            
            # 多尺度处理（所有层除最后一层）
            if layer_block['scale_processor']:
                x = layer_block['scale_processor'](x)  # [B, T, D]
            
            # 预测编码层（核心处理）
            prediction_result = layer_block['predictive_layer'](
                x, 
                memory_vector=memory_vector
            )
            x = prediction_result['final_prediction']  # [B, T, D]
            
            # 保存结果和误差
            all_results.append(prediction_result)
            
            # 获取最后一层最后一个时间步的误差
            if 'predictions' in prediction_result and prediction_result['predictions']:
                last_pred = prediction_result['predictions'][-1]
                # 取最后一个时间步的误差
                last_error = last_pred['error'][:, -1, :]  # [B, D]
                layer_errors.append(last_error)
            else:
                layer_errors.append(torch.zeros(x.size(0), device=x.device))
        
        # 应用神经调制系统
        modulation = self.neuromodulation(layer_errors)  # [B, num_layers]
        
        # 最终输出（带残差连接）
        output = self.output_layer(x) + self.residual(original_x)
        
        return {
            'output': output,  # [B, T, D]
            'all_results': all_results,
            'modulation': modulation,
            'memory_vector': memory_vector
        }
    
    def predict(self, input_sequence):
        """
        统一预测接口
        输入: input_sequence [B, T, D]
        输出: predictions [B, T, D]
        """
        # 自动处理不同维度输入
        if input_sequence.dim() == 2:
            # 单个时间序列 [T, D] -> 添加批次维度
            input_sequence = input_sequence.unsqueeze(0)
        elif input_sequence.dim() == 3:
            # 已经是 [B, T, D]
            pass
        else:
            raise ValueError("输入应为 [B, T, D] 或 [T, D]")
        
        # 验证特征维度
        if input_sequence.size(-1) != self.input_dim:
            raise ValueError(f"最后维度应为 {self.input_dim}, 实际为 {input_sequence.size(-1)}")
        
        with torch.no_grad():
            results = self.forward(input_sequence)
            return results['output']
        
class UnifiedTrainer:
    """预测编码系统统一训练器 - 适配全局维度 [B, T, D]"""
    
    def __init__(self, model, 
                 learning_rate=1e-3, 
                 memory_lr=5e-4, 
                 clip_value=1.0,
                 free_energy_weight=0.5,
                 kl_weight=0.1):
        self.model = model
        self.clip_value = clip_value
        self.free_energy_weight = free_energy_weight
        self.kl_weight = kl_weight
        
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
            patience=5,
            verbose=True
        )
        
        # 梯度缩放器 (用于混合精度训练)
        self.scaler = GradScaler()
        
        

    def compute_prediction_loss(self, predictions, targets):
        """
        计算预测损失 (支持 [B, T, D] 维度)
        输入: 
            predictions [B, T, D]
            targets [B, T, D]
        """
        # 维度验证
        assert predictions.dim() == 3 and targets.dim() == 3, \
            "预测和目标都应为三维 [B, T, D]"
        assert predictions.shape == targets.shape, \
            f"预测和目标形状不匹配: {predictions.shape} vs {targets.shape}"
        
        # 计算每个时间步的MSE
        return F.mse_loss(predictions, targets)
    
    def compute_total_free_energy(self, results):
        """计算系统的总自由能 (适配时间序列)"""
        total_free_energy = 0.0
        total_elements = 0
        
        # 遍历所有层的结果
        for layer_result in results.get('all_results', []):
            # 遍历该层的每次迭代
            for iter_data in layer_result.get('predictions', []):
                # 获取自由能 [B, T]
                free_energy = iter_data.get('free_energy', None)
                if free_energy is not None:
                    # 累加所有元素
                    total_free_energy += free_energy.sum().item()
                    total_elements += free_energy.numel()
        
        # 计算平均自由能
        return total_free_energy / total_elements if total_elements > 0 else 0.0
    
    def compute_kl_divergence(self, results):
        """计算KL散度正则化项 (适配时间序列)"""
        kl_loss = 0.0
        kl_elements = 0
        
        # 遍历所有层的结果
        for layer_result in results.get('all_results', []):
            # 遍历该层的每次迭代
            for iter_data in layer_result.get('predictions', []):
                # 获取精度参数
                mu = iter_data.get('mu', None)  # [B, T, 1]
                log_var = iter_data.get('log_var', None)  # [B, T, 1]
                
                if mu is not None and log_var is not None:
                    # 计算KL散度 (每个时间步)
                    kl_div = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp(), dim=[1, 2])
                    kl_loss += kl_div.sum().item()
                    kl_elements += kl_div.numel()
        
        return kl_loss / kl_elements if kl_elements > 0 else 0.0
    
    def train_step(self, inputs, targets, use_amp=True):
        """
        执行单次训练步骤 (支持混合精度)
        输入: inputs [B, T, D], targets [B, T, D]
        """
        # 维度验证
        assert inputs.dim() == 3 and inputs.shape[-1] == self.model.input_dim, \
            f"输入维度错误: 应为 [B, T, {self.model.input_dim}], 实际为 {inputs.shape}"
        assert targets.dim() == 3 and targets.shape[-1] == self.model.input_dim, \
            f"目标维度错误: 应为 [B, T, {self.model.input_dim}], 实际为 {targets.shape}"
        
        # 混合精度训练上下文
        with autocast(enabled=use_amp):
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算各种损失
            prediction_loss = self.compute_prediction_loss(outputs['output'], targets)
            free_energy = self.compute_total_free_energy(outputs)
            kl_divergence = self.compute_kl_divergence(outputs)
            
            # 组合损失
            total_loss = prediction_loss + \
                         self.free_energy_weight * free_energy + \
                         self.kl_weight * kl_divergence
        
        # 优化器清零梯度
        self.optimizer.zero_grad()
        if self.memory_optimizer:
            self.memory_optimizer.zero_grad()
        
        # 混合精度反向传播
        if use_amp:
            self.scaler.scale(total_loss).backward()
            
            # 混合精度梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            
            # 更新参数
            self.scaler.step(self.optimizer)
            if self.memory_optimizer:
                self.scaler.step(self.memory_optimizer)
            self.scaler.update()
        else:
            # 标准反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            
            # 更新参数
            self.optimizer.step()
            if self.memory_optimizer:
                self.memory_optimizer.step()
        
        # 更新学习率
        self.scheduler.step(total_loss.item())
        
        return {
            'total_loss': total_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'free_energy': free_energy,
            'kl_divergence': kl_divergence
        }
    
    def evaluate(self, inputs, targets):
        """评估模型性能 (无梯度)"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            
            # 计算损失
            prediction_loss = self.compute_prediction_loss(outputs['output'], targets).item()
            free_energy = self.compute_total_free_energy(outputs)
            kl_divergence = self.compute_kl_divergence(outputs)
            
            # 计算序列预测准确率
            accuracy = self.compute_sequence_accuracy(outputs['output'], targets)
            
            return {
                'prediction_loss': prediction_loss,
                'free_energy': free_energy,
                'kl_divergence': kl_divergence,
                'accuracy': accuracy,
                'memory_vector': outputs.get('memory_vector')
            }
    
    def compute_sequence_accuracy(self, predictions, targets, threshold=0.5):
        """
        计算序列预测准确率
        输入: predictions [B, T, D], targets [B, T, D]
        输出: 准确率标量
        """
        # 计算每个时间步的正确率
        correct = (torch.abs(predictions - targets) < threshold).float()
        
        # 沿特征维度平均
        feature_accuracy = torch.mean(correct, dim=-1)  # [B, T]
        
        # 沿时间步平均
        time_accuracy = torch.mean(feature_accuracy, dim=-1)  # [B]
        
        # 沿批次平均
        return torch.mean(time_accuracy).item()
    
class TestPredictiveCodingSystem(unittest.TestCase):
    """预测编码系统测试套件 - 统一维度 [B, T, D=384]"""
    
    def setUp(self):
        """初始化测试环境"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 全局维度参数
        self.B = 8  # 批处理大小
        self.T = 20  # 序列长度
        self.D = 384  # 特征维度
        
        # 生成正弦波序列数据
        t = np.linspace(0, 4*np.pi, self.T)
        self.data = np.zeros((self.B, self.T, self.D))
        for i in range(self.B):
            for j in range(self.D):
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
        """测试记忆编码器维度一致性"""
        # 创建编码器
        encoder = AdvancedPredictiveCodingSystem(
            input_size=self.D,
            hidden_size=128,
            memory_dim=64
        )
        
        # 前向传播
        memory_vector = encoder(self.inputs)  # 输入 [B, T, D]
        
        # 验证输出维度
        self.assertEqual(memory_vector.shape, (self.B, 64))
        
        # 验证反向传播
        loss = memory_vector.sum()
        loss.backward()
        for name, param in encoder.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
    
    def test_iterative_predictive_layer(self):
        """测试迭代预测层维度一致性"""
        # 创建预测层
        layer = EnhancedIterativePredictiveLayer(
            input_dim=self.D,
            hidden_dim=32,
            min_iter=2,
            max_iter=5
        )
        
        # 测试单时间步
        sample = self.inputs[0]  # 第一个样本 [T, D]
        results = layer(sample.unsqueeze(0))  # 添加批次维度 [1, T, D]
        
        # 验证输出结构
        self.assertIn('final_prediction', results)
        self.assertIn('iterations', results)
        self.assertIn('predictions', results)
        
        # 验证输出维度
        self.assertEqual(results['final_prediction'].shape, (1, self.T, self.D))
        
        # 测试批量处理
        batch = self.inputs  # [B, T, D]
        batch_results = layer(batch)
        self.assertEqual(batch_results['final_prediction'].shape, (self.B, self.T, self.D))
    
    def test_multi_scale_processor(self):
        """测试多尺度处理器维度一致性"""
        # 创建处理器
        processor = MultiScaleProcessor(
            input_dim=self.D,
            scales=[1, 2, 4]
        )
        
        # 测试处理
        sample = self.inputs[0]  # [T, D]
        output = processor(sample.unsqueeze(0))  # [1, T, D] -> [1, T, D]
        
        # 验证输出维度
        self.assertEqual(output.shape, (1, self.T, self.D))
    
    def test_dynamic_iteration_controller(self):
        """测试动态迭代控制器维度一致性"""
        # 创建控制器
        controller = DynamicIterationController(
            input_dim=self.D,
            min_iter=2,
            max_iter=8
        )
        
        # 测试不同误差水平
        small_error = torch.randn(4, self.T, self.D) * 0.1  # [4, T, D]
        large_error = torch.randn(4, self.T, self.D) * 1.0
        
        iter_small = controller(small_error)
        iter_large = controller(large_error)
        
        # 验证输出维度
        self.assertEqual(iter_small.shape, (4,))
        
        # 验证迭代次数范围
        self.assertTrue(torch.all(iter_small >= 2))
        self.assertTrue(torch.all(iter_small <= 8))
        
        # 验证大误差需要更多迭代
        self.assertTrue(torch.all(iter_large > iter_small))
    
    def test_full_system_forward(self):
        """测试完整系统前向传播维度一致性"""
        # 创建系统
        system = AdvancedPredictiveCodingSystem(
            input_dim=self.D,
            layer_dims=[128, 96, 64],
            memory_dim=64
        )
        
        # 前向传播
        outputs = system(self.inputs)  # 输入 [B, T, D]
        
        # 验证输出结构
        self.assertIn('output', outputs)
        self.assertIn('all_results', outputs)
        self.assertIn('modulation', outputs)
        self.assertIn('memory_vector', outputs)
        
        # 验证输出形状
        self.assertEqual(outputs['output'].shape, (self.B, self.T, self.D))
        self.assertEqual(outputs['memory_vector'].shape, (self.B, 64))
        self.assertEqual(outputs['modulation'].shape, (self.B, 3))  # 3层
    
    def test_training_loop(self):
        """测试训练循环维度一致性"""
        # 创建系统和训练器
        system = AdvancedPredictiveCodingSystem(
            input_dim=self.D,
            layer_dims=[96, 64],
            memory_dim=64
        )
        trainer = UnifiedTrainer(system, learning_rate=1e-3)
        
        # 初始评估
        eval_start = trainer.evaluate(self.inputs, self.targets)
        
        # 训练一个epoch
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            # 验证数据批次维度
            self.assertEqual(inputs.shape[-1], self.D)
            self.assertEqual(targets.shape[-1], self.D)
            
            # 训练步骤
            train_stats = trainer.train_step(inputs, targets)
            
            # 验证训练统计
            self.assertIn('total_loss', train_stats)
            self.assertIn('prediction_loss', train_stats)
            
            if batch_idx >= 2:  # 运行几个批次
                break
        
        # 训练后评估
        eval_end = trainer.evaluate(self.inputs, self.targets)
        
        # 验证损失下降
        self.assertLess(eval_end['prediction_loss'], eval_start['prediction_loss'])
    
    def test_convergence_detection(self):
        """测试收敛检测机制维度一致性"""
        # 创建预测层
        layer = EnhancedIterativePredictiveLayer(
            input_dim=self.D,
            hidden_dim=32,
            min_iter=2,
            max_iter=10
        )
        
        # 测试收敛情况
        convergent_input = torch.zeros(1, self.T, self.D)  # 零输入 [1, T, D]
        results = layer(convergent_input)
        
        # 验证提前终止
        self.assertTrue(torch.all(results['iterations'] < 10))
    
    def test_uncertainty_quantification(self):
        """测试不确定性量化维度一致性"""
        # 创建精度网络
        precision_net = BayesianPrecisionNetwork(
            input_dim=self.D,
            hidden_dim=16
        )
        
        # 生成测试误差
        error = torch.randn(4, self.T, self.D)  # [4, T, D]
        
        # 前向传播
        outputs = precision_net(error)
        
        # 验证输出结构
        self.assertIn('precision', outputs)
        self.assertIn('mu', outputs)
        self.assertIn('log_var', outputs)
        
        # 验证输出维度
        self.assertEqual(outputs['precision'].shape, (4, self.T, 1))
    
    def test_analyzer_tool(self):
        """测试分析工具维度一致性"""
        # 创建系统和分析器
        system = AdvancedPredictiveCodingSystem(
            input_dim=self.D,
            layer_dims=[96, 64]
        )
        analyzer = PredictiveCodingAnalyzer(system)
        
        # 测试预测流可视化
        fig = analyzer.visualize_prediction_flow(self.inputs[:2])  # 输入 [2, T, D]
        self.assertIsInstance(fig, plt.Figure)
        
        # 测试不确定性热力图
        heatmap = analyzer.uncertainty_heatmap(self.inputs[:2], time_step=10)
        self.assertIsInstance(heatmap, plt.Figure)
    
    def test_performance_benchmark(self):
        """性能基准测试维度一致性"""
        # 创建系统
        system = AdvancedPredictiveCodingSystem(
            input_dim=self.D,
            layer_dims=[128, 96, 64]
        )
        
        # 预热
        with torch.no_grad():
            _ = system(self.inputs[:1])
        
        # 测试推理速度
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = system(self.inputs[:4])  # [4, T, D]
        inference_time = (time.time() - start_time) / 10
        
        # 测试训练速度
        trainer = UnifiedTrainer(system)
        start_time = time.time()
        for inputs, targets in self.dataloader:  # 数据维度 [4, T, D]
            trainer.train_step(inputs, targets)
            break  # 只测试一个批次
        training_time = time.time() - start_time
        
        # 内存占用测试
        param_size = sum(p.numel() * p.element_size() for p in system.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in system.buffers())
        model_size = (param_size + buffer_size) / (1024 ** 2)  # MB
        
        print(f"平均推理时间 (batch=4): {inference_time:.4f}s")
        print(f"单批次训练时间: {training_time:.4f}s")
        print(f"模型内存占用: {model_size:.2f} MB")
        
        # 验证性能在可接受范围内
        self.assertLess(inference_time, 0.5)  # 500ms内完成推理
        self.assertLess(training_time, 1.0)  # 1秒内完成一个批次训练
        self.assertLess(model_size, 50)  # 小于50MB

if __name__ == '__main__':
    unittest.main(verbosity=2)
