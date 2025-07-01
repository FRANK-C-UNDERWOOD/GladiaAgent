"""
🧠 seRNN 模块：Spatially-Embedded Recurrent Neural Network
Author: DOCTOR + 歌蕾蒂娅 (2025)

模块用途：
- 在 RNN 中加入“神经元空间位置”作为连接结构限制
- 实现空间稀疏性约束，更符合生物神经网络的连接模式
- 可用于 Agent 空间导航记忆、脑连接模拟、图式记忆建构

主要组件：
1. seRNNCell       - 单个时间步的带空间惩罚的 RNN 单元
2. seRNN           - 多步序列建模的循环网络结构
3. spatial_regularizer - 连接距离正则项（用于加权 loss）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math

class PredictiveCodingLayer(nn.Module):
    """预测编码层 - 实现预测误差计算和自上而下的预测"""
    
    def __init__(self, input_size: int, hidden_size: int, prediction_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prediction_size = prediction_size
        
        # 预测网络 (自上而下) - 修复维度匹配问题
        self.prediction_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, prediction_size)
        )
        
        # 误差网络 (自下而上) - 确保输入输出维度正确
        self.error_net = nn.Sequential(
            nn.Linear(prediction_size, hidden_size // 2),  # 减少维度避免过拟合
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # 不确定性估计
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softplus()
        )
        
        # 输入投影层 - 处理维度不匹配
        if input_size != prediction_size:
            self.input_projection = nn.Linear(input_size, prediction_size)
        else:
            self.input_projection = nn.Identity()
        
    def forward(self, input_data: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 投影输入到正确维度
        projected_input = self.input_projection(input_data)
        
        # 生成预测
        prediction = self.prediction_net(hidden_state)
        
        # 计算预测误差 - 现在维度匹配
        prediction_error = projected_input - prediction
        
        # 处理误差信号
        error_signal = self.error_net(prediction_error)
        
        # 估计不确定性
        uncertainty = self.uncertainty_net(hidden_state)
        
        return prediction, error_signal, uncertainty

class SelectiveGatingMechanism(nn.Module):
    """选择性门控机制 - 增强版，包含预测误差驱动的注意力"""
    
    def __init__(self, input_size: int, hidden_size: int, spatial_dim: Optional[int] = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.spatial_dim = spatial_dim
        
        # 输入投影层 - 确保维度匹配
        self.input_projection = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        
        # 传统门控 - 使用投影后的维度
        gate_input_size = hidden_size + hidden_size  # 投影后的输入 + 隐藏状态
        self.forget_gate = nn.Linear(gate_input_size, hidden_size)
        self.input_gate = nn.Linear(gate_input_size, hidden_size)
        self.candidate_gate = nn.Linear(gate_input_size, hidden_size)
        self.output_gate = nn.Linear(gate_input_size, hidden_size)
        
        # 预测误差驱动的注意力门控
        self.error_attention = nn.Linear(hidden_size, hidden_size)
        
        # 空间注意力 (如果提供了空间维度)
        if spatial_dim:
            self.spatial_attention = nn.Linear(hidden_size, spatial_dim)
        
        # 时间尺度自适应
        self.temporal_scaling = nn.Parameter(torch.ones(1))
        
    def forward(self, input_data: torch.Tensor, hidden_state: torch.Tensor, 
                cell_state: torch.Tensor, error_signal: torch.Tensor, 
                uncertainty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 投影输入到隐藏维度
        projected_input = self.input_projection(input_data)
        
        # 组合输入
        combined = torch.cat([projected_input, hidden_state], dim=-1)
        
        # 基本门控
        forget = torch.sigmoid(self.forget_gate(combined))
        input_gate = torch.sigmoid(self.input_gate(combined))
        candidate = torch.tanh(self.candidate_gate(combined))
        output = torch.sigmoid(self.output_gate(combined))
        
        # 误差驱动的注意力权重
        error_attention = torch.sigmoid(self.error_attention(error_signal))
        
        # 不确定性调节的门控
        uncertainty_modulated_input = input_gate * (1 + uncertainty.squeeze(-1))
        uncertainty_modulated_forget = forget * (1 - uncertainty.squeeze(-1) * 0.1)
        
        # 更新细胞状态
        cell_state = (uncertainty_modulated_forget * cell_state + 
                     uncertainty_modulated_input * candidate * error_attention)
        
        # 时间尺度自适应
        cell_state = cell_state * self.temporal_scaling
        
        # 输出门控
        hidden_state = output * torch.tanh(cell_state)
        
        return hidden_state, cell_state

class SpatialEmbeddingLayer(nn.Module):
    """空间嵌入层 - 处理空间结构信息"""
    
    def __init__(self, hidden_size: int, spatial_dim: int, embedding_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial_dim = spatial_dim
        self.embedding_dim = embedding_dim
        
        # 空间位置编码
        self.spatial_embedding = nn.Embedding(spatial_dim, embedding_dim)
        
        # 空间关系建模 - 确保维度匹配
        self.spatial_transform = nn.Linear(embedding_dim, hidden_size)
        
        # 距离衰减参数
        self.distance_decay = nn.Parameter(torch.tensor(1.0))
        
        # 层标准化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, spatial_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            spatial_positions: [batch_size, seq_len] - 空间位置索引
        """
        # 获取空间嵌入
        spatial_emb = self.spatial_embedding(spatial_positions)
        spatial_features = self.spatial_transform(spatial_emb)
        
        # 计算空间距离权重
        batch_size, seq_len, _ = hidden_states.shape
        distance_matrix = torch.abs(spatial_positions.unsqueeze(-1) - spatial_positions.unsqueeze(-2)).float()
        distance_weights = torch.exp(-distance_matrix * self.distance_decay)
        
        # 应用空间权重
        weighted_hidden = torch.bmm(distance_weights, hidden_states)
        
        # 结合空间特征
        enhanced_hidden = hidden_states + spatial_features + weighted_hidden * 0.1
        
        # 层标准化
        enhanced_hidden = self.layer_norm(enhanced_hidden)
        
        return enhanced_hidden

class SeRNN(nn.Module):
    """增强版seRNN - 整合预测编码框架，支持384维词嵌入"""
    
    def __init__(self, 
                 input_size: int = 384,  # 默认384维词嵌入
                 hidden_size: int = 384,  # 保持一致的隐藏维度
                 spatial_dim: int = 1000,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 hierarchical_scales: List[int] = [1, 4, 16],
                 prediction_size: Optional[int] = None,
                 device: str = 'cpu'):  # 新增device参数
        super().__init__()
        
        self.device = device  # 保存设备信息
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.spatial_dim = spatial_dim
        self.num_layers = num_layers
        self.hierarchical_scales = hierarchical_scales
        
        # 如果没有指定预测大小，使用输入大小
        self.prediction_size = prediction_size or input_size
        
        # 初始化各个层
        self.pc_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.pc_layers.append(
                PredictiveCodingLayer(layer_input_size, hidden_size, self.prediction_size)
            )
        
        # 选择性门控机制
        self.selective_gates = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.selective_gates.append(
                SelectiveGatingMechanism(layer_input_size, hidden_size, spatial_dim)
            )
        
        # 空间嵌入层
        self.spatial_embedding = SpatialEmbeddingLayer(
            hidden_size, spatial_dim, min(hidden_size // 2, 192)
        )
        
        # 层次化预测网络
        self.hierarchical_predictors = nn.ModuleList([nn.LSTM(hidden_size, hidden_size, batch_first=True)
                                                      for _ in hierarchical_scales])
        
        # 多尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(len(hierarchical_scales) * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.prediction_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.prediction_loss_weight = nn.Parameter(torch.tensor(1.0))
        self.uncertainty_loss_weight = nn.Parameter(torch.tensor(0.1))
        
        self.to(self.device)  # 这里将模型转移到指定设备上

        
    def forward(self, 
                input_sequence: torch.Tensor, 
                spatial_positions: torch.Tensor,
                hidden_states: Optional[List[torch.Tensor]] = None,
                cell_states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            input_sequence: [batch_size, seq_len, input_size] - 384维词嵌入
            spatial_positions: [batch_size, seq_len] - 空间位置
            hidden_states: List of hidden states for each layer
            cell_states: List of cell states for each layer
        """
        batch_size, seq_len, _ = input_sequence.shape
        device = input_sequence.device
        
        # 初始化状态
        if hidden_states is None:
            hidden_states = [torch.zeros(batch_size, self.hidden_size, device=device) 
                           for _ in range(self.num_layers)]
        if cell_states is None:
            cell_states = [torch.zeros(batch_size, self.hidden_size, device=device) 
                          for _ in range(self.num_layers)]
        
        # 存储输出和中间结果
        outputs = []
        all_predictions = []
        all_uncertainties = []
        prediction_errors = []
        
        # 逐时间步处理
        for t in range(seq_len):
            current_input = input_sequence[:, t, :]
            layer_input = current_input
            
            # 逐层处理
            for layer_idx in range(self.num_layers):
                # 预测编码
                prediction, error_signal, uncertainty = self.pc_layers[layer_idx](
                    layer_input, hidden_states[layer_idx])
                
                # 选择性门控
                hidden_states[layer_idx], cell_states[layer_idx] = self.selective_gates[layer_idx](
                    layer_input, hidden_states[layer_idx], cell_states[layer_idx], 
                    error_signal, uncertainty)
                
                # 应用dropout
                hidden_states[layer_idx] = self.dropout(hidden_states[layer_idx])
                
                # 为下一层准备输入
                layer_input = hidden_states[layer_idx]
                
                # 记录预测和不确定性（只记录最后一层）
                if layer_idx == self.num_layers - 1:
                    all_predictions.append(prediction)
                    all_uncertainties.append(uncertainty)
                    # 计算预测误差 - 确保维度匹配
                    if prediction.shape[-1] == current_input.shape[-1]:
                        pred_error = torch.abs(current_input - prediction).mean(dim=-1, keepdim=True)
                    else:
                        # 如果维度不匹配，使用投影后的误差
                        pred_error = torch.abs(error_signal).mean(dim=-1, keepdim=True)
                    prediction_errors.append(pred_error)
            
            # 层次化预测
            hierarchical_outputs = []
            for scale_idx, scale in enumerate(self.hierarchical_scales):
                if t % scale == 0:  # 按不同时间尺度采样
                    scale_input = hidden_states[-1].unsqueeze(1)
                    scale_output, _ = self.hierarchical_predictors[scale_idx](scale_input)
                    hierarchical_outputs.append(scale_output.squeeze(1))
                else:
                    hierarchical_outputs.append(torch.zeros_like(hidden_states[-1]))
            
            # 多尺度融合
            fused_output = self.scale_fusion(torch.cat(hierarchical_outputs, dim=-1))
            
            # 最终输出
            output = self.output_projection(fused_output)
            outputs.append(output)
        
        # 堆叠输出
        output_sequence = torch.stack(outputs, dim=1)
        
        # 空间嵌入处理（在序列结束后）
        if len(hidden_states) > 0:
            stacked_hidden = torch.stack([h.unsqueeze(1) for h in hidden_states], dim=2)  # [batch, 1, layers, hidden]
            stacked_hidden = stacked_hidden.squeeze(1)  # [batch, layers, hidden]
            
            # 为空间嵌入准备位置
            last_positions = spatial_positions[:, -1].unsqueeze(1).expand(-1, self.num_layers)
            enhanced_hidden = self.spatial_embedding(stacked_hidden, last_positions)
            
            # 更新最终隐藏状态
            for i in range(self.num_layers):
                hidden_states[i] = enhanced_hidden[:, i, :]
        
        # 计算损失组件
        if all_predictions and prediction_errors and all_uncertainties:
            prediction_loss = torch.mean(torch.stack(prediction_errors))
            uncertainty_loss = torch.mean(torch.stack(all_uncertainties))
            
            # 堆叠预测结果
            stacked_predictions = torch.stack(all_predictions, dim=1)
            stacked_uncertainties = torch.stack(all_uncertainties, dim=1)
            stacked_errors = torch.stack(prediction_errors, dim=1)
        else:
            # 如果没有预测结果，使用默认值
            prediction_loss = torch.tensor(0.0, device=device)
            uncertainty_loss = torch.tensor(0.0, device=device)
            stacked_predictions = torch.zeros(batch_size, seq_len, self.prediction_size, device=device)
            stacked_uncertainties = torch.zeros(batch_size, seq_len, 1, device=device)
            stacked_errors = torch.zeros(batch_size, seq_len, 1, device=device)
        
        # 返回结果
        return output_sequence, {
            'predictions': stacked_predictions,
            'uncertainties': stacked_uncertainties,
            'prediction_errors': stacked_errors,
            'prediction_loss': prediction_loss,
            'uncertainty_loss': uncertainty_loss,
            'total_loss': (self.prediction_loss_weight * prediction_loss + 
                          self.uncertainty_loss_weight * uncertainty_loss),
            'hidden_states': hidden_states,
            'cell_states': cell_states
        }

# 训练函数 - 修复verbose参数问题
class SeRNNTrainer:
    def __init__(self, model: SeRNN, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # 修复：移除verbose参数，因为不是所有PyTorch版本都支持
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5)
        
    def train_step(self, input_sequence: torch.Tensor, 
                   target_sequence: torch.Tensor,
                   spatial_positions: torch.Tensor) -> dict:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            # 前向传播
            output_sequence, auxiliary_outputs = self.model(input_sequence, spatial_positions)
            
            # 计算主要损失
            reconstruction_loss = F.mse_loss(output_sequence, target_sequence)
            
            # 总损失
            total_loss = (reconstruction_loss + 
                         auxiliary_outputs['prediction_loss'] * 0.5 +
                         auxiliary_outputs['uncertainty_loss'] * 0.1)
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            return {
                'total_loss': total_loss.item(),
                'reconstruction_loss': reconstruction_loss.item(),
                'prediction_loss': auxiliary_outputs['prediction_loss'].item(),
                'uncertainty_loss': auxiliary_outputs['uncertainty_loss'].item()
            }
            
        except Exception as e:
            print(f"训练步骤出错: {e}")
            print(f"输入形状: {input_sequence.shape}")
            print(f"目标形状: {target_sequence.shape}")
            print(f"空间位置形状: {spatial_positions.shape}")
            raise e
    
    def evaluate(self, input_sequence: torch.Tensor, 
                 target_sequence: torch.Tensor,
                 spatial_positions: torch.Tensor) -> dict:
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            output_sequence, auxiliary_outputs = self.model(input_sequence, spatial_positions)
            reconstruction_loss = F.mse_loss(output_sequence, target_sequence)
            
            return {
                'reconstruction_loss': reconstruction_loss.item(),
                'prediction_accuracy': 1.0 - torch.mean(auxiliary_outputs['prediction_errors']).item(),
                'mean_uncertainty': torch.mean(auxiliary_outputs['uncertainties']).item()
            }
    
    def step_scheduler(self, val_loss: float):
        """手动调用学习率调度器"""
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(val_loss)
        new_lr = self.optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"学习率调整: {old_lr:.2e} -> {new_lr:.2e}")

# 使用示例 - 适配384维词嵌入
def create_sample_data(batch_size: int = 32, seq_len: int = 50, 
                      input_size: int = 384, spatial_dim: int = 1000):  # 384维词嵌入
    """创建示例数据"""
    # 生成输入序列 - 模拟384维词嵌入
    input_sequence = torch.randn(batch_size, seq_len, input_size) * 0.1  # 较小的随机值
    
    # 生成空间位置（随机）
    spatial_positions = torch.randint(0, spatial_dim, (batch_size, seq_len))
    
    # 目标序列（下一个时间步的预测）
    target_sequence = torch.cat([input_sequence[:, 1:, :], 
                                torch.randn(batch_size, 1, input_size) * 0.1], dim=1)
    
    return input_sequence, target_sequence, spatial_positions

# 主函数
def main():
    # 模型参数 - 适配384维词嵌入
    config = {
        'input_size': 384,        # 384维词嵌入
        'hidden_size': 384,       # 保持一致的隐藏维度
        'spatial_dim': 1000,      # 扩大空间维度
        'num_layers': 3,
        'dropout': 0.1,
        'hierarchical_scales': [1, 4, 16],
        'prediction_size': 384    # 预测输出也是384维
    }
    
    print("创建增强版seRNN模型（384维词嵌入）...")
    model = SeRNN(**config)
    trainer = SeRNNTrainer(model, learning_rate=0.001)
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练循环
    num_epochs = 50  # 减少epoch数用于测试
    print(f"开始训练 {num_epochs} 个epoch...")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        try:
            # 生成训练数据
            input_seq, target_seq, spatial_pos = create_sample_data()
            
            # 训练步骤
            train_metrics = trainer.train_step(input_seq, target_seq, spatial_pos)
            
            # 评估和学习率调整
            if epoch % 5 == 0:
                eval_metrics = trainer.evaluate(input_seq, target_seq, spatial_pos)
                current_loss = eval_metrics['reconstruction_loss']
                
                # 调用学习率调度器
                trainer.step_scheduler(current_loss)
                
                # 更新最佳损失
                if current_loss < best_loss:
                    best_loss = current_loss
                    print(f"★ 新的最佳损失: {best_loss:.6f}")
                
                print(f"Epoch {epoch:3d}:")
                print(f"  Train Loss: {train_metrics['total_loss']:.6f}")
                print(f"  Eval Loss: {eval_metrics['reconstruction_loss']:.6f}")
                print(f"  Prediction Accuracy: {eval_metrics['prediction_accuracy']:.4f}")
                print(f"  Mean Uncertainty: {eval_metrics['mean_uncertainty']:.6f}")
                print(f"  Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.2e}")
                print("-" * 50)
                
        except Exception as e:
            print(f"训练在epoch {epoch}时出错: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("训练成功完成!")
    return model, trainer

if __name__ == "__main__":
    try:
        model, trainer = main()
        print("模型训练完成！")
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'model_config': {
                'input_size': 384,
                'hidden_size': 384,
                'spatial_dim': 1000,
                'num_layers': 3,
                'dropout': 0.1,
                'hierarchical_scales': [1, 4, 16],
                'prediction_size': 384
            }
        }, 'enhanced_sernn_384dim.pth')
        
        print("模型已保存到 'enhanced_sernn_384dim.pth'")
        
        # 加载模型示例
        print("\n加载模型示例:")
        checkpoint = torch.load('enhanced_sernn_384dim.pth')
        config = checkpoint['model_config']
        loaded_model = SeRNN(**config)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        print("模型加载成功!")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
