"""
🧠 PredictiveCodingAgent 模块
Author: DOCTOR + 歌蕾蒂娅 (2025)

本模块实现一个具备感知预测、误差反向修正与记忆机制的基础预测编码 Agent。
该结构可用于主动感知、异常检测、新奇性记忆、自适应控制等场景。

核心功能：
1. encode_input()   - 将输入编码为隐状态向量
2. decode_prediction() - 从隐状态生成预测
3. forward_predict()   - 执行多轮预测-误差-修正闭环推理
4. update_memory()     - 将高预测误差的输入记入记忆库
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings

class ContinuousTimeNeuron(nn.Module):
    """
    连续时间神经元 - 修复批次维度问题版本
    基于预测编码理论的生物启发神经元模型
    """
    def __init__(self, input_size: int = 64, hidden_size: int = 128, 
                 memory_capacity: int = 1000, tau: float = 0.1, 
                 learning_rate: float = 0.01):
        super(ContinuousTimeNeuron, self).__init__()
        
        # 确保维度参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_capacity = memory_capacity
        self.tau = tau  # 时间常数
        self.learning_rate = learning_rate
        
        # 权重矩阵 - 维度明确定义
        self.W_input = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_recurrent = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_output = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        
        # 偏置项
        self.bias_hidden = nn.Parameter(torch.zeros(hidden_size))
        self.bias_output = nn.Parameter(torch.zeros(input_size))
        
        # 状态变量 - 修复：不再保存批次状态，每次前向传播时重新初始化
        self.hidden_size_dim = hidden_size
        
        # 记忆存储
        self.memory_bank = []
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化权重"""
        nn.init.xavier_uniform_(self.W_input)
        nn.init.xavier_uniform_(self.W_recurrent)
        nn.init.xavier_uniform_(self.W_output)
    
    def _init_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入张量 (batch_size, input_size)
            hidden_state: 可选的初始隐藏状态 (batch_size, hidden_size)
        Returns:
            prediction: 预测输出 (batch_size, input_size)
            new_hidden_state: 新的隐藏状态 (batch_size, hidden_size)
        """
        # 维度检查
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.size(0)
        device = x.device
        
        assert x.size(1) == self.input_size, \
            f"输入维度不匹配: 期望 {self.input_size}, 得到 {x.size(1)}"
        
        # 初始化或验证隐藏状态
        if hidden_state is None:
            current_hidden_state = self._init_hidden_state(batch_size, device)
        else:
            assert hidden_state.size() == (batch_size, self.hidden_size), \
                f"隐藏状态维度不匹配: 期望 ({batch_size}, {self.hidden_size}), 得到 {hidden_state.size()}"
            current_hidden_state = hidden_state
        
        # 计算新的隐藏状态
        input_contribution = torch.matmul(x, self.W_input.t())
        recurrent_contribution = torch.matmul(current_hidden_state, self.W_recurrent.t())
        
        # 连续时间动力学 (简化的欧拉方法)
        dh_dt = (-current_hidden_state + torch.tanh(
            input_contribution + recurrent_contribution + self.bias_hidden
        )) / self.tau
        
        new_hidden_state = current_hidden_state + dh_dt * 0.01  # 时间步长
        
        # 生成预测
        prediction = torch.matmul(new_hidden_state, self.W_output.t()) + self.bias_output
        
        return prediction, new_hidden_state
    
    def update_memory(self, memory_vector: torch.Tensor):
        """更新记忆库"""
        if len(self.memory_bank) >= self.memory_capacity:
            self.memory_bank.pop(0)
        self.memory_bank.append(memory_vector.detach().cpu())

class PredictiveCodingAgent(nn.Module):
    """
    预测编码智能体 - 修复批次维度问题版本
    整合多个连续时间神经元的预测编码网络
    """
    def __init__(self, num_inputs: int = 384, encoding_dim: int = 64, 
                 hidden_size: int = 128, num_neurons: int = 4,
                 memory_capacity: int = 1000):
        super(PredictiveCodingAgent, self).__init__()
        
        # 网络参数
        self.num_inputs = num_inputs
        self.encoding_dim = encoding_dim
        self.hidden_size = hidden_size
        self.num_neurons = num_neurons
        self.memory_capacity = memory_capacity
        
        # 编码器：384 -> 64
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, encoding_dim),
            nn.Tanh()  # 限制编码范围
        )
        
        # 连续时间神经元组
        self.neurons = nn.ModuleList([
            ContinuousTimeNeuron(
                input_size=encoding_dim,  # 与编码器输出对齐
                hidden_size=hidden_size,
                memory_capacity=memory_capacity
            ) for _ in range(num_neurons)
        ])
        
        # 解码器：64 -> 384
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_inputs)
        )
        
        # 注意力机制用于融合多个神经元输出
        self.attention = nn.MultiheadAttention(
            embed_dim=encoding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(encoding_dim)
        self.layer_norm2 = nn.LayerNorm(encoding_dim)
        
        # 记忆编码参数
        self.memory_embed_dim = 30  # 固定记忆向量维度
        self.memory_projection = nn.Linear(self.memory_embed_dim, encoding_dim)
        
        # 神经元状态管理 - 修复：使用字典存储不同形状的状态
        self.neuron_states = {}
    
    def _get_neuron_states(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """获取或创建神经元状态"""
        key = f"{batch_size}_{device}"
        if key not in self.neuron_states:
            self.neuron_states[key] = [
                torch.zeros(batch_size, self.hidden_size, device=device)
                for _ in range(self.num_neurons)
            ]
        return self.neuron_states[key]
    
    def encode_input(self, subject: Any, predicate: Any, obj: Any) -> torch.Tensor:
        """
        编码三元组输入为固定维度的记忆向量
        Args:
            subject, predicate, obj: 三元组元素
        Returns:
            encoded_memory: 固定30维的记忆向量
        """
        def pad_or_truncate_encode(text: Any, length: int = 10) -> List[float]:
            """将文本编码为固定长度的数值向量"""
            text_str = str(text)[:length]
            text_str = text_str.ljust(length, ' ')  # 用空格填充到固定长度
            return [float(ord(c)) / 127.0 for c in text_str]  # 归一化到[-1,1]
        
        # 编码每个元素为10维
        s_enc = pad_or_truncate_encode(subject, 10)
        p_enc = pad_or_truncate_encode(predicate, 10)
        o_enc = pad_or_truncate_encode(obj, 10)
        
        # 组合为30维向量
        memory_vector = torch.tensor(s_enc + p_enc + o_enc, dtype=torch.float32)
        
        # 维度检查
        assert memory_vector.shape[0] == self.memory_embed_dim, \
            f"记忆向量维度错误: 期望 {self.memory_embed_dim}, 得到 {memory_vector.shape[0]}"
        
        return memory_vector
    
    def predict_with_memory(self, memory_vector: torch.Tensor, 
                          context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        基于记忆向量生成预测
        Args:
            memory_vector: 记忆向量 (batch_size, memory_embed_dim)
            context: 可选的上下文信息 (batch_size, num_inputs)
        Returns:
            prediction: 预测结果 (batch_size, num_inputs)
        """
        if memory_vector.dim() == 1:
            memory_vector = memory_vector.unsqueeze(0)
        
        batch_size = memory_vector.size(0)
        device = memory_vector.device
        
        # 将记忆向量投影到编码维度
        encoded_memory = self.memory_projection(memory_vector)  # (batch_size, encoding_dim)
        
        # 如果有上下文，结合上下文信息
        if context is not None:
            # 维度检查
            assert context.size(-1) == self.num_inputs, \
                f"上下文维度不匹配: 期望 {self.num_inputs}, 得到 {context.size(-1)}"
            
            # 编码上下文
            encoded_context = self.encoder(context)  # (batch_size, encoding_dim)
            
            # 融合记忆和上下文
            encoded_input = (encoded_memory + encoded_context) / 2.0
        else:
            encoded_input = encoded_memory
        
        # 多神经元处理 - 修复：每次都重新初始化状态
        neuron_outputs = []
        
        for neuron in self.neurons:
            # 每次调用都使用新的隐藏状态
            prediction, _ = neuron(encoded_input, hidden_state=None)
            neuron_outputs.append(prediction.unsqueeze(1))  # (batch_size, 1, encoding_dim)
        
        # 堆叠神经元输出
        stacked_outputs = torch.cat(neuron_outputs, dim=1)  # (batch_size, num_neurons, encoding_dim)
        
        # 注意力机制融合
        attended_output, attention_weights = self.attention(
            stacked_outputs, stacked_outputs, stacked_outputs
        )  # (batch_size, num_neurons, encoding_dim)
        
        # 平均池化
        fused_output = attended_output.mean(dim=1)  # (batch_size, encoding_dim)
        
        # 层归一化
        fused_output = self.layer_norm1(fused_output)
        
        # 残差连接
        fused_output = fused_output + encoded_input
        fused_output = self.layer_norm2(fused_output)
        
        # 解码
        prediction = self.decoder(fused_output)  # (batch_size, num_inputs)
        
        return prediction
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        完整的前向传播 - 修复批次维度问题版本
        Args:
            inputs: 输入张量 (batch_size, seq_len, num_inputs) 或 (batch_size, num_inputs)
        Returns:
            outputs: 输出预测 (batch_size, seq_len, num_inputs)
            total_loss: 总损失
            metrics: 性能指标字典
        """
        # 维度处理
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)  # 添加序列维度
        
        batch_size, seq_len, input_dim = inputs.shape
        device = inputs.device
        
        # 维度检查
        assert input_dim == self.num_inputs, \
            f"输入维度不匹配: 期望 {self.num_inputs}, 得到 {input_dim}"
        
        # 编码输入
        inputs_reshaped = inputs.view(-1, input_dim)  # (batch_size*seq_len, num_inputs)
        encoded_inputs = self.encoder(inputs_reshaped)  # (batch_size*seq_len, encoding_dim)
        encoded_inputs = encoded_inputs.view(batch_size, seq_len, self.encoding_dim)
        
        # 初始化神经元状态
        neuron_states = [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_neurons)
        ]
        
        # 逐时间步处理
        outputs = []
        prediction_errors = []
        
        for t in range(seq_len):
            current_input = encoded_inputs[:, t, :]  # (batch_size, encoding_dim)
            
            # 多神经元处理
            step_outputs = []
            step_errors = []
            
            for i, neuron in enumerate(self.neurons):
                prediction, new_state = neuron(current_input, neuron_states[i])
                neuron_states[i] = new_state  # 更新状态
                
                # 计算预测误差
                error = F.mse_loss(prediction, current_input, reduction='none')
                
                step_outputs.append(prediction.unsqueeze(1))
                step_errors.append(error.mean(dim=1, keepdim=True))
            
            # 融合神经元输出
            stacked_outputs = torch.cat(step_outputs, dim=1)  # (batch_size, num_neurons, encoding_dim)
            
            # 注意力融合
            attended_output, _ = self.attention(
                stacked_outputs, stacked_outputs, stacked_outputs
            )
            fused_output = attended_output.mean(dim=1)  # (batch_size, encoding_dim)
            
            # 层归一化和残差连接
            fused_output = self.layer_norm1(fused_output)
            fused_output = fused_output + current_input
            fused_output = self.layer_norm2(fused_output)
            
            outputs.append(fused_output.unsqueeze(1))
            prediction_errors.extend(step_errors)
        
        # 组合输出
        encoded_outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, encoding_dim)
        
        # 解码
        encoded_outputs_reshaped = encoded_outputs.view(-1, self.encoding_dim)
        decoded_outputs = self.decoder(encoded_outputs_reshaped)
        final_outputs = decoded_outputs.view(batch_size, seq_len, self.num_inputs)
        
        # 计算损失
        reconstruction_loss = F.mse_loss(final_outputs, inputs)
        prediction_loss = torch.stack(prediction_errors).mean() if prediction_errors else torch.tensor(0.0, device=device)
        total_loss = reconstruction_loss + 0.1 * prediction_loss
        
        # 性能指标
        metrics = {
            'reconstruction_loss': reconstruction_loss,
            'prediction_loss': prediction_loss,
            'total_loss': total_loss,
            'input_norm': inputs.norm(dim=-1).mean(),
            'output_norm': final_outputs.norm(dim=-1).mean()
        }
        
        return final_outputs, total_loss, metrics
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆使用摘要"""
        memory_info = {}
        for i, neuron in enumerate(self.neurons):
            memory_info[f'neuron_{i}'] = {
                'memory_count': len(neuron.memory_bank),
                'memory_capacity': neuron.memory_capacity,
                'utilization': len(neuron.memory_bank) / neuron.memory_capacity
            }
        return memory_info
    
    def reset_states(self):
        """重置所有神经元状态"""
        self.neuron_states.clear()

# 使用示例和测试
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = PredictiveCodingAgent(
        num_inputs=384,
        encoding_dim=64,
        hidden_size=128,
        num_neurons=4,
        memory_capacity=1000
    ).to(device)
    
    print("🔍 模型结构:")
    print(f"  参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试不同批次大小的基本功能
    print("\n📊 测试不同批次大小:")
    
    for batch_size in [1, 4, 8]:
        seq_len = 32
        test_input = torch.randn(batch_size, seq_len, 384).to(device)
        
        print(f"\n  批次大小 {batch_size}:")
        print(f"    输入形状: {test_input.shape}")
        
        # 前向传播
        with torch.no_grad():
            outputs, loss, metrics = model(test_input)
        
        print(f"    ✅ 输出形状: {outputs.shape}")
        print(f"    ✅ 总损失: {loss.item():.6f}")
    
    # 测试记忆编码和预测
    print("\n🧠 测试记忆功能:")
    
    # 测试不同批次大小的记忆预测
    for batch_size in [1, 3]:
        memory_vec = model.encode_input("subject_example", "predicate_example", "object_example")
        memory_batch = memory_vec.unsqueeze(0).repeat(batch_size, 1).to(device)
        
        print(f"\n  批次大小 {batch_size}:")
        print(f"    记忆向量形状: {memory_batch.shape}")
        
        # 基于记忆的预测
        with torch.no_grad():
            memory_prediction = model.predict_with_memory(memory_batch)
        
        print(f"    ✅ 记忆预测形状: {memory_prediction.shape}")
    
    # 记忆使用情况
    memory_summary = model.get_memory_summary()
    print(f"\n📈 记忆使用情况: {memory_summary}")
    
    print("\n🎉 所有批次大小测试通过！模型就绪。")
    
    # 清理状态
    model.reset_states()
    print("🧹 状态已清理")
