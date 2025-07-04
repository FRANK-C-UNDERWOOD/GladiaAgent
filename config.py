import torch
import os
from typing import Dict, List, Any
# 配置类
class PredictiveConfig:
    """预测编码配置 - 适配新架构"""
    def __init__(self):
        self.input_dim = 512      # 输入维度（SeRNN输出）
        self.hidden_dim = 256     # 隐藏层维度
        self.output_dim = 128     # 输出维度
        self.memory_size = 1000   # 记忆库大小
        self.confidence_threshold = 0.7 # 置信度阈值
        self.attention_heads = 8  # 注意力头数
        self.dropout = 0.1        # Dropout率

class LLMConfig:
    """大语言模型配置"""
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.model_name = "deepseek-chat"
        self.max_tokens = 2048    # 最大token数
        self.temperature = 0.7    # 温度参数
        self.timeout = 30          # 超时时间(秒)

class Config:
    """全局配置类 - 适配新架构"""
    def __init__(self):
        # 设备配置
        self.device = self._get_device()

        # seRNN模块配置 - 输入为384维嵌入
        self.sernn_config = {
            'input_size': 384,     # 输入维度 (来自Sentence Transformers)
            'hidden_size': 512,     # 隐藏层大小
            'spatial_dim': 1000,    # 空间维度
            'num_layers': 3,        # 层数
            'dropout': 0.1,         # Dropout率
            'device': self.device   # 设备
        }
        
        # 核心预测编码模块配置 - 用于存储决策
        self.core_pc_module_config = {
            'num_inputs': 384,           # 输入维度 (Sentence Transformers嵌入)
            'encoding_dim': 128,         # 编码维度
            'neuron_hidden_size': 256,   # 神经元隐藏大小
            'num_neurons': 4,            # 神经元数量
            'neuron_memory_capacity': 1000, # 神经元记忆容量
            'storage_threshold': 0.65    # 存储阈值
        }

        # 通用预测编码配置
        self.predictive_config = PredictiveConfig()  # 通用预测编码配置
        
        # 处理流水线配置
        self.pipeline_config = {
            'use_sernn': True,            # 启用seRNN模块
            'use_predictive': True,       # 启用预测编码模块
            'use_pipeline_predictive_module': False, # 是否启用流水线预测模块
            'batch_size': 32,             # 批大小
            'enable_integration': True   # 启用系统集成
        }
        
        # 训练配置
        self.training_config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'weight_decay': 1e-4,
            'scheduler': 'cosine'
        }
        
        # LLM配置
        self.llm_config = LLMConfig()
    
    def _get_device(self) -> str:
        """自动选择计算设备"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def get_sernn_config(self) -> Dict[str, Any]:
        """获取seRNN配置"""
        return self.sernn_config
    
    def get_pc_config(self) -> Dict[str, Any]:
        """获取预测编码配置"""
        return self.core_pc_module_config

# 全局配置实例
config = Config()
