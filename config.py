import torch
from adapters import PDAAdapter
import os



class Config:
    def __init__(self):

        # 设备配置
        self.device = self._get_device()

        # TN模块配置 - 统一使用384维
        self.tn_config = {
            'embed_dim': 384,          # 词嵌入维度设为384
            'num_entities': 10000,     # 实体数量
            'num_relations': 100,      # 关系数量
            #'tensor_dim': 384,         # 张量维度与嵌入维度一致
            #'compression_dim': 384,    # 压缩后输出维度
            'device': self.device
        }
        
        # seRNN模块配置 - 输入维度与TN输出对齐
        self.sernn_config = {
            'input_size': 384,          # 输入维度与TN输出维度一致
            'hidden_size': 512,         # 隐藏层维度可以不同
            #'output_size': 384,
            'spatial_dim':1000,
            'num_layers': 3,
            'dropout': 0.1,
            'device': self.device
        }
        self.predictive_config = {
            'num_inputs': 384,
            'encoding_dim': 64,
            'hidden_size': 128,
            'num_neurons': 4,
            'memory_capacity': 1000
        } 

        self.pipeline_config = {
            'use_tn': True,           # 是否使用TN模块
            'use_sernn': True,        # 是否使用seRNN模块  
            'use_predictive': True,   # 是否使用预测模块
            'batch_size': 32,         # 批处理大小
            'enable_integration': True # 是否启用模块集成
        }
        
        # 训练配置
        self.training_config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'weight_decay': 1e-4,
            'scheduler': 'cosine'
        }
        # 增强的PDA配置
        self.predictive_config = PredictiveConfig()
        
        # LLM配置
        self.llm_config = LLMConfig()

        # 设备配置
        self.device = self._get_device()
    
    def _get_device(self):
        """自动选择设备"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def get_tn_config(self):
        """获取TN模块配置"""
        config = self.tn_config.copy()
        if config['device'] == 'auto':
            config['device'] = self.device
        return config
    
    def get_sernn_config(self):
        """获取seRNN模块配置"""
        config = self.sernn_config.copy()
        if config['device'] == 'auto':
            config['device'] = self.device
        return config
class PredictiveConfig:
    def __init__(self):
        self.input_dim = 512
        self.hidden_dim = 256
        self.output_dim = 128
        self.memory_size = 1000
        self.confidence_threshold = 0.7
        self.attention_heads = 8
        self.dropout = 0.1
  
    # 添加字典转换支持
    def __iter__(self):
        yield "input_dim", self.input_dim
        yield "hidden_dim", self.hidden_dim
        yield "output_dim", self.output_dim
        yield "memory_size", self.memory_size
        yield "confidence_threshold", self.confidence_threshold
        yield "attention_heads", self.attention_heads
        yield "dropout", self.dropout
        # 列出所有需要暴露的参数
    
    # 添加键值访问支持
    def __getitem__(self, key):
        return getattr(self, key)
class LLMConfig:
    def __init__(self):
        self.api_key = ''  # 需要设置
        self.model_name = "deepseek-chat"  # 或其他模型
        self.max_tokens = 2048
        self.temperature = 0.7
        self.timeout = 30
# 全局配置实例
config = Config()