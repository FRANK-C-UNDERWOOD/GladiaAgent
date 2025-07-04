# adapters.py
import torch
import numpy as np
from typing import List, Dict, Any, Union, Tuple, Optional
import torch.nn as nn
import time

class SeRNNAdapter:
    """SeRNN模块适配器 - 处理空间嵌入向量"""
    
    def __init__(self, sernn_model):
        self.sernn = sernn_model
    
    def process_tn_output(self, tn_vectors: torch.Tensor) -> torch.Tensor:
        """处理来自Sentence Transformers的嵌入向量"""
        try:
            # 确保输入维度正确
            if tn_vectors.dim() == 1:
                tn_vectors = tn_vectors.unsqueeze(0)
            
            # 直接传入SeRNN模型处理
            if tn_vectors.dim() == 2:  # [batch, dim]
                sernn_input = tn_vectors.unsqueeze(1)  # -> [batch, 1, dim]
            else:  # Already [batch, seq, dim]
                sernn_input = tn_vectors
                
            # 创建模拟空间位置
            batch_size = tn_vectors.size(0)
            seq_len = sernn_input.size(1)
            spatial_positions = torch.randint(0, self.sernn.spatial_dim, 
                                            (batch_size, seq_len), 
                                            device=tn_vectors.device)
            
            # 处理输入
            output_sequence, _ = self.sernn(sernn_input, spatial_positions)
            return output_sequence
            
        except Exception as e:
            print(f"Error in SeRNN processing: {e}")
            return torch.zeros(1, self.sernn.hidden_size)
    
    def extract_spatial_embeddings(self, output: torch.Tensor) -> Dict[str, Any]:
        """提取空间嵌入信息"""
        return {
            'embeddings': output.detach().cpu().numpy(),
            'shape': output.shape,
            'spatial_features': {
                'mean': float(torch.mean(output)),
                'std': float(torch.std(output))
            }
        }

class PredictiveAdapter:
    """预测编码适配器 - 处理空间嵌入"""
    
    def __init__(self, predictive_agent):
        self.predictive_agent = predictive_agent
    
    def process_sernn_output(self, sernn_output: torch.Tensor) -> Dict[str, Any]:
        """处理来自SeRNN的输出"""
        try:
            # 确保输入格式正确
            if sernn_output.dim() == 2:  # [batch, dim]
                sernn_output = sernn_output.unsqueeze(1)  # -> [batch, 1, dim]
            
            # 调用预测编码模型
            predictions = self.predictive_agent(sernn_output)
            return {
                'predictions': predictions[0],
                'confidence': self._calculate_confidence(predictions),
                'features': sernn_output.detach().cpu().numpy()
            }
        except Exception as e:
            print(f"Error in predictive processing: {e}")
            return {'predictions': None, 'confidence': 0.0, 'features': None}
    
    def _calculate_confidence(self, predictions) -> float:
        """计算预测置信度"""
        if predictions is None or not hasattr(predictions, 'var'):
            return 0.0
        
        # 简单的置信度计算
        if isinstance(predictions, torch.Tensor):
            variance = torch.var(predictions)
            return float(1.0 / (1.0 + variance))
        return 0.5

class PDAAdapter:
    """PDA适配器 - 提供预测编码功能"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.prediction_layers = self._build_prediction_layers()
        self.memory_bank = PredictiveMemoryBank(config)
        self.confidence_estimator = ConfidenceEstimator(config)
        
    def _build_prediction_layers(self):
        """构建多层预测网络"""
        layers = nn.ModuleDict({
            'encoder': nn.Linear(
                self.config.predictive_config.input_dim,
                self.config.predictive_config.hidden_dim
            ),
            'predictor': nn.Linear(
                self.config.predictive_config.hidden_dim,
                self.config.predictive_config.output_dim
            ),
            'attention': nn.MultiheadAttention(
                self.config.predictive_config.hidden_dim,
                num_heads=8
            )
        })
        return layers.to(self.device)
    
    def process_seRNN_output(self, seRNN_output: torch.Tensor) -> Dict:
        """处理SeRNN输出并进行预测编码"""
        # 编码输入
        encoded = self.prediction_layers['encoder'](seRNN_output)
        
        # 注意力机制增强
        attended, attention_weights = self.prediction_layers['attention'](
            encoded, encoded, encoded
        )
        
        # 预测生成
        predictions = self.prediction_layers['predictor'](attended)
        
        # 置信度估计
        confidence = self.confidence_estimator.estimate(predictions, seRNN_output)
        
        # 记忆更新
        self.memory_bank.update(encoded, predictions, confidence)
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'attention_weights': attention_weights,
            'encoded_features': encoded,
            'memory_state': self.memory_bank.get_state()
        }
    
    def extract_triples_with_llm(self, text: str, llm_client) -> List[Tuple]:
        """结合LLM进行三元组提取"""
        # 使用预测编码增强文本理解
        enhanced_context = self.memory_bank.get_relevant_context(text)
        
        # 构建LLM提示
        prompt = self._build_triple_extraction_prompt(text, enhanced_context)
        
        # 调用LLM
        llm_response = llm_client.generate(prompt)
        
        # 解析三元组
        triples = self._parse_triples(llm_response)
        
        # 使用预测编码验证和优化
        validated_triples = self._validate_triples_with_prediction(triples)
        
        return validated_triples
    
    def _build_triple_extraction_prompt(self, text: str, context: Dict) -> str:
        """构建三元组提取提示"""
        return f"""
        基于以下上下文和文本，提取准确的三元组关系：
        
        上下文信息：{context.get('relevant_knowledge', '')}
        置信度参考：{context.get('confidence_threshold', 0.8)}
        
        文本：{text}
        
        请以JSON格式返回三元组列表，格式为：
        [{"subject": "主体", "predicate": "谓词", "object": "宾体", "confidence": 0.95}]
        """
    
    def _validate_triples_with_prediction(self, triples: List[Tuple]) -> List[Tuple]:
        """使用预测编码验证三元组"""
        validated = []
        for triple in triples:
            # 预测编码验证
            prediction_score = self._predict_triple_validity(triple)
            if prediction_score > self.config.predictive_config.confidence_threshold:
                validated.append(triple)
        return validated

class PredictiveMemoryBank:
    """预测编码记忆库"""
    
    def __init__(self, config):
        self.config = config
        self.memory_size = config.predictive_config.memory_size
        self.memory_buffer = []
        self.knowledge_graph = {}
        
    def update(self, features: torch.Tensor, predictions: torch.Tensor, confidence: float):
        """更新记忆库"""
        memory_item = {
            'features': features.detach().cpu(),
            'predictions': predictions.detach().cpu(),
            'confidence': confidence,
            'timestamp': torch.tensor(time.time())
        }
        
        self.memory_buffer.append(memory_item)
        
        # 保持记忆库大小
        if len(self.memory_buffer) > self.memory_size:
            self.memory_buffer.pop(0)
    
    def get_relevant_context(self, query: str) -> Dict:
        """获取相关上下文"""
        if not self.memory_buffer:
            return {'relevant_knowledge': '', 'confidence_threshold': 0.5}
        
        # 获取最近的高置信度记忆
        recent_high_conf = [
            item for item in self.memory_buffer[-10:]
            if item['confidence'] > 0.8
        ]
        
        return {
            'relevant_knowledge': f"基于{len(recent_high_conf)}条相关记忆",
            'confidence_threshold': np.mean([item['confidence'] for item in recent_high_conf]) if recent_high_conf else 0.5
        }

class ConfidenceEstimator:
    """置信度估计器"""
    
    def __init__(self, config):
        self.config = config
        self.estimator = nn.Sequential(
            nn.Linear(config.predictive_config.output_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(config.device)
    
    def estimate(self, predictions: torch.Tensor, original_input: torch.Tensor) -> float:
        """估计预测置信度"""
        # 展平输入
        predictions_flat = predictions.flatten()
        input_flat = original_input.flatten()
        
        # 确保向量长度一致
        min_len = min(len(predictions_flat), len(input_flat))
        if min_len == 0:
            return 0.5
        
        # 截断或填充
        predictions_flat = predictions_flat[:min_len]
        input_flat = input_flat[:min_len]
        
        # 合并输入
        combined = torch.cat([predictions_flat, input_flat])
        confidence = self.estimator(combined.unsqueeze(0))
        return confidence.item()
        """估计预测置信度"""
        combined = torch.cat([predictions.flatten(), original_input.flatten()])
        confidence = self.estimator(combined.unsqueeze(0))
        return confidence.item()
