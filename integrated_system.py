# integrated_system.py
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from config import Config
from adapters import TNAdapter,SeRNNAdapter,PredictiveAdapter
from adapters import PDAAdapter
import asyncio
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from PDA import PredictiveDialogAgent

@dataclass
class ProcessingResult:
    """处理结果数据类"""
    original_input: Any
    triples: List
    compressed_vectors: np.ndarray
    spatial_embeddings: torch.Tensor
    predictions: Any
    metadata: Dict[str, Any]

class IntegratedSystem:
    """模块化集成系统"""
    
    def __init__(self, config: Config):
        self.config = config
        self.modules = {}
        self.adapters = {}
        self.pipeline = []
        # 集成sentenceTransformer和deepseek
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.deepseek_client = AsyncOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=self.config.llm_config.api_key
        )
        # 集成PDA对话agent
        self.pda_agent = PredictiveDialogAgent(
            deepseek_api_key=self.config.llm_config.api_key
        )
        # 记忆、认知、历史系统完全共享
        self.memory_bank = self.pda_agent.memory_bank
        self.pc_core = self.pda_agent.pc_core
        self.dialog_buffer = self.pda_agent.dialog_buffer
        self.pda_agent.load_memory()  # 启动时加载记忆
        self._initialize_modules()
        self._initialize_adapters()
        self._build_pipeline()
    
    def _initialize_modules(self):
        """初始化各个模块"""
        from TN import TripleCompressor
        from seRNN import SeRNN  
        from PredictiveCoding import PredictiveCodingAgent
        
        # 初始化TN模块
        if self.config.pipeline_config['use_tn']:
            self.modules['tn'] = TripleCompressor(**self.config.tn_config)
        
        # 初始化SeRNN模块
        if self.config.pipeline_config['use_sernn']:
            self.modules['sernn'] = SeRNN(**self.config.sernn_config)
        
        # 初始化预测编码模块
        if self.config.pipeline_config['use_predictive']:
            pcfg = self.config.predictive_config
            self.modules['predictive'] = PredictiveCodingAgent(
                num_inputs=pcfg.input_dim,
                encoding_dim=pcfg.hidden_dim,
                hidden_size=pcfg.output_dim,
                num_neurons=getattr(pcfg, 'num_neurons', 4),
                memory_capacity=getattr(pcfg, 'memory_size', 1000)
            )
    
    def _initialize_adapters(self):
        """初始化适配器"""
        if 'tn' in self.modules:
            self.adapters['tn'] = TNAdapter(self.modules['tn'])
        
        if 'sernn' in self.modules:
            self.adapters['sernn'] = SeRNNAdapter(self.modules['sernn'])
        
        if 'predictive' in self.modules:
            self.adapters['predictive'] = PredictiveAdapter(self.modules['predictive'])
        
        if self.config.pipeline_config.get('use_pda', False):
            self.adapters['pda'] = PDAAdapter(self.config)
    
    def _build_pipeline(self):
        """构建处理管道"""
        if self.config.pipeline_config['use_tn']:
            self.pipeline.append(('triple_compression', self._process_tn))
        
        if self.config.pipeline_config['use_sernn']:
            self.pipeline.append(('spatial_embedding', self._process_sernn))
        
        if self.config.pipeline_config['use_predictive']:
            self.pipeline.append(('predictive_coding', self._process_predictive))
        
        if self.config.pipeline_config.get('use_pda', False):
            self.pipeline.append(('pda', self._process_pda))
    
    def _process_tn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """TN处理步骤"""
        triples = data.get('triples', [])
        
        if not triples:
            print("Warning: No triples found for TN processing")
            data['compressed_vectors'] = np.array([])
            return data
        
        try:
            # 准备和压缩三元组
            formatted_triples = self.adapters['tn'].prepare_triples_for_compression(triples)
            compressed_vectors = self.adapters['tn'].compress_and_flatten(formatted_triples)
            
            data['compressed_vectors'] = compressed_vectors
            data['tn_features'] = self.adapters['tn'].extract_spatial_features(compressed_vectors)
            
            print(f"TN processing completed: {len(formatted_triples)} triples compressed")
            
        except Exception as e:
            print(f"Error in TN processing: {e}")
            data['compressed_vectors'] = np.array([])
            data['tn_features'] = {}
        
        return data
    
    def _process_sernn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """SeRNN处理步骤"""
        compressed_vectors = data.get('compressed_vectors', np.array([]))
        
        if compressed_vectors.size == 0:
            print("Warning: No compressed vectors for SeRNN processing")
            data['spatial_embeddings'] = torch.tensor([])
            return data
        
        try:
            # 准备输入并处理
            tn_input = self.adapters['tn'].prepare_for_sernn(compressed_vectors)
            spatial_output = self.adapters['sernn'].process_tn_output(tn_input)
            
            data['spatial_embeddings'] = spatial_output
            data['sernn_info'] = self.adapters['sernn'].extract_spatial_embeddings(spatial_output)
            
            print(f"SeRNN processing completed: output shape {spatial_output.shape}")
            
        except Exception as e:
            print(f"Error in SeRNN processing: {e}")
            data['spatial_embeddings'] = torch.tensor([])
            data['sernn_info'] = {}
        
        return data
    
    def _process_predictive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """预测编码处理步骤"""
        spatial_embeddings = data.get('spatial_embeddings', torch.tensor([]))
        
        if spatial_embeddings.numel() == 0:
            print("Warning: No spatial embeddings for predictive processing")
            data['predictions'] = None
            return data
        
        try:
            # 预测处理
            prediction_result = self.adapters['predictive'].process_sernn_output(spatial_embeddings)
            
            data['predictions'] = prediction_result['predictions']
            data['prediction_confidence'] = prediction_result['confidence']
            data['prediction_features'] = prediction_result['features']
            
            print(f"Predictive processing completed with confidence: {prediction_result['confidence']:.3f}")
            
        except Exception as e:
            print(f"Error in predictive processing: {e}")
            data['predictions'] = None
            data['prediction_confidence'] = 0.0
        
        return data
    
    def _process_pda(self, data: Dict[str, Any]) -> Dict[str, Any]:
        spatial_embeddings = data.get('spatial_embeddings', torch.tensor([]))
        if spatial_embeddings.numel() == 0:
            print("Warning: No spatial embeddings for PDA processing")
            data['pda_predictions'] = None
            return data
        try:
            pda_result = self.adapters['pda'].process_seRNN_output(spatial_embeddings)
            data['pda_predictions'] = pda_result['predictions']
            data['pda_confidence'] = pda_result['confidence']
            data['pda_attention_weights'] = pda_result['attention_weights']
            data['pda_encoded_features'] = pda_result['encoded_features']
            data['pda_memory_state'] = pda_result['memory_state']
            print(f"PDA processing completed with confidence: {pda_result['confidence']:.3f}")
        except Exception as e:
            print(f"Error in PDA processing: {e}")
            data['pda_predictions'] = None
            data['pda_confidence'] = 0.0
        return data
    
    def process(self, input_data: Any, triples: Optional[List] = None) -> ProcessingResult:
        """主处理函数"""
        # 初始化处理数据
        data = {
            'original_input': input_data,
            'triples': triples or self._extract_triples_from_input(input_data),
            'metadata': {
                'timestamp': torch.tensor(0.0),  # 可以使用实际时间戳
                'processing_steps': []
            }
        }
        # 可用self.pc_core.predict(...)评估认知误差
        # 可用self.memory_bank.query_memory(...)检索知识
        # 执行处理管道
        for step_name, step_func in self.pipeline:
            try:
                print(f"Executing step: {step_name}")
                data = step_func(data)
                data['metadata']['processing_steps'].append(step_name)
            except Exception as e:
                print(f"Critical error in {step_name}: {e}")
                break
        # 创建结果对象
        result = ProcessingResult(
            original_input=data['original_input'],
            triples=data['triples'],
            compressed_vectors=data.get('compressed_vectors', np.array([])),
            spatial_embeddings=data.get('spatial_embeddings', torch.tensor([])),
            predictions=data.get('predictions'),
            metadata=data['metadata']
        )
        return result
    
    def _extract_triples_from_input(self, input_data: Any) -> List:
        """从输入数据中提取三元组，优先用deepseek LLM自动抽取"""
        if isinstance(input_data, str):
            prompt = (
                "请从下面的文本中抽取所有三元组（主体, 谓词, 宾体），以JSON数组返回：\n" + input_data
            )
            async def call_deepseek():
                try:
                    response = await self.deepseek_client.chat.completions.create(
                        model=self.config.llm_config.model_name,
                        messages=[
                            {"role": "system", "content": "你是知识抽取专家"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.config.llm_config.temperature,
                        max_tokens=self.config.llm_config.max_tokens
                    )
                    result = response.choices[0].message.content.strip()
                    import json
                    triples = json.loads(result)
                    if isinstance(triples, list):
                        return [tuple(item.values()) for item in triples if isinstance(item, dict)]
                except Exception as e:
                    print(f"LLM三元组抽取失败，回退本地规则: {e}")
                return [("subject", "predicate", "object")]
            # 用asyncio.run保证同步主流程可用
            return asyncio.run(call_deepseek())
        elif isinstance(input_data, list):
            return input_data
        else:
            return []
    
    def batch_process(self, input_batch: List[Any]) -> List[ProcessingResult]:
        """批处理功能"""
        results = []
        batch_size = self.config.pipeline_config['batch_size']
        
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i + batch_size]
            batch_results = [self.process(item) for item in batch]
            results.extend(batch_results)
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'modules': list(self.modules.keys()),
            'pipeline_steps': [step[0] for step in self.pipeline],
            'config': {
                'tn': self.config.tn_config,
                'sernn': self.config.sernn_config,
                'predictive': self.config.predictive_config,
                'pipeline': self.config.pipeline_config
            }
        }
    
    def chat_with_agent(self, user_input: str) -> dict:
        """与PDA智能体进行多轮对话，返回AI回复、记忆统计、思维链、认知误差"""
        import asyncio
        reply = asyncio.run(self.pda_agent.dialog_round(user_input))
        return {
            "reply": reply,
            "memory_stats": self.pda_agent.get_memory_stats(),
            "thought_chain": self.dialog_buffer.chain_text(),
            "prediction_error": self.pc_core.current_prediction_error
        }

    def save_all_memory(self):
        self.pda_agent.save_memory()
