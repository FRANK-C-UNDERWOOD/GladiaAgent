import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, AsyncGenerator
from dataclasses import dataclass
from config import Config
from adapters import SeRNNAdapter, PredictiveAdapter
import asyncio
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI, OpenAI
from PDA import PredictiveDialogAgent
import os
import json
import traceback
import time
import threading

CORE_MEMORY_DIR = "core_sernn_memory"

@dataclass
class ProcessingResult:
    original_input: Any
    triples: List
    compressed_vectors: np.ndarray
    spatial_embeddings: torch.Tensor
    predictions: Any
    metadata: Dict[str, Any]

class IntegratedSystem:
    def __init__(self, config: Config):
        self.config = config
        self.modules = {}
        self.adapters = {}
        self.pipeline = []
        self.knowledge_base_vectors: Dict[str, torch.Tensor] = {}
        self.memory_base_path = CORE_MEMORY_DIR
        self.kb_file_path = os.path.join(self.memory_base_path, "knowledge_base.pt")
        self.sernn_model_file_path = os.path.join(self.memory_base_path, "sernn_model_state.pth")
        os.makedirs(self.memory_base_path, exist_ok=True)

        # 加载Sentence Transformers模型
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # 初始化模块
        self._initialize_modules()
        self._initialize_adapters()
        self._build_pipeline()
        self._load_core_knowledge()

        # 初始化同步和异步客户端
        self.async_deepseek_client = AsyncOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=self.config.llm_config.api_key
        )
        self.sync_deepseek_client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=self.config.llm_config.api_key
        )
        
        self.pda_agent = PredictiveDialogAgent(
            deepseek_api_key=self.config.llm_config.api_key,
            integrated_system_ref=self
        )

    def _initialize_modules(self):
        from seRNN import SeRNN
        from PredictiveCoding import PredictiveCodingAgent

        # 只初始化seRNN和预测编码模块
        if self.config.pipeline_config['use_sernn']:
            self.modules['sernn'] = SeRNN(**self.config.sernn_config).to(self.config.device)

        if self.config.pipeline_config['use_predictive']:
            pcfg = self.config.core_pc_module_config
            self.modules['predictive_coding_module'] = PredictiveCodingAgent(
                num_inputs=pcfg['num_inputs'],
                encoding_dim=pcfg['encoding_dim'],
                hidden_size=pcfg['neuron_hidden_size'],
                num_neurons=pcfg['num_neurons'],
                memory_capacity=pcfg['neuron_memory_capacity']
            ).to(self.config.device)

    def _initialize_adapters(self):
        # 只初始化seRNN和预测编码的适配器
        if 'sernn' in self.modules:
            self.adapters['sernn'] = SeRNNAdapter(self.modules['sernn'])
        if 'predictive_coding_module' in self.modules:
            self.adapters['predictive'] = PredictiveAdapter(self.modules['predictive_coding_module'])

    def _build_pipeline(self):
        # 直接开始sernn处理
        if self.config.pipeline_config['use_sernn']:
            self.pipeline.append(('spatial_embedding', self._process_sernn))
        
        # 预测编码处理保持不变
        if self.config.pipeline_config.get('use_pipeline_predictive_module', False) and \
           'predictive_adapter_for_pipeline' in self.modules and \
           'predictive_pipeline' in self.adapters:
            self.pipeline.append(('predictive_coding_processing', self._process_predictive_module))

    def _get_triple_str_key(self, triple: tuple) -> str:
        return json.dumps(sorted(triple), ensure_ascii=False)

    def _process_sernn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理seRNN模块"""
        # 直接从数据中获取压缩向量
        compressed_vectors = data.get('compressed_vectors', torch.empty(0))
        if compressed_vectors.numel() == 0 or 'sernn' not in self.modules:
            data['spatial_embeddings'] = torch.tensor([])
            return data
        try:
            if compressed_vectors.dim() == 2: # [batch, dim]
                sernn_input = compressed_vectors.unsqueeze(1) # -> [batch, 1, dim] for seq_len=1
            else: # Already [batch, seq, dim]
                sernn_input = compressed_vectors

            batch_s, seq_s, _ = sernn_input.shape
            # seRNN expects spatial_positions: [batch_size, seq_len]
            mock_spatial_positions = torch.randint(0, self.config.sernn_config['spatial_dim'], 
                                                   (batch_s, seq_s), device=sernn_input.device)
            
            output_sequence, _ = self.modules['sernn'](sernn_input, mock_spatial_positions)
            data['spatial_embeddings'] = output_sequence
        except Exception as e:
            print(f"Error in SeRNN processing: {e}")
            traceback.print_exc()
            data['spatial_embeddings'] = torch.tensor([])
        return data

    def _process_predictive_module(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理预测编码模块"""
        spatial_embeddings = data.get('spatial_embeddings', torch.tensor([]))
        if spatial_embeddings.numel() == 0 or 'predictive' not in self.adapters:
            data['predictions'] = None
            return data
        try:
            prediction_result = self.adapters['predictive'].process_sernn_output(spatial_embeddings)
            data['predictions'] = prediction_result.get('predictions')
        except Exception as e:
            print(f"Error in general predictive coding module processing: {e}")
            traceback.print_exc()
            data['predictions'] = None
        return data

    def generate_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """使用Sentence Transformers生成384维嵌入"""
        print(f"[DEBUG EMBEDDING] generate_embeddings called with texts: {texts}")
        try:
            if isinstance(texts, str):
                # Process single string
                print(f"[DEBUG EMBEDDING] Encoding single text item: '{texts}'")
                try:
                    # Encode as a list containing one item for consistency with batch processing
                    embeddings = self.embedder.encode([texts], convert_to_tensor=True)
                    print(f"[DEBUG EMBEDDING] Single text item encoded successfully, shape: {embeddings.shape}")
                except Exception as item_e:
                    print(f"[DEBUG EMBEDDING] ERROR encoding single text item ('{texts}'): {item_e}")
                    traceback.print_exc()
                    raise item_e # Re-raise the specific error
            
            elif isinstance(texts, list):
                if not texts: # Handles empty list case
                    print("[DEBUG EMBEDDING] Input 'texts' is an empty list, returning empty tensor of shape (0, embedding_dim).")
                    # Ensure correct empty shape, get embedding_dim from embedder
                    embedding_dim = self.embedder.get_sentence_embedding_dimension() if hasattr(self.embedder, 'get_sentence_embedding_dimension') else 384
                    return torch.empty((0, embedding_dim), device=self.config.device) 
                
                embeddings_list = []
                # all_successful = True # Not strictly needed if we re-raise on item error
                for i, text_item in enumerate(texts):
                    print(f"[DEBUG EMBEDDING] Encoding item {i+1}/{len(texts)}: '{text_item}'")
                    if not isinstance(text_item, str):
                        print(f"[DEBUG EMBEDDING] ERROR: Item {i+1} is not a string, it's a {type(text_item)}. Value: '{text_item}'.")
                        # Option 1: Raise an error
                        raise TypeError(f"Item {i+1} in list is not a string: {type(text_item)}")
                        # Option 2: Skip (less safe, might hide problems)
                        # all_successful = False
                        # continue 

                    try:
                        # Encode each item as a list containing one item
                        emb = self.embedder.encode([text_item], convert_to_tensor=True)
                        embeddings_list.append(emb)
                        print(f"[DEBUG EMBEDDING] Item {i+1} encoded successfully, shape: {emb.shape}")
                    except Exception as item_e:
                        print(f"[DEBUG EMBEDDING] ERROR encoding item {i+1} ('{text_item}'): {item_e}")
                        traceback.print_exc()
                        # If one item fails, you might want to stop or handle it
                        # For debugging, let's re-raise to see the error immediately for that item
                        raise item_e 
                
                if not embeddings_list: # If list was not empty but all items failed or were skipped
                    print("[DEBUG EMBEDDING] No items were successfully encoded from the list.")
                    embedding_dim = self.embedder.get_sentence_embedding_dimension() if hasattr(self.embedder, 'get_sentence_embedding_dimension') else 384
                    return torch.empty((0, embedding_dim), device=self.config.device)

                try:
                    embeddings = torch.cat(embeddings_list, dim=0) # Concatenate along batch dimension
                except Exception as cat_e:
                    print(f"[DEBUG EMBEDDING] ERROR during torch.cat: {cat_e}")
                    print(f"[DEBUG EMBEDDING] shapes of embeddings in list: {[e.shape for e in embeddings_list]}")
                    traceback.print_exc()
                    raise cat_e

            else:
                print(f"[DEBUG EMBEDDING] ERROR: Invalid type for 'texts' argument: {type(texts)}. Value: {texts}")
                raise TypeError(f"Input 'texts' must be str or List[str], got {type(texts)}")

            print(f"[DEBUG EMBEDDING] Final embedding successful, shape: {embeddings.shape}")
            return embeddings.to(self.config.device)

        except Exception as e:
            # This outer catch is for errors not caught by more specific handlers above
            # or errors in the logic of this function itself.
            print(f"[DEBUG EMBEDDING] UNHANDLED ERROR in generate_embeddings: {e}")
            print(f"[DEBUG EMBEDDING] Original texts argument was: {texts}") # Be careful if texts is huge
            traceback.print_exc()
            raise # Re-raise the exception

    def _process_triples(self, triples: List[Tuple]) -> torch.Tensor:
        """处理三元组为384维嵌入"""
        print(f"[DEBUG EMBEDDING] _process_triples called with triples: {triples}") # ADD THIS
        try:
            if not triples:
                print("[DEBUG EMBEDDING] Input 'triples' is empty in _process_triples.") # ADD THIS
                return torch.empty(0) # This was already here
            
            triplet_texts = [f"{head} {relation} {tail}" for head, relation, tail in triples]
            print(f"[DEBUG EMBEDDING] Generated triplet_texts: {triplet_texts}") # ADD THIS
            return self.generate_embeddings(triplet_texts)
        except Exception as e:
            print(f"[DEBUG EMBEDDING] ERROR in _process_triples: {e}") # ADD THIS
            traceback.print_exc() # ADD THIS
            raise # Re-raise

    def process(self, input_data: Any, triples: Optional[List] = None) -> ProcessingResult:
        # 使用同步方式提取三元组
        if triples is None:
            if isinstance(input_data, str):
                # 同步提取三元组
                current_triples = self._extract_triples_from_input_sync(input_data)
            else:
                current_triples = []
        else:
            current_triples = triples
        
        compressed_vectors = self._process_triples(current_triples) 
        
        data_dict = {
            'original_input': input_data,
            'triples': current_triples, 
            'compressed_vectors': compressed_vectors, 
            'metadata': {'timestamp': time.time(), 'processing_steps': []} 
        }
        
        for step_name, step_func in self.pipeline:
            try:
                data_dict = step_func(data_dict)
                data_dict['metadata']['processing_steps'].append(step_name)
            except Exception as e:
                print(f"Critical error in pipeline step {step_name}: {e}")
                traceback.print_exc()
                break 
        
        # --- ADD THIS SECTION TO ITERATE AND STORE TRR PLES ---
        if current_triples: 
            print(f"[DEBUG PROCESS] Attempting to store {len(current_triples)} extracted triples.")
            for triple_to_store in current_triples:
                try:
                    if isinstance(triple_to_store, tuple) and len(triple_to_store) == 3 and \
                       all(isinstance(el, str) for el in triple_to_store):
                        self.store_triple_with_pc(triple_to_store)
                    else:
                        print(f"[DEBUG PROCESS] Skipping invalid triple format for storage: {triple_to_store}")
                except Exception as store_e:
                    print(f"[DEBUG PROCESS] Error calling store_triple_with_pc for triple {triple_to_store}: {store_e}")
                    traceback.print_exc()
        # --- END OF ADDED SECTION ---
        
        # Convert to numpy AFTER potential storage, if necessary for ProcessingResult
        # Or ensure store_triple_with_pc works with tensors if that's what compressed_vectors is
        final_compressed_vectors = data_dict.get('compressed_vectors', torch.empty(0))
        if isinstance(final_compressed_vectors, torch.Tensor):
            final_compressed_vectors = final_compressed_vectors.cpu().numpy()

        return ProcessingResult(
            original_input=data_dict.get('original_input'),
            triples=data_dict.get('triples', []),
            compressed_vectors=final_compressed_vectors,
            spatial_embeddings=data_dict.get('spatial_embeddings', torch.tensor([])),
            predictions=data_dict.get('predictions'),
            metadata=data_dict.get('metadata', {})
        )

    def _extract_triples_from_input_sync(self, input_data: str) -> List[Tuple[str, str, str]]:
        """同步方式提取三元组（避免嵌套事件循环）"""
        # 使用同步客户端提取三元组
        prompt = f"""
        请从以下中文文本中抽取所有可识别的知识三元组（主语-谓语-宾语）。
        每个三元组请严格按照JSON对象格式表示，包含 "subject"（主体）、"relation"（关系）和 "object"（客体）三个键。
        将所有抽取的JSON对象组成一个JSON数组返回。如果找不到任何三元组，请返回一个空数组 `[]`。
        例如:
        输入：小明在公园里踢足球，天气很好。
        输出：[
            {{"subject": "小明", "relation": "在公园里踢", "object": "足球"}}
        ]
        现在请从以下文本中提取三元组：
            {input_data}
        """
        try:
            response = self.sync_deepseek_client.chat.completions.create(
                model=self.config.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个精确的知识抽取引擎。请严格按照用户要求的JSON格式输出（JSON对象数组，每个对象包含subject, relation, object键）。如果找不到三元组，返回空数组[]。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, 
                max_tokens=self.config.llm_config.max_tokens,
                response_format={"type": "json_object"}
            )
            result_content = response.choices[0].message.content.strip()
            
            parsed_json = json.loads(result_content)
            triple_list_json = parsed_json.get("triples", parsed_json) if isinstance(parsed_json, dict) else parsed_json

            if isinstance(triple_list_json, list):
                extracted_tuples = []
                for item in triple_list_json:
                    if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                        extracted_tuples.append((str(item["subject"]), str(item["relation"]), str(item["object"])))
                    elif isinstance(item, list) and len(item) == 3 and all(isinstance(x, str) for x in item):
                        extracted_tuples.append(tuple(item)) # type: ignore
                return extracted_tuples
            return []
        except json.JSONDecodeError as e:
            print(f"LLM三元组抽取失败 (JSON Decode Error): {e}. Response: {result_content if 'result_content' in locals() else 'N/A'}")
            return []
        except Exception as e:
            print(f"LLM三元组抽取失败: {e}")
            traceback.print_exc()
            return []

    def batch_process(self, input_batch: List[Any]) -> List[ProcessingResult]:
        """批量处理输入"""
        results = []
        for item_input in input_batch:
            if isinstance(item_input, list):
                results.append(self.process(None, triples=item_input))
            else:
                results.append(self.process(item_input))
        return results

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'modules': list(self.modules.keys()),
            'pipeline_steps': [step[0] for step in self.pipeline],
            'config_summary': {
                'sernn_config_keys': list(self.config.sernn_config.keys()) if 'sernn' in self.modules else [],
                'core_pc_config_keys': list(self.config.core_pc_module_config.keys()) if hasattr(self.config, 'core_pc_module_config') else []
            },
            'knowledge_base_vector_count': len(self.knowledge_base_vectors),
            'embedding_model': 'paraphrase-MiniLM-L6-v2'
        }

    async def chat_with_agent_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """流式与对话代理交互"""
        # 使用同步方式处理用户输入（避免嵌套事件循环）
        self.process(user_input)
        
        # 获取流式响应生成器
        async for chunk in self.pda_agent.dialog_round(user_input):
            yield chunk

    async def get_embedding_for_query(self, query_text: str) -> torch.Tensor:
        """获取查询文本的嵌入"""
        return self.generate_embeddings(query_text)

    def query_core_knowledge_base(self, query_text: Optional[str] = None, 
                                query_vector: Optional[torch.Tensor] = None, 
                                top_k: int = 5) -> List[Tuple[str, torch.Tensor, float]]:
        """查询核心知识库"""
        # 生成查询向量
        if query_vector is None and query_text:
            # 使用同步方式获取嵌入
            query_vector = self.generate_embeddings(query_text)
        
        if query_vector is None or not self.knowledge_base_vectors:
            return []

        # 相似度计算
        query_vector_cpu = query_vector.cpu().float()
        similarities = []
        
        for triple_key, kb_vector in self.knowledge_base_vectors.items():
            kb_vector_cpu = kb_vector.cpu().float()
            similarity = torch.cosine_similarity(query_vector_cpu, kb_vector_cpu, dim=1).item()
            similarities.append((triple_key, kb_vector_cpu, similarity))
        
        # 返回Top K结果
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]

    def _save_core_knowledge(self):
        """保存核心知识库"""
        try:
            cpu_knowledge_base = {k: v.cpu() for k, v in self.knowledge_base_vectors.items()}
            torch.save(cpu_knowledge_base, self.kb_file_path)
            if 'sernn' in self.modules and hasattr(self.modules['sernn'], 'state_dict'):
                torch.save(self.modules['sernn'].state_dict(), self.sernn_model_file_path)
        except Exception as e:
            print(f"Error saving core knowledge: {e}")
            traceback.print_exc()

    def _load_core_knowledge(self):
        """加载核心知识库"""
        try:
            if os.path.exists(self.kb_file_path):
                self.knowledge_base_vectors = torch.load(self.kb_file_path, map_location=self.config.device)
            else:
                self.knowledge_base_vectors = {}

            if 'sernn' in self.modules and os.path.exists(self.sernn_model_file_path):
                try:
                    self.modules['sernn'].load_state_dict(torch.load(self.sernn_model_file_path, map_location=self.config.device))
                    self.modules['sernn'].eval()
                except Exception as e:
                    print(f"  Error loading SeRNN model state: {e}")
        except Exception as e:
            print(f"Error loading core knowledge: {e}")
            traceback.print_exc()
            self.knowledge_base_vectors = {}

    def save_all_memory(self):
        """保存所有记忆"""
        self._save_core_knowledge()

    def get_core_knowledge_vector_by_key(self, triple_key: str) -> Optional[torch.Tensor]:
        """通过键获取核心知识向量"""
        return self.knowledge_base_vectors.get(triple_key)

    def store_triple_with_pc(self, triple: Tuple[str, str, str]):
        """使用SeRNN + 预测编码判断是否存储三元组"""
        print(f"[DEBUG STORE] store_triple_with_pc CALLED with triple: {triple}")
        try:                                                  
            # 1. 编码三元组为句子文本并生成嵌入向量
            triple_text = f"{triple[0]} {triple[1]} {triple[2]}"
            embedding = self.generate_embeddings(triple_text).squeeze(0)  # [384]
            triple_key = self._get_triple_str_key(triple)
            store_vector = False
                                                                                      
            # 2. 检查系统是否可执行预测编码逻辑
            if 'sernn' in self.modules and 'predictive_coding_module' in self.modules \
               and self.config.pipeline_config.get('use_sernn', False) \
               and self.config.pipeline_config.get('use_predictive', False):
            
                # 2.1 输入嵌入送入SeRNN：注意添加mock空间位置
                sernn_input = embedding.unsqueeze(0).unsqueeze(0).to(self.config.device)  # [1, 1, 384]
                mock_spatial_positions = torch.randint(0, self.config.sernn_config['spatial_dim'], (1, 1), device=sernn_input.device)
                sernn_output, _ = self.modules['sernn'](sernn_input, mock_spatial_positions)
                sernn_vector = sernn_output.squeeze(0)  # shape: [1, 384]

                # 2.2 SeRNN输出送入预测编码模块
                pc_input = sernn_vector.to(self.config.device)
                _, loss_tensor, metrics = self.modules['predictive_coding_module'](pc_input)

                prediction_loss = metrics.get('prediction_loss', loss_tensor).item()

                # 2.3 判断是否存储（默认阈值为0.65）
                threshold = self.config.core_pc_module_config.get('storage_threshold', 0.65)
                if prediction_loss > threshold:
                    store_vector = True
                    print(f"[PC_STORE] Loss={prediction_loss:.4f} 超过阈值，存储该向量")
                else:
                    print(f"[PC_SKIP] Loss={prediction_loss:.4f} 未超过阈值，不存储")

                # 3. 存储为核心记忆
                if store_vector:
                    self.knowledge_base_vectors[triple_key] = sernn_vector.detach().cpu()
                    print(f"[STORED] 新三元组: {triple_key}")

            else:
                # fallback：不通过SeRNN/PC时，判断是否已存在
                if triple_key not in self.knowledge_base_vectors:
                    self.knowledge_base_vectors[triple_key] = embedding.cpu()
                    print(f"[FALLBACK STORE] 未启用PC模块，直接存储三元组: {triple_key}")

        except Exception as e:
            print(f"❌❌ Error storing triple via PC+SeRNN: {e}")
            traceback.print_exc()
