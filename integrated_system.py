import torch
import numpy as np
from typing import List, Dict, Any, Optional,Tuple
from dataclasses import dataclass
from config import Config
from adapters import TNAdapter,SeRNNAdapter,PredictiveAdapter
from adapters import PDAAdapter
import asyncio
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from PDA import PredictiveDialogAgent
import os # For path operations
import json # For hashing triples if needed
import traceback # For detailed error logging (added for debugging)

# Define a constant for the memory directory for core TN/seRNN knowledge
CORE_MEMORY_DIR = "core_tn_sernn_memory"

@dataclass
class ProcessingResult:
    """处理结果数据类"""
    original_input: Any
    triples: List
    compressed_vectors: np.ndarray # Actually torch.Tensor from TNAdapter, but kept np for ProcessingResult consistency
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

        # Core TN/seRNN knowledge base
        self.knowledge_base_vectors: Dict[str, torch.Tensor] = {}
        self.memory_base_path = CORE_MEMORY_DIR
        self.kb_file_path = os.path.join(self.memory_base_path, "knowledge_base.pt")
        self.tn_model_file_path = os.path.join(self.memory_base_path, "tn_model_state.pth")
        self.sernn_model_file_path = os.path.join(self.memory_base_path, "sernn_model_state.pth")

        # Ensure memory directory exists
        os.makedirs(self.memory_base_path, exist_ok=True)

        # Initialize modules (TN, SeRNN etc.) first, then try to load their states
        self._initialize_modules()
        self._initialize_adapters()
        self._build_pipeline()

        # Load core knowledge (TN/seRNN states and KB vectors)
        self._load_core_knowledge() # This already has the DEBUG logging from previous attempts

        # 集成sentenceTransformer和deepseek
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.deepseek_client = AsyncOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=self.config.llm_config.api_key
        )
        # 集成PDA对话agent
        self.pda_agent = PredictiveDialogAgent(
            deepseek_api_key=self.config.llm_config.api_key,
            integrated_system_ref=self # Pass reference to self
        )

    def _initialize_modules(self):
        """初始化各个模块"""
        from TN import TripleCompressor
        from seRNN import SeRNN
        from PredictiveCoding import PredictiveCodingAgent # Assuming this is general purpose

        if self.config.pipeline_config['use_tn']:
            self.modules['tn'] = TripleCompressor(**self.config.tn_config)

        if self.config.pipeline_config['use_sernn']:
            self.modules['sernn'] = SeRNN(**self.config.sernn_config)

        if self.config.pipeline_config['use_predictive']:
            pcfg = self.config.predictive_config
            # This PredictiveCodingAgent might be different from PDA's pc_core
            # Or could be related to how new knowledge is added to self.knowledge_base_vectors
            self.modules['predictive_coding_module'] = PredictiveCodingAgent(
                num_inputs=pcfg.input_dim, # Ensure these config names are correct
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

        if 'predictive_coding_module' in self.modules:
            self.adapters['predictive'] = PredictiveAdapter(self.modules['predictive_coding_module'])

    def _build_pipeline(self):
        """构建处理管道"""
        if self.config.pipeline_config['use_tn']:
            self.pipeline.append(('triple_compression', self._process_tn))

        if self.config.pipeline_config['use_sernn']:
            self.pipeline.append(('spatial_embedding', self._process_sernn))

        if self.config.pipeline_config['use_predictive'] and 'predictive_coding_module' in self.modules :
            self.pipeline.append(('predictive_coding_processing', self._process_predictive_module))

    def _get_triple_str_key(self, triple: tuple) -> str:
        """Generates a canonical string key for a triple."""
        return json.dumps(sorted(triple), ensure_ascii=False)

    def _process_tn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """TN处理步骤. Also updates the persistent knowledge_base_vectors."""
        triples = data.get('triples', [])
        print(f"DEBUG_STORAGE: _process_tn received initial triples: {triples}")

        if not triples or 'tn' not in self.adapters:
            print(f"DEBUG_STORAGE: No triples provided or TN adapter not found. Triples: {triples}, TN adapter exists: {'tn' in self.adapters}")
            data['compressed_vectors'] = torch.empty(0)
            return data

        try:
            print(f"DEBUG_STORAGE: Calling prepare_triples_for_compression with: {triples}")
            formatted_triples_ids = self.adapters['tn'].prepare_triples_for_compression(triples)
            print(f"DEBUG_STORAGE: formatted_triples_ids from adapter: {formatted_triples_ids}, Type: {type(formatted_triples_ids)}")

            if not isinstance(formatted_triples_ids, torch.Tensor) or formatted_triples_ids.numel() == 0:
                print("DEBUG_STORAGE: No valid formatted_triples_ids tensor returned from adapter or it's empty. Skipping compression and storage.")
                data['compressed_vectors'] = torch.empty(0)
                return data

            print(f"DEBUG_STORAGE: Calling compress_and_flatten with formatted_triples_ids shape: {formatted_triples_ids.shape}")
            compressed_triplet_vectors = self.adapters['tn'].compress_and_flatten(formatted_triples_ids)
            print(f"DEBUG_STORAGE: compressed_triplet_vectors from adapter shape: {compressed_triplet_vectors.shape if isinstance(compressed_triplet_vectors, torch.Tensor) else 'None'}, Type: {type(compressed_triplet_vectors)}")

            if not isinstance(compressed_triplet_vectors, torch.Tensor) or compressed_triplet_vectors.numel() == 0:
                print("DEBUG_STORAGE: compressed_triplet_vectors is not a tensor or is empty after compress_and_flatten. Skipping storage.")
                data['compressed_vectors'] = torch.empty(0)
                return data

            data['compressed_vectors'] = compressed_triplet_vectors

            with torch.no_grad():
                for i, original_triple_tuple in enumerate(triples):
                    if i >= len(compressed_triplet_vectors):
                        print(f"DEBUG_STORAGE: Warning - more original triples ({len(triples)}) than compressed vectors ({len(compressed_triplet_vectors)}). Index i={i} is out of bounds.")
                        break

                    triple_key = self._get_triple_str_key(original_triple_tuple)
                    current_vector = compressed_triplet_vectors[i].detach().clone()
                    print(f"DEBUG_STORAGE: Processing triple_key: {triple_key}, original_triple: {original_triple_tuple}, current_vector shape: {current_vector.shape}, first 3 elements: {current_vector[:3] if current_vector.numel() > 0 else 'empty tensor'}")

                    store_vector = False
                    prediction_loss = 0.0

                    has_pc_module = 'predictive_coding_module' in self.modules
                    print(f"DEBUG_STORAGE: PC Module exists in self.modules: {has_pc_module}.")

                    if has_pc_module and hasattr(self.config.predictive_config, 'storage_threshold'):
                        actual_storage_threshold = self.config.predictive_config.storage_threshold
                        print(f"DEBUG_STORAGE: Using 'storage_threshold' from self.config.predictive_config: {actual_storage_threshold} (type: {type(actual_storage_threshold)})")

                        pc_module = self.modules['predictive_coding_module']
                        pc_input_device = next(pc_module.parameters()).device
                        pc_input = current_vector.to(pc_input_device).unsqueeze(0).unsqueeze(0)
                        print(f"DEBUG_STORAGE: PC_Module input device: {pc_input.device}, shape: {pc_input.shape}")

                        try:
                            _, loss_tensor, metrics = pc_module(pc_input)
                            prediction_loss = loss_tensor.item()
                            print(f"DEBUG_STORAGE: PC_Module - Triple: {triple_key}, Prediction Loss: {prediction_loss:.4f}, Threshold: {actual_storage_threshold}")

                            if prediction_loss > actual_storage_threshold:
                                store_vector = True
                                print(f"DEBUG_STORAGE: PC_Module decision: High prediction loss. Marking for storage.")
                            else:
                                print(f"DEBUG_STORAGE: PC_Module decision: Low prediction loss. Not storing based on threshold.")
                        except Exception as pc_error:
                            print(f"DEBUG_STORAGE: Error during predictive_coding_module processing for {triple_key}: {pc_error}")
                            traceback.print_exc()
                            if triple_key not in self.knowledge_base_vectors:
                                store_vector = True
                                print(f"DEBUG_STORAGE: Storing {triple_key} due to PC module error and novelty.")
                    else:
                        missing_reasons = []
                        if not has_pc_module: missing_reasons.append("PC module not in self.modules")
                        if not hasattr(self.config, 'predictive_config'): missing_reasons.append("self.config.predictive_config does not exist")
                        elif not hasattr(self.config.predictive_config, 'storage_threshold'):
                            missing_reasons.append(f"'storage_threshold' attribute not found in self.config.predictive_config (type: {type(self.config.predictive_config)})")
                        print(f"DEBUG_STORAGE: PC module-based storage skipped. Reasons: {'; '.join(missing_reasons)}. Fallback logic activated for triple: {triple_key}")

                        if triple_key not in self.knowledge_base_vectors:
                            store_vector = True
                            print(f"DEBUG_STORAGE: Fallback: Storing new triple: {triple_key} as it's not in KB.")
                        else:
                            print(f"DEBUG_STORAGE: Fallback: Triple {triple_key} already in KB. Not storing.")

                    if store_vector:
                        self.knowledge_base_vectors[triple_key] = current_vector.cpu()
                        print(f"DEBUG_STORAGE: Successfully stored vector for triple: {triple_key}. KB size now: {len(self.knowledge_base_vectors)}")
                    else:
                        print(f"DEBUG_STORAGE: Vector for triple {triple_key} was NOT stored. KB size: {len(self.knowledge_base_vectors)}")

                    if hasattr(self.pda_agent, 'current_prediction_error'):
                         self.pda_agent.current_prediction_error = prediction_loss

            print(f"DEBUG_STORAGE: TN processing completed for this batch. Triples processed: {len(triples)}. Final KB size: {len(self.knowledge_base_vectors)}")

        except Exception as e:
            print(f"DEBUG_STORAGE: Major error in _process_tn for triples {triples}: {e}")
            traceback.print_exc()
            data['compressed_vectors'] = torch.empty(0)

        return data

    def _process_sernn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """SeRNN处理步骤"""
        # Input should be torch.Tensor from _process_tn
        compressed_vectors = data.get('compressed_vectors', torch.empty(0))

        if compressed_vectors.numel() == 0 or 'sernn' not in self.adapters:
            print("Warning: No compressed vectors or SeRNN adapter for SeRNN processing")
            data['spatial_embeddings'] = torch.tensor([])
            return data

        try:
            sernn_input = compressed_vectors.unsqueeze(1)
            processed_output_from_adapter = self.adapters['sernn'].process_vectors(compressed_vectors)
            data['spatial_embeddings'] = processed_output_from_adapter
            print(f"SeRNN processing completed. Output shape: {data['spatial_embeddings'].shape}")
        except Exception as e:
            print(f"Error in SeRNN processing: {e}")
            data['spatial_embeddings'] = torch.tensor([])
        return data

    def _process_predictive_module(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predictive Coding Module processing step"""
        spatial_embeddings = data.get('spatial_embeddings', torch.tensor([]))
        if spatial_embeddings.numel() == 0 or 'predictive' not in self.adapters:
            print("Warning: No spatial embeddings or predictive adapter for predictive processing")
            data['predictions'] = None
            return data
        try:
            prediction_result = self.adapters['predictive'].process_sernn_output(spatial_embeddings)
            data['predictions'] = prediction_result['predictions']
            print(f"Predictive Coding Module processing completed.")
        except Exception as e:
            print(f"Error in predictive coding module processing: {e}")
            data['predictions'] = None
        return data

    def process(self, input_data: Any, triples: Optional[List] = None) -> ProcessingResult:
        """主处理函数"""
        current_triples = triples or self._extract_triples_from_input(input_data)
        data = {
            'original_input': input_data,
            'triples': current_triples,
            'metadata': {'timestamp': torch.tensor(0.0), 'processing_steps': []}
        }
        for step_name, step_func in self.pipeline:
            try:
                print(f"Executing pipeline step: {step_name}")
                data = step_func(data)
                data['metadata']['processing_steps'].append(step_name)
            except Exception as e:
                print(f"Critical error in pipeline step {step_name}: {e}")
                break
        return ProcessingResult(
            original_input=data['original_input'],
            triples=data['triples'],
            compressed_vectors=data.get('compressed_vectors', torch.empty(0)).cpu().numpy() if isinstance(data.get('compressed_vectors'), torch.Tensor) else data.get('compressed_vectors', np.array([])),
            spatial_embeddings=data.get('spatial_embeddings', torch.tensor([])),
            predictions=data.get('predictions'),
            metadata=data['metadata']
        )

    def _extract_triples_from_input(self, input_data: Any) -> List[Tuple[str, str, str]]:
        """从输入数据中提取三元组，优先用deepseek LLM自动抽取"""
        if not isinstance(input_data, str):
            if isinstance(input_data, list):
                return input_data
            return []

        prompt = f"""
        请从以下中文文本中抽取所有可识别的知识三元组（主语-谓语-宾语）。
        每个三元组请严格按照JSON对象格式表示，包含 "subject"（主体）、"relation"（关系）和 "object"（客体）三个键。
        将所有抽取的JSON对象组成一个JSON数组返回。如果找不到任何三元组，请返回一个空数组 `[]`。

        例如:
        输入：小明在公园里踢足球，天气很好。
        输出：[
            {{"subject": "小明", "relation": "在公园里踢", "object": "足球"}}
        ]

        输入：地球是行星。太阳是恒星。
        输出：[
            {{"subject": "地球", "relation": "是", "object": "行星"}},
            {{"subject": "太阳", "relation": "是", "object": "恒星"}}
        ]

        现在请从以下文本中提取三元组：
            {input_data}
            """

        async def call_deepseek_for_triples():
            try:
                response = await self.deepseek_client.chat.completions.create(
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
                        # Enhanced parsing: Check if item is a list of 3 strings (potential fallback from LLM)
                        elif isinstance(item, list) and len(item) == 3 and all(isinstance(x, str) for x in item):
                            print(f"Warning: LLM returned a list-formatted triple {item}, converting to tuple.")
                            extracted_tuples.append((item[0], item[1], item[2]))
                        else:
                            print(f"Warning: LLM returned a malformed triple item: {item}")
                    return extracted_tuples
                else:
                    print(f"Warning: LLM did not return a list of triples. Got: {triple_list_json}")
                    return []

            except json.JSONDecodeError as e:
                print(f"LLM三元组抽取失败 (JSON Decode Error): {e}. Response: {result_content}")
                return []
            except Exception as e:
                print(f"LLM三元组抽取失败: {e}")
                traceback.print_exc() # Added for more detail on other exceptions
                return []

        return asyncio.run(call_deepseek_for_triples())

    def batch_process(self, input_batch: List[Any]) -> List[ProcessingResult]:
        """批处理功能"""
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
                'tn_config_keys': list(self.config.tn_config.keys()) if 'tn' in self.modules else [],
                'sernn_config_keys': list(self.config.sernn_config.keys()) if 'sernn' in self.modules else [],
            },
            'knowledge_base_vector_count': len(self.knowledge_base_vectors)
        }

    def chat_with_agent(self, user_input: str) -> dict:
        """与PDA智能体进行多轮对话，返回AI回复、记忆统计、思维链、认知误差"""
        print(f"User input for chat: {user_input}")
        print("Processing input to update core knowledge base before chat response...")
        _ = self.process(user_input)

        import asyncio
        reply = asyncio.run(self.pda_agent.dialog_round(user_input))

        return {
            "reply": reply,
            "pda_memory_stats": self.pda_agent.get_memory_stats(),
            "core_kb_vector_count": len(self.knowledge_base_vectors),
            "pda_thought_chain": self.pda_agent.dialog_buffer.chain_text(),
            "pda_prediction_error": self.pda_agent.current_prediction_error
        }

    def _save_core_knowledge(self):
        """Saves the TN/seRNN knowledge base and model states."""
        print(f"Attempting to save core knowledge to {self.memory_base_path}...")
        try:
            cpu_knowledge_base = {k: v.cpu() for k, v in self.knowledge_base_vectors.items()}
            torch.save(cpu_knowledge_base, self.kb_file_path)
            print(f"  Knowledge base vectors saved to {self.kb_file_path} ({len(self.knowledge_base_vectors)} items).")

            if 'tn' in self.modules and hasattr(self.modules['tn'], 'state_dict'):
                torch.save(self.modules['tn'].state_dict(), self.tn_model_file_path)
                print(f"  TN model state saved to {self.tn_model_file_path}.")

            if 'sernn' in self.modules and hasattr(self.modules['sernn'], 'state_dict'):
                torch.save(self.modules['sernn'].state_dict(), self.sernn_model_file_path)
                print(f"  SeRNN model state saved to {self.sernn_model_file_path}.")

            print("Core knowledge saved successfully.")
        except Exception as e:
            print(f"Error saving core knowledge: {e}")
            traceback.print_exc()


    def _load_core_knowledge(self):
        """Loads the TN/seRNN knowledge base and model states."""
        print(f"Attempting to load core knowledge from {self.memory_base_path}...")
        try:
            if os.path.exists(self.kb_file_path):
                self.knowledge_base_vectors = torch.load(self.kb_file_path)
                print(f"DEBUG: Successfully loaded knowledge base from {self.kb_file_path}.") # Retained DEBUG log
                print(f"DEBUG: Number of items loaded: {len(self.knowledge_base_vectors)}") # Retained DEBUG log
                if self.knowledge_base_vectors:
                    print("DEBUG: Sample keys from loaded knowledge base:") # Retained DEBUG log
                    count = 0
                    for key in self.knowledge_base_vectors.keys():
                        print(f"  - {key}")
                        count += 1
                        if count >= 5: break
                else: print("DEBUG: Knowledge base is empty after loading.") # Retained DEBUG log
            else:
                print(f"DEBUG: Knowledge base file not found: {self.kb_file_path}. Starting with empty KB.") # Retained DEBUG log
                self.knowledge_base_vectors = {}

            if 'tn' in self.modules and os.path.exists(self.tn_model_file_path):
                try:
                    self.modules['tn'].load_state_dict(torch.load(self.tn_model_file_path, map_location=self.modules['tn'].device))
                    self.modules['tn'].eval()
                    print(f"  TN model state loaded from {self.tn_model_file_path}.")
                except Exception as e:
                    print(f"  Error loading TN model state (file might be incompatible or model structure changed): {e}")
                    traceback.print_exc()
            elif 'tn' in self.modules:
                 print(f"  TN model state file not found: {self.tn_model_file_path}. TN model uses initial weights.")

            if 'sernn' in self.modules and os.path.exists(self.sernn_model_file_path):
                try:
                    self.modules['sernn'].load_state_dict(torch.load(self.sernn_model_file_path, map_location=self.modules['sernn'].device))
                    self.modules['sernn'].eval()
                    print(f"  SeRNN model state loaded from {self.sernn_model_file_path}.")
                except Exception as e:
                    print(f"  Error loading SeRNN model state (file might be incompatible or model structure changed): {e}")
                    traceback.print_exc()
            elif 'sernn' in self.modules:
                print(f"  SeRNN model state file not found: {self.sernn_model_file_path}. SeRNN model uses initial weights.")

            print("Core knowledge loading process completed.")
        except Exception as e:
            print(f"Error loading core knowledge: {e}")
            traceback.print_exc()
            self.knowledge_base_vectors = {}

    def save_all_memory(self):
        """Saves Core TN/seRNN knowledge. PDA no longer has separate memory to save."""
        print("Saving Core TN/seRNN knowledge...")
        self._save_core_knowledge()
        print("All memory saving routines complete.")

    def get_core_knowledge_vector_by_key(self, triple_key: str) -> Optional[torch.Tensor]:
        """Retrieves a specific vector from the core knowledge base by its triple key."""
        return self.knowledge_base_vectors.get(triple_key)

    def query_core_knowledge_base(self, query_text: Optional[str] = None, query_vector: Optional[torch.Tensor] = None, top_k: int = 5) -> List[Tuple[str, torch.Tensor, float]]:
        print(f"DEBUG_RECALL: query_core_knowledge_base called. Query_text: '{query_text}', Query_vector shape: {query_vector.shape if query_vector is not None else 'None'}, Top_k: {top_k}")

        if query_vector is None:
            print(f"DEBUG_RECALL: query_core_knowledge_base: query_vector is None. Attempting to use query_text: '{query_text}'")
            if query_text is None or 'tn' not in self.modules:
                print("DEBUG_RECALL: Error: Query text provided but TN module not available for vectorization, or no query provided.")
                return []

            print("DEBUG_RECALL: Warning: Querying by text directly in query_core_knowledge_base is not fully implemented and may not be reliable. Provide a query_vector.")
            try:
                query_triples = self._extract_triples_from_input(query_text)
                if not query_triples:
                    print(f"DEBUG_RECALL: No triples extracted from query_text '{query_text}' for direct query.")
                    return []

                temp_formatted_ids = self.adapters['tn'].prepare_triples_for_compression([query_triples[0]])
                if not isinstance(temp_formatted_ids, torch.Tensor) or temp_formatted_ids.numel() == 0:
                     print(f"DEBUG_RECALL: Failed to format query triple from text for direct query: {query_triples[0]}")
                     return []
                temp_query_vector = self.adapters['tn'].compress_and_flatten(temp_formatted_ids)
                if not isinstance(temp_query_vector, torch.Tensor) or temp_query_vector.numel() == 0:
                    print(f"DEBUG_RECALL: Failed to compress query triple from text for direct query: {query_triples[0]}")
                    return []
                query_vector = temp_query_vector[0].detach().clone()
                print(f"DEBUG_RECALL: Derived query_vector from text. Shape: {query_vector.shape if query_vector is not None else 'None'}")
            except Exception as e:
                print(f"DEBUG_RECALL: Error deriving query_vector from text in query_core_knowledge_base: {e}")
                traceback.print_exc()
                return []

        if query_vector is None:
            print("DEBUG_RECALL: query_vector is still None after attempting text derivation. Returning empty.")
            return []

        if not self.knowledge_base_vectors:
            print("DEBUG_RECALL: Knowledge base is empty. Returning empty list.")
            return []

        print(f"DEBUG_RECALL: Current knowledge_base_vectors size: {len(self.knowledge_base_vectors)}")
        query_vector_cpu = query_vector.cpu()

        similarities = []
        for triple_key, kb_vector in self.knowledge_base_vectors.items():
            kb_vector_cpu = kb_vector.cpu()
            similarity = torch.cosine_similarity(query_vector_cpu.unsqueeze(0), kb_vector_cpu.unsqueeze(0)).item()
            similarities.append((triple_key, kb_vector_cpu, similarity))

        similarities.sort(key=lambda x: x[2], reverse=True)
        top_results = similarities[:top_k]
        print(f"DEBUG_RECALL: Top {top_k} similar items (Key, Similarity):")
        for key, _, sim_score in top_results:
            print(f"  - Key: {key}, Similarity: {sim_score:.4f}")
        if not top_results:
            print("DEBUG_RECALL: No similar items found.")
        return top_results

    async def get_tn_query_vector(self, query_text: str) -> Optional[torch.Tensor]:
        print(f"DEBUG_RECALL: get_tn_query_vector called with query_text: '{query_text}'")
        if 'tn' not in self.modules or 'tn' not in self.adapters:
            print("DEBUG_RECALL: Error: TN module/adapter not available for query vectorization.")
            return None

        prompt = (
            "请从下面的文本中抽取所有知识三元组（主体, 关系, 客体）。"
            "每个三元组请严格按照JSON对象 `{\"subject\": \"主体\", \"relation\": \"关系\", \"object\": \"客体\"}` 的格式表示。"
            "当用户语法并不完整时,llm自己根据语义判断主体,客体和关系."
            "将所有这些JSON对象组成一个JSON数组返回。如果找不到三元组，返回空数组 `[]`。"
            f"文本内容：\n{query_text}"
        )
        query_triples_tuples = []
        try:
            print(f"DEBUG_RECALL: Calling LLM for triple extraction from query_text: '{query_text}'")
            response = await self.deepseek_client.chat.completions.create(
                model=self.config.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个精确的知识抽取引擎。严格按照用户要求的JSON格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, max_tokens=self.config.llm_config.max_tokens,
                response_format={"type": "json_object"}
            )
            result_content = response.choices[0].message.content.strip()
            print(f"DEBUG_RECALL: LLM raw response for query triple extraction: {result_content}")
            parsed_json = json.loads(result_content)
            triple_list_json = parsed_json.get("triples", parsed_json) if isinstance(parsed_json, dict) else parsed_json

            if isinstance(triple_list_json, list):
                for item in triple_list_json:
                    if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                        query_triples_tuples.append((str(item["subject"]), str(item["relation"]), str(item["object"])))
            print(f"DEBUG_RECALL: Extracted query_triples_tuples: {query_triples_tuples}")
        except Exception as e:
            print(f"DEBUG_RECALL: LLM triple extraction for query failed: {e}")
            traceback.print_exc()
            return None

        if not query_triples_tuples:
            print(f"DEBUG_RECALL: No triples extracted from query text: '{query_text}' for vectorization.")
            return None

        try:
            first_triple_to_process = [query_triples_tuples[0]]
            print(f"DEBUG_RECALL: Processing first query triple with TN: {first_triple_to_process}")

            formatted_triples_ids = self.adapters['tn'].prepare_triples_for_compression(first_triple_to_process)
            print(f"DEBUG_RECALL: formatted_triples_ids for query triple: {formatted_triples_ids}")
            if not isinstance(formatted_triples_ids, torch.Tensor) or formatted_triples_ids.numel() == 0:
                print(f"DEBUG_RECALL: Could not format query triple for TN (or empty tensor): {first_triple_to_process[0]}")
                return None

            compressed_vector = self.adapters['tn'].compress_and_flatten(formatted_triples_ids)
            print(f"DEBUG_RECALL: compressed_vector for query triple shape: {compressed_vector.shape if isinstance(compressed_vector, torch.Tensor) else 'None'}")

            if isinstance(compressed_vector, torch.Tensor) and compressed_vector.numel() > 0:
                final_query_vector = compressed_vector[0].detach().clone()
                print(f"DEBUG_RECALL: Successfully generated query vector. Shape: {final_query_vector.shape}, First 3 elements: {final_query_vector[:3] if final_query_vector.numel() > 0 else 'empty tensor'}")
                return final_query_vector
            else:
                print(f"DEBUG_RECALL: TN module produced an empty or invalid vector for query triple: {first_triple_to_process[0]}")
                return None
        except Exception as e:
            print(f"DEBUG_RECALL: Error processing query triple with TN: {e}")
            traceback.print_exc()
            return None

# Ensure a newline character at the very end of the file


        

