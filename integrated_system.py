# integrated_system.py
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
        self._load_core_knowledge()

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
        # PDA's internal memory system (GraphMemoryBank, pc_core, memory_bank) is removed.
        # self.memory_bank = self.pda_agent.memory_bank # Removed
        # self.pc_core = self.pda_agent.pc_core # Removed
        
        # PDA's dialog_buffer is still internal to PDA and can be accessed if needed,
        # but not typically aliased here unless IntegratedSystem directly manipulates it.
        # For now, assume dialog_buffer access is via pda_agent if necessary.
        # self.dialog_buffer = self.pda_agent.dialog_buffer # Retained if needed for other parts of IntegratedSystem
        
        # self.pda_agent.load_memory() # Removed, PDA no longer loads its own separate memory
    
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
        
        # PDA adapter might not be used if PDA is self-contained for chat
        # if self.config.pipeline_config.get('use_pda', False):
        #     self.adapters['pda'] = PDAAdapter(self.config) # Re-evaluate if this is needed
    
    def _build_pipeline(self):
        """构建处理管道"""
        if self.config.pipeline_config['use_tn']:
            self.pipeline.append(('triple_compression', self._process_tn))
        
        if self.config.pipeline_config['use_sernn']:
            self.pipeline.append(('spatial_embedding', self._process_sernn))
        
        # The general 'predictive_coding' step might use the 'predictive_coding_module'
        # This is distinct from PDA's internal predictive coding.
        if self.config.pipeline_config['use_predictive'] and 'predictive_coding_module' in self.modules :
            self.pipeline.append(('predictive_coding_processing', self._process_predictive_module))
        
        # 'pda' step might be removed if chat is handled separately
        # if self.config.pipeline_config.get('use_pda', False):
        #     self.pipeline.append(('pda_processing', self._process_pda_module))

    def _get_triple_str_key(self, triple: tuple) -> str:
        """Generates a canonical string key for a triple."""
        return json.dumps(sorted(triple), ensure_ascii=False)

    def _process_tn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """TN处理步骤. Also updates the persistent knowledge_base_vectors."""
        triples = data.get('triples', [])
        
        if not triples or 'tn' not in self.adapters:
            print("Warning: No triples or TN adapter for TN processing")
            data['compressed_vectors'] = torch.empty(0) 
            return data
        
        try:
            formatted_triples_ids = self.adapters['tn'].prepare_triples_for_compression(triples)
            # TNAdapter.compress_and_flatten now returns a tensor
            compressed_triplet_vectors = self.adapters['tn'].compress_and_flatten(formatted_triples_ids) 
            
            data['compressed_vectors'] = compressed_triplet_vectors # Should be torch.Tensor

            # Update persistent knowledge base with new triple embeddings
            # Assuming formatted_triples_ids corresponds row-wise to compressed_triplet_vectors
            # And 'triples' are the original string triples
            with torch.no_grad(): # Ensure no gradient tracking for KB update
                for i, original_triple_tuple in enumerate(triples):
                    triple_key = self._get_triple_str_key(original_triple_tuple)
                    current_vector = compressed_triplet_vectors[i].detach().clone()

                    # --- Predictive Coding Placeholder ---
                    # The decision to store or update this vector in self.knowledge_base_vectors
                    # should be driven by a predictive coding mechanism.
                    # This mechanism would assess the novelty or significance of the new vector,
                    # possibly by comparing it against existing knowledge or predictions.
                    # For example:
                    #
                    # significance_signal = 0.0
                    # if 'predictive_coding_module' in self.modules:
                    #     # This module would need to be designed to output such a signal
                    #     # based on the new vector and possibly the current system state.
                    #     significance_signal = self.modules['predictive_coding_module'].assess_vector_significance(current_vector)
                    # elif hasattr(self.pda_agent, 'pc_core'):
                    #     # PDA's pc_core might also be adapted or used if its predict() method
                    #     # can be mapped to this context.
                    #     # This is less direct as pc_core currently works with different inputs.
                    #     pass # Placeholder for PDA pc_core logic
                    #
                    # # Example threshold-based storage:
                    # if significance_signal > some_threshold or triple_key not in self.knowledge_base_vectors:
                    #    self.knowledge_base_vectors[triple_key] = current_vector.cpu()
                    #    print(f"KB: Stored/Updated vector for triple: {triple_key} (Signal: {significance_signal:.2f})")
                    #
                    # --- Predictive Coding Logic ---
                    store_vector = False
                    prediction_loss = 0.0 # Default, updated if PC module runs

                    if 'predictive_coding_module' in self.modules and hasattr(self.config, 'predictive_config') and hasattr(self.config.predictive_config, 'storage_threshold'):
                        pc_module = self.modules['predictive_coding_module']
                        # PredictiveCodingAgent.forward expects input like [batch, seq_len, num_inputs]
                        # current_vector is [384], so reshape to [1, 1, 384]
                        pc_input = current_vector.to(pc_module.encoder[0].weight.device).unsqueeze(0).unsqueeze(0) # Ensure correct device
                        
                        try:
                            _, loss_tensor, metrics = pc_module(pc_input)
                            prediction_loss = loss_tensor.item() # Use total_loss or prediction_loss from metrics
                            
                            if prediction_loss > self.config.predictive_config.storage_threshold:
                                store_vector = True
                                print(f"KB: High prediction loss ({prediction_loss:.4f}) for {triple_key}. Marking for storage.")
                            else:
                                print(f"KB: Low prediction loss ({prediction_loss:.4f}) for {triple_key}. Not storing based on threshold.")
                        except Exception as pc_error:
                            print(f"Error during predictive_coding_module processing for {triple_key}: {pc_error}")
                            # Fallback: store if new and PC module fails
                            if triple_key not in self.knowledge_base_vectors:
                                store_vector = True
                                print(f"KB: Storing {triple_key} due to PC module error and novelty.")
                    
                    else: # Fallback if no predictive coding module or threshold
                        if triple_key not in self.knowledge_base_vectors:
                            store_vector = True
                            print(f"KB: No PC module/threshold, storing new triple: {triple_key}")

                    if store_vector:
                        self.knowledge_base_vectors[triple_key] = current_vector.cpu()
                        print(f"KB: Stored vector for triple: {triple_key}")
                    
                    # Update PDA's current_prediction_error with the latest loss
                    if hasattr(self.pda_agent, 'current_prediction_error'):
                         self.pda_agent.current_prediction_error = prediction_loss

                    # --- End Predictive Coding Logic ---
            
            # tn_features might be deprecated if not used, or ensure it handles tensors
            # data['tn_features'] = self.adapters['tn'].extract_spatial_features(compressed_triplet_vectors)
            
            print(f"TN processing completed: {len(triples)} triples processed into vectors.")
            
        except Exception as e:
            print(f"Error in TN processing: {e}")
            data['compressed_vectors'] = torch.empty(0)
            # data['tn_features'] = {}
        
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
            # Adapter should handle preparing for seRNN (e.g., batching, spatial positions if any)
            # Assuming tn_input is now directly the compressed_vectors tensor
            # The SeRNNAdapter needs to be robust to handle this.
            # Spatial positions need to be generated or passed if seRNN uses them.
            # For now, let's assume seRNN can process a batch of vectors.
            # This part needs careful review of SeRNNAdapter and SeRNN input requirements.
            
            # Placeholder for actual spatial positions if required by your seRNN version
            # This might come from data, or be default indices.
            batch_size, seq_len_dim = compressed_vectors.shape[0], compressed_vectors.shape[1] if compressed_vectors.ndim > 1 else 1
            if compressed_vectors.ndim == 1: # If it's a flat list of vectors from TN.
                 # This means TN output a single vector, or adapter flattened it.
                 # SeRNN typically expects sequences. This logic needs alignment.
                 # For now, assuming adapter handles if it's a single vector.
                 # If TN outputs [N, D], SeRNN might process this as a sequence of N items.
                 # Or adapter might batch it as [1, N, D] for SeRNN.
                 # Let's assume adapter prepares it correctly for seRNN.
                 # For now, if it's [N,D] from TN, we might treat N as batch for seRNN, each item a seq of 1.
                 # Or treat as 1 batch of N items.
                 # This is a critical point of how TN outputs are fed to SeRNN.
                 # The 'SeRNNAdapter.process_tn_output' needs to be well-defined.
                 # For now, let's assume it takes the vectors and any necessary spatial info.
                 # A simple case: if spatial_dim is part of seRNN config but not input, it might use internal indexing.
                 # Let's assume spatial_positions are not explicitly passed here for simplicity,
                 # relying on SeRNNAdapter or SeRNN defaults if needed.
                pass


            # The call to process_tn_output in SeRNNAdapter needs to be robust.
            # It might need to reshape compressed_vectors or generate spatial_positions.
            # For example, if compressed_vectors is [N, 384], it could be treated as [N, 1, 384]
            # with spatial_positions as torch.zeros(N, 1).
            # This depends on SeRNN's design.
            
            # Simplified:
            # spatial_output = self.adapters['sernn'].process_input_vectors(compressed_vectors)
            # This requires SeRNNAdapter to have such a method.
            # For now, using the existing adapter method, assuming it's compatible:
            
            # The current SeRNNAdapter.process_tn_output expects a dict with 'tn_output'
            # Let's make it simpler: adapter directly processes the tensor
            # This means SeRNNAdapter.process_tn_output needs to be updated or a new method created.
            # For now, we'll assume the adapter can handle the tensor directly.
            # This is a placeholder and likely needs SeRNNAdapter modification.
            
            # Let's assume `compressed_vectors` is [batch_size, feature_dim] from TN
            # SeRNN expects [batch_size, seq_len, input_size].
            # We might treat each vector as a sequence of length 1.
            sernn_input = compressed_vectors.unsqueeze(1) # [batch_size, 1, feature_dim]
            
            # Spatial positions would be [batch_size, 1]
            # This is a simplification. A real graph would have more complex spatial_positions.
            # Using zeros as placeholder if not otherwise defined by the adapter.
            # The actual spatial_positions should be meaningful for seRNN.
            num_sequences = sernn_input.shape[0]
            spatial_positions_for_sernn = torch.zeros(num_sequences, 1, dtype=torch.long, device=sernn_input.device)
            
            # The SeRNN model itself is called here.
            # The adapter might just pass through or do minimal formatting.
            # The SeRNN.forward method is complex.
            if 'sernn' in self.modules:
                 # output_sequence, aux_info = self.modules['sernn'](sernn_input, spatial_positions_for_sernn)
                 # data['spatial_embeddings'] = output_sequence.squeeze(1) # Assuming output is [batch, 1, dim]
                
                 # Using the adapter as intended:
                 # The adapter's process_tn_output should internally call self.model(...)
                 # and handle input/output shapes.
                 # For now, this is a conceptual placeholder as SeRNNAdapter needs to be robust.
                 # This call assumes SeRNNAdapter.process_tn_output can take the raw tensor.
                 # This is a major point of potential failure if adapter isn't ready.
                 
                 # Simpler: Assume adapter takes tensor and returns tensor
                 # This requires adapter modification from its current form in adapters.py (if it expects dict)
                processed_output_from_adapter = self.adapters['sernn'].process_vectors(compressed_vectors) # New method needed in adapter
                data['spatial_embeddings'] = processed_output_from_adapter

                # data['sernn_info'] = self.adapters['sernn'].extract_spatial_embeddings(spatial_output) # If adapter returns dict
                print(f"SeRNN processing completed. Output shape: {data['spatial_embeddings'].shape}")
            else:
                data['spatial_embeddings'] = torch.tensor([])


        except Exception as e:
            print(f"Error in SeRNN processing: {e}")
            data['spatial_embeddings'] = torch.tensor([])
            # data['sernn_info'] = {}
        
        return data

    # Renamed to avoid conflict with PDA's internal predictive coding (pc_core)
    def _process_predictive_module(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predictive Coding Module processing step"""
        spatial_embeddings = data.get('spatial_embeddings', torch.tensor([]))
        
        if spatial_embeddings.numel() == 0 or 'predictive' not in self.adapters:
            print("Warning: No spatial embeddings or predictive adapter for predictive processing")
            data['predictions'] = None # Or some other field like 'module_predictions'
            return data
        
        try:
            prediction_result = self.adapters['predictive'].process_sernn_output(spatial_embeddings)
            data['predictions'] = prediction_result['predictions'] # Or 'module_predictions'
            # data['prediction_confidence'] = prediction_result['confidence']
            # data['prediction_features'] = prediction_result['features']
            print(f"Predictive Coding Module processing completed.")
        except Exception as e:
            print(f"Error in predictive coding module processing: {e}")
            data['predictions'] = None
        return data

    # This was for a PDA adapter, which might not be used if chat is separate
    # def _process_pda_module(self, data: Dict[str, Any]) -> Dict[str, Any]: ...

    def process(self, input_data: Any, triples: Optional[List] = None) -> ProcessingResult:
        """主处理函数"""
        current_triples = triples or self._extract_triples_from_input(input_data)
        data = {
            'original_input': input_data,
            'triples': current_triples, # Triples as list of tuples
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
            predictions=data.get('predictions'), # or 'module_predictions'
            metadata=data['metadata']
        )

    def _extract_triples_from_input(self, input_data: Any) -> List[Tuple[str, str, str]]:
        """从输入数据中提取三元组，优先用deepseek LLM自动抽取"""
        if not isinstance(input_data, str):
            if isinstance(input_data, list): # Assuming list of pre-formatted triples
                return input_data
            return []

        prompt = f"""
        从以下中文文本中抽取三元组（主语-谓语-宾语）关系，以 (主体, 动作, 客体) 的格式返回一个 JSON 数组。例如：
        输入：小明在公园里踢足球。
        输出：[["小明", "踢", "足球"]]

        请从文本中提取所有可以识别出的三元组：
            {input_data}
            """
        
        async def call_deepseek_for_triples():
            try:
                response = await self.deepseek_client.chat.completions.create(
                    model=self.config.llm_config.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个精确的知识抽取引擎。严格按照用户要求的JSON格式输出。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1, # Lower temperature for more deterministic extraction
                    max_tokens=self.config.llm_config.max_tokens,
                    response_format={"type": "json_object"} # Request JSON output
                )
                result_content = response.choices[0].message.content.strip()
                
                # Expecting result_content to be a stringified JSON like: '{"triples": [{"subject": "s", "relation": "r", "object": "o"}, ...]}'
                # Or directly a stringified JSON array: '[{"subject": "s", ...}]'
                parsed_json = json.loads(result_content)

                # Check if the response is a dict with a 'triples' key or directly a list
                triple_list_json = parsed_json.get("triples", parsed_json) if isinstance(parsed_json, dict) else parsed_json

                if isinstance(triple_list_json, list):
                    extracted_tuples = []
                    for item in triple_list_json:
                        if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                            extracted_tuples.append((str(item["subject"]), str(item["relation"]), str(item["object"])))
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
                return []
        
        return asyncio.run(call_deepseek_for_triples())

    def batch_process(self, input_batch: List[Any]) -> List[ProcessingResult]:
        """批处理功能"""
        results = []
        # Note: batch_size from config is not used here, processes one by one.
        # If batching is desired for TN/seRNN, process method needs to handle batches.
        for item_input in input_batch:
            if isinstance(item_input, list): # Assume it's a list of triples for one item
                results.append(self.process(None, triples=item_input))
            else: # Assume it's text input
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
        # Optionally, process user_input to extract triples and update core knowledge base
        # This makes the chat interactive with the TN/seRNN memory
        print(f"User input for chat: {user_input}")
        print("Processing input to update core knowledge base before chat response...")
        _ = self.process(user_input) # Process input, updates self.knowledge_base_vectors

        # Now, PDA can generate a response. Its internal memory might also be used.
        # If PDA needs to query self.knowledge_base_vectors, it needs a method to do so.
        import asyncio
        reply = asyncio.run(self.pda_agent.dialog_round(user_input)) # PDA's own dialog logic
        
        return {
            "reply": reply,
            # pda_memory_stats now gets stats from core KB via integrated_system_ref
            "pda_memory_stats": self.pda_agent.get_memory_stats(), 
            "core_kb_vector_count": len(self.knowledge_base_vectors), # Also available in pda_memory_stats
            "pda_thought_chain": self.pda_agent.dialog_buffer.chain_text(), # Accessing dialog_buffer via pda_agent
            "pda_prediction_error": self.pda_agent.current_prediction_error # Accessing placeholder error from pda_agent
        }

    def _save_core_knowledge(self):
        """Saves the TN/seRNN knowledge base and model states."""
        print(f"Attempting to save core knowledge to {self.memory_base_path}...")
        try:
            # Save knowledge base vectors
            # Move all tensors in knowledge_base_vectors to CPU before saving
            cpu_knowledge_base = {k: v.cpu() for k, v in self.knowledge_base_vectors.items()}
            torch.save(cpu_knowledge_base, self.kb_file_path)
            print(f"  Knowledge base vectors saved to {self.kb_file_path} ({len(self.knowledge_base_vectors)} items).")

            # Save TN model state if module exists
            if 'tn' in self.modules and hasattr(self.modules['tn'], 'state_dict'):
                torch.save(self.modules['tn'].state_dict(), self.tn_model_file_path)
                print(f"  TN model state saved to {self.tn_model_file_path}.")
            
            # Save SeRNN model state if module exists
            if 'sernn' in self.modules and hasattr(self.modules['sernn'], 'state_dict'):
                torch.save(self.modules['sernn'].state_dict(), self.sernn_model_file_path)
                print(f"  SeRNN model state saved to {self.sernn_model_file_path}.")
            
            print("Core knowledge saved successfully.")
        except Exception as e:
            print(f"Error saving core knowledge: {e}")

    def _load_core_knowledge(self):
        """Loads the TN/seRNN knowledge base and model states."""
        print(f"Attempting to load core knowledge from {self.memory_base_path}...")
        try:
            # Load knowledge base vectors
            if os.path.exists(self.kb_file_path):
                self.knowledge_base_vectors = torch.load(self.kb_file_path)
                # Ensure vectors are on the correct device if needed after loading, though CPU is fine for storage.
                # If processing happens on GPU, they might be moved there.
                print(f"  Knowledge base vectors loaded from {self.kb_file_path} ({len(self.knowledge_base_vectors)} items).")
            else:
                print(f"  Knowledge base file not found: {self.kb_file_path}. Starting with empty KB.")
                self.knowledge_base_vectors = {}

            # Load TN model state if module exists
            if 'tn' in self.modules and os.path.exists(self.tn_model_file_path):
                try:
                    self.modules['tn'].load_state_dict(torch.load(self.tn_model_file_path, map_location=self.modules['tn'].device))
                    self.modules['tn'].eval() # Set to evaluation mode
                    print(f"  TN model state loaded from {self.tn_model_file_path}.")
                except Exception as e:
                    print(f"  Error loading TN model state (file might be incompatible or model structure changed): {e}")
            elif 'tn' in self.modules:
                 print(f"  TN model state file not found: {self.tn_model_file_path}. TN model uses initial weights.")
            
            # Load SeRNN model state if module exists
            if 'sernn' in self.modules and os.path.exists(self.sernn_model_file_path):
                try:
                    self.modules['sernn'].load_state_dict(torch.load(self.sernn_model_file_path, map_location=self.modules['sernn'].device))
                    self.modules['sernn'].eval() # Set to evaluation mode
                    print(f"  SeRNN model state loaded from {self.sernn_model_file_path}.")
                except Exception as e:
                    print(f"  Error loading SeRNN model state (file might be incompatible or model structure changed): {e}")
            elif 'sernn' in self.modules:
                print(f"  SeRNN model state file not found: {self.sernn_model_file_path}. SeRNN model uses initial weights.")
                
            print("Core knowledge loading process completed.")
        except Exception as e:
            print(f"Error loading core knowledge: {e}")
            self.knowledge_base_vectors = {} # Ensure clean state on error

    def save_all_memory(self):
        """Saves Core TN/seRNN knowledge. PDA no longer has separate memory to save."""
        # Save Core TN/seRNN knowledge
        print("Saving Core TN/seRNN knowledge...")
        self._save_core_knowledge()
        
        print("All memory saving routines complete.")

    def get_core_knowledge_vector_by_key(self, triple_key: str) -> Optional[torch.Tensor]:
        """Retrieves a specific vector from the core knowledge base by its triple key."""
        return self.knowledge_base_vectors.get(triple_key)

    def query_core_knowledge_base(self, query_text: Optional[str] = None, query_vector: Optional[torch.Tensor] = None, top_k: int = 5) -> List[Tuple[str, torch.Tensor, float]]:
        """
        Queries the core knowledge base for relevant triple embeddings.
        Accepts either a query_text (which will be converted to a vector using TN)
        or a pre-computed query_vector.
        Returns a list of (triple_key, embedding_tensor, similarity_score).
        """
        if query_vector is None:
            if query_text is None or 'tn' not in self.modules:
                print("Error: Query text provided but TN module not available for vectorization, or no query provided.")
                return []
            
            # Convert query_text to triples (e.g., by forming a dummy triple or LLM extraction)
            # For simplicity, let's assume query_text itself can be a basis for a special triple,
            # or we extract triples from it. This part needs careful design for effective querying.
            # Simplistic approach: treat query_text as a single "query" entity.
            # This is highly conceptual for querying. A better way is to extract actual triples from query_text.
            # For now, let's assume query_text is a triple string representation that can be vectorized.
            # This part is complex: how to get a query_vector from query_text that is comparable to stored triple embeddings?
            # Option 1: LLM extracts a triple from query_text, then we get its TN vector.
            # Option 2: A special "query" processing by TN.
            # For now, this example assumes query_vector must be provided if query_text is not a direct key.
            # This function needs more robust query vector generation.
            
            # Placeholder: If query_text is a direct key:
            # vec = self.get_core_knowledge_vector_by_key(query_text)
            # if vec is not None: query_vector = vec.to(self.modules['tn'].device) # Move to TN's device for comparison
            # else: print("Could not convert query_text to vector directly."); return []

            # A more realistic approach for query_text:
            # 1. Extract triples from query_text
            query_triples = self._extract_triples_from_input(query_text)
            if not query_triples:
                print(f"No triples extracted from query text: '{query_text}'")
                return []
            
            # 2. For simplicity, use the vector of the first extracted triple as the query vector.
            #    A more advanced method might average vectors or use a more sophisticated approach.
            #    This also assumes the triple is already in a format that prepare_triples_for_compression can handle.
            #    The prepare_triples_for_compression needs actual IDs, not strings.
            #    This entire query vector generation from text is non-trivial.
            
            # Let's assume for now that if query_text is provided, we expect it to be a key,
            # or that the PDA should provide a query_vector.
            # This method will primarily focus on similarity search if a query_vector is given.
            print("Warning: Querying by text is complex and not fully implemented. Provide a query_vector.")
            return [] # Fallback for now if only text is given without a clear vectorization strategy here.


        if query_vector is None:
            return []

        if not self.knowledge_base_vectors:
            return []

        query_vector_cpu = query_vector.cpu() # Move query vector to CPU for comparison with stored CPU tensors

        similarities = []
        for triple_key, kb_vector in self.knowledge_base_vectors.items():
            # kb_vector is already on CPU
            similarity = torch.cosine_similarity(query_vector_cpu.unsqueeze(0), kb_vector.unsqueeze(0)).item()
            similarities.append((triple_key, kb_vector, similarity))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        return similarities[:top_k]

    async def get_tn_query_vector(self, query_text: str) -> Optional[torch.Tensor]:
        """
        Converts a natural language query text into a 384-dim query vector
        by extracting triples and processing them with the TN module.
        Returns the vector of the first successfully processed triple, or None.
        """
        if 'tn' not in self.modules or 'tn' not in self.adapters:
            print("Error: TN module/adapter not available for query vectorization.")
            return None

        # 1. Extract triples from the query text using the same LLM method.
        #    Making _extract_triples_from_input awaitable or creating an async version.
        #    For now, assuming _extract_triples_from_input can be awaited if called from async.
        #    This requires _extract_triples_from_input to be an async def.
        #    Let's make a simplified assumption or an async helper if needed.
        #    The current _extract_triples_from_input uses asyncio.run, which is problematic if called from an async method.
        #    We'll need to call the underlying async 'call_deepseek_for_triples' directly.
        
        # Re-using the async part of _extract_triples_from_input directly:
        prompt = (
            "请从下面的文本中抽取所有知识三元组（主体, 关系, 客体）。"
            "每个三元组请严格按照JSON对象 `{\"subject\": \"主体\", \"relation\": \"关系\", \"object\": \"客体\"}` 的格式表示。"
            "将所有这些JSON对象组成一个JSON数组返回。如果找不到三元组，返回空数组 `[]`。"
            f"文本内容：\n{query_text}"
        )
        query_triples_tuples = []
        try:
            response = await self.deepseek_client.chat.completions.create(
                model=self.config.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "你是一个精确的知识抽取引擎。严格按照用户要求的JSON格式输出。"},
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
                for item in triple_list_json:
                    if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                        query_triples_tuples.append((str(item["subject"]), str(item["relation"]), str(item["object"])))
        except Exception as e:
            print(f"LLM triple extraction for query failed: {e}")
            return None

        if not query_triples_tuples:
            print(f"No triples extracted from query text: '{query_text}' for vectorization.")
            return None

        # 2. Process the first extracted triple (or a combination) through TN.
        #    This requires mapping string triples to IDs, similar to _process_tn.
        #    The TNAdapter's prepare_triples_for_compression expects list of string tuples.
        try:
            # Use only the first extracted triple for the query vector for simplicity
            first_triple_to_process = [query_triples_tuples[0]] 
            
            formatted_triples_ids = self.adapters['tn'].prepare_triples_for_compression(first_triple_to_process)
            if not formatted_triples_ids:
                print(f"Could not format query triple for TN: {first_triple_to_process[0]}")
                return None
                
            compressed_vector = self.adapters['tn'].compress_and_flatten(formatted_triples_ids) # Returns a tensor

            if compressed_vector.numel() > 0:
                return compressed_vector[0].detach().clone() # Return the first (and only) vector
            else:
                print(f"TN module produced an empty vector for query triple: {first_triple_to_process[0]}")
                return None
        except Exception as e:
            print(f"Error processing query triple with TN: {e}")
            return None

# Example of how PredictiveCodingAgent might be used (conceptual)
# def update_knowledge_base_with_predictive_coding(self, new_triples_with_vectors):
#     for triple_key, vector in new_triples_with_vectors.items():
#         # Assume self.modules['predictive_coding_module'] has a method to assess novelty
#         # This is highly conceptual and depends on PredictiveCodingAgent's design
#         is_novel_or_significant = self.modules['predictive_coding_module'].assess(vector) 
#         if is_novel_or_significant:
#             self.knowledge_base_vectors[triple_key] = vector.cpu() # Store on CPU
#             print(f"KB: Added/Updated vector for {triple_key} based on predictive coding.")

# Note: The interaction between PDA's pc_core and the main knowledge_base_vectors
# and the 'predictive_coding_module' needs to be clearly defined.
# For now, new vectors are added to knowledge_base_vectors if the triple_key is new.
# A more sophisticated predictive coding mechanism would determine when and how to store/update.
        

