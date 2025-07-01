"""
🎯 PredictiveDialogAgent - 智能对话代理核心系统
技术架构：预测编码(Predictive Coding) + 深度记忆管理 + DeepSeek大模型集成
功能亮点：
1. 自适应记忆触发机制
2. 思维链(CoT)可视化管理
3. 动态三元组抽取
4. 记忆持久化存储
5. 实时预测误差监控
"""
import os
import torch

# 设置镜像源（支持HTTP协议避免SSL问题）
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"  # 国内镜像源[5](@ref)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

from openai import AsyncOpenAI
from collections import deque
from typing import List, Tuple, Dict, Union, Any
import asyncio
import json
import hashlib

# --------------- 核心对话代理实现 ---------------
class DialogHistoryBuffer:
    """对话历史与思维链管理"""
    def __init__(self, max_len=10):
        self.history = deque(maxlen=max_len)
        self.thought_chain = []

    def add_dialog(self, user_input: str, agent_response: str):
        self.history.append(("user", user_input))
        self.history.append(("assistant", agent_response))

    def add_thought_step(self, thought: str):
        self.thought_chain.append(thought)
        if len(self.thought_chain) > 5:
            self.thought_chain.pop(0)

    def context_text(self) -> str:
        """格式化对话历史"""
        return "\n".join([f"{'User' if role=='user' else 'AI'}: {text}" 
                         for role, text in self.history])

    def chain_text(self) -> str:
        """格式化思维链"""
        return "\n".join([f"Step {i+1}: {thought}" 
                         for i, thought in enumerate(self.thought_chain)])

    def clear_chain(self):
        self.thought_chain.clear()

class PredictiveDialogAgent:
    """
    预测编码对话代理主类
    Refactored to use a unified memory system managed by IntegratedSystem.
    """
    def __init__(self, 
                 deepseek_api_key: str,
                 integrated_system_ref: Any, # Reference to the IntegratedSystem instance
                 tool_threshold: float = 0.25,
                 memory_threshold: float = 0.15): # memory_threshold might be re-evaluated
        
        self.integrated_system = integrated_system_ref # Store reference to IntegratedSystem

        # 核心组件初始化
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2') # May still be used for some local text processing
        self.client = AsyncOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=deepseek_api_key
        )
        
       
        self.current_prediction_error = 0.1 # Default placeholder value

        # 对话管理系统
        self.dialog_buffer = DialogHistoryBuffer(max_len=8)
        
        # 阈值设置
        self.tool_trigger_threshold = tool_threshold
        # memory_threshold will need to be re-evaluated in context of new predictive coding logic
        self.memory_trigger_threshold = memory_threshold 
        
       
    
    async def extract_triplet(self, text: str) -> Union[Tuple[str, str, str], None]:
        """使用大模型抽取知识三元组"""
        PROMPT = (
            "请从下面的对话内容中抽取核心知识三元组(主体, 关系, 客体)。"
            "格式要求：严格按[主体, 关系, 客体]返回，不要解释。"
            f"对话内容：{text}"
        )
        
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是有三重蒸馏能力的知识抽取专家"},
                    {"role": "user", "content": PROMPT}
                ]
            )
            result = response.choices[0].message.content.strip()
            
            # 解析三元组格式
            if result.startswith("[") and result.endswith("]"):
                items = result[1:-1].split(",")
                if len(items) == 3:
                    return tuple(i.strip() for i in items)
        except Exception as e:
            print(f"[ERROR] Triplet extraction failed: {str(e)}")
        return None

    async def generate_response(self, prompt: str) -> str:
        """生成对话响应的核心逻辑"""
        # 1. 上下文构建
        dialog_ctx = self.dialog_buffer.context_text()
        thought_chain = self.dialog_buffer.chain_text() or "首次思考路径"
        
        # 2. 记忆检索 (using Core Unified Memory via IntegratedSystem)
        memory_block = "无相关核心知识库记忆" # Default if no relevant info found
        query_vector_for_kb = await self.integrated_system.get_tn_query_vector(prompt)

        if query_vector_for_kb is not None:
            # Query the core knowledge base
            # Ensure query_vector_for_kb is on the correct device if IntegratedSystem expects it
            # query_core_knowledge_base expects query_vector to be on CPU for its internal logic
            retrieved_items = self.integrated_system.query_core_knowledge_base(
                query_vector=query_vector_for_kb.cpu(), 
                top_k=3
            )
            if retrieved_items:
                formatted_memory_results = []
                for triple_key_str, _, score in retrieved_items:
                    # triple_key_str is a JSON string of a sorted tuple, e.g., '["客体", "主体", "关系"]'
                    # We can parse it back or use it as is.
                    try:
                        # Attempt to parse the triple_key back into a readable format
                        triple_tuple = tuple(json.loads(triple_key_str))
                        # Reconstruct a more readable string if possible, or use key as is
                        readable_triple = f"({triple_tuple[0]}, {triple_tuple[1]}, {triple_tuple[2]})"
                        formatted_memory_results.append(f"- {readable_triple} (相似度: {score:.3f})")
                    except json.JSONDecodeError:
                        formatted_memory_results.append(f"- {triple_key_str} (相似度: {score:.3f})") # Fallback to raw key

                memory_block = "\n".join(formatted_memory_results)
        else:
            memory_block = "无法为当前输入生成查询向量"

        # prediction_error is now a member `self.current_prediction_error`
        # It should be updated by IntegratedSystem or a dedicated predictive coding module
        # For now, it uses the placeholder value.
        prediction_error = self.current_prediction_error 
        
        # 3. 构建系统提示（多重认知来源整合）
        SYSTEM_PROMPT = f"""
# 角色设定
你是歌蕾蒂娅,实验室的科研助手，拥有跨学科知识和实验数据分析能力。

# 对话历史
{dialog_ctx}

# 思维链记录
{thought_chain}

# 关联记忆
{memory_block}

# 认知状态
预测误差: {prediction_error:.3f} | {'需要深度思考' if prediction_error > 0.2 else '常规应答模式'}

# 响应要求
1. 严格保持专业严谨的学术用语习惯
2. 对复杂概念提供简明背景解释
3. 核心回答控制在200字内
4. 结尾标注[思考完成]
"""
        # 4. 调用DeepSeek API生成响应
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.35,
                max_tokens=512,
                stream=True
            )
            
            # 流式接收处理
            full_response = ""
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            return full_response
        
        except Exception as e:
            print(f"\n[API ERROR] {str(e)}")
            return "请求处理遇到技术问题，请稍后再试"

    async def dialog_round(self, user_input: str) -> str:
        """处理单轮对话的全流程"""
        # Step 1: 基础响应生成
        self.dialog_buffer.add_thought_step("开始解析用户问题语义框架")
        # Note: The process of extracting triples from user_input and updating the 
        # core knowledge base (self.integrated_system.knowledge_base_vectors)
        # is now handled by IntegratedSystem.chat_with_agent -> IntegratedSystem.process()
        # *before* this dialog_round method is called by chat_with_agent.

        response = await self.generate_response(user_input) # generate_response will be updated later to use new KB
        
        
        # Step 3: 更新对话历史
        self.dialog_buffer.add_dialog(user_input, response)
        return response
    
    # def save_memory(self, path: str = "agent_memory.gmb"): ... (Removed - unified memory saved by IntegratedSystem)
    
    # def load_memory(self, path: str = "agent_memory.gmb"): ... (Removed - unified memory loaded by IntegratedSystem)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计信息 (now reflects core KB via IntegratedSystem)"""
        if hasattr(self.integrated_system, 'knowledge_base_vectors'):
            # TODO: Add more stats from integrated_system if needed (e.g. TN/SeRNN model info)
            return {
                "core_knowledge_base_vector_count": len(self.integrated_system.knowledge_base_vectors),
                "last_pda_prediction_error": self.current_prediction_error # PDA's own error metric
            }
        return {
            "core_knowledge_base_vector_count": 0,
            "last_pda_prediction_error": self.current_prediction_error,
            "info": "IntegratedSystem reference not available or KB not found."
        }

# ==================== 启动入口 ====================
async def main():
    # 初始化代理 (需设置真实API密钥)
    agent = PredictiveDialogAgent(deepseek_api_key="")
    print("PDA 预测对话系统启动成功")
    # 加载历史记忆
    agent.load_memory()
    print("历史记忆加载成功")
    # 对话演示
    while True:
        try:
            user_input = input("\n👤 您: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            print("\n🤖 AI: ", end="")
            response = await agent.dialog_round(user_input)
            
        except KeyboardInterrupt:
            print("\n对话结束")
            break
    
    # 保存记忆
    agent.save_memory()

if __name__ == "__main__":
    asyncio.run(main())
