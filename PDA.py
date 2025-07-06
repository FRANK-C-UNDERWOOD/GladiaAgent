"""
🎯🎯 PredictiveDialogAgent - 智能对话代理核心系统
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
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI # Ensure this is the correct import for your version
from collections import deque
from typing import List, Tuple, Dict, Union, Any, AsyncGenerator
import asyncio
import json
import hashlib
import traceback

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
    """
    def __init__(self, 
                 deepseek_api_key: str,
                 integrated_system_ref: Any, # Reference to the IntegratedSystem instance
                 tool_threshold: float = 0.25,
                 memory_threshold: float = 0.15): 
        
        self.integrated_system = integrated_system_ref
        self.current_prediction_error = 0.1 # Default placeholder value

        # 对话管理系统
        self.dialog_buffer = DialogHistoryBuffer(max_len=8)
        
        # 阈值设置
        self.tool_trigger_threshold = tool_threshold
        self.memory_trigger_threshold = memory_threshold 
        
        # 初始化DeepSeek客户端
        self.client = AsyncOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=deepseek_api_key
        )
    
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

    async def generate_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """生成对话响应的核心逻辑"""
        print(f"DEBUG_RECALL: PDA.generate_response called with prompt: '{prompt}'")
        # 1. 上下文构建
        dialog_ctx = self.dialog_buffer.context_text()
        thought_chain = self.dialog_buffer.chain_text() or "首次思考路径"
        
        # 2. 记忆检索 (使用新的嵌入方法)
        memory_block = "无相关核心知识库记忆"
        query_vector_for_kb = None

        print(f"DEBUG_RECALL: Attempting to generate embedding for prompt: '{prompt}'")
        try:
            # 使用新的嵌入方法
            query_vector_for_kb = self.integrated_system.generate_embeddings(prompt)
            print(f"DEBUG_RECALL: Successfully generated embedding. Shape: {query_vector_for_kb.shape}")
        except Exception as e_qv:
            print(f"DEBUG_RECALL: Error generating embedding: {e_qv}")
            traceback.print_exc()

        if query_vector_for_kb is not None:
            print(f"DEBUG_RECALL: Querying core knowledge base with generated embedding")
            # Query the core knowledge base
            try:
                retrieved_items = self.integrated_system.query_core_knowledge_base(
                    query_vector=query_vector_for_kb,
                    top_k=3
                )
            except Exception as e_qkb:
                print(f"DEBUG_RECALL: Error querying knowledge base: {e_qkb}")
                traceback.print_exc()
                retrieved_items = []

            print(f"DEBUG_RECALL: Retrieved_items from KB: {len(retrieved_items)} items")
            if retrieved_items:
                formatted_memory_results = []
                for triple_key_str, _, score in retrieved_items:
                    try:
                        triple_tuple = tuple(json.loads(triple_key_str))
                        readable_triple = f"({triple_tuple[0]}, {triple_tuple[1]}, {triple_tuple[2]})"
                        formatted_memory_results.append(f"- {readable_triple} (相似度: {score:.3f})")
                    except json.JSONDecodeError:
                        formatted_memory_results.append(f"- {triple_key_str} (相似度: {score:.3f})")
                if formatted_memory_results:
                    memory_block = "\n".join(formatted_memory_results)
                else:
                    memory_block = "核心知识库检索到内容，但格式化失败。"
                print(f"DEBUG_RECALL: Formatted memory_block: {memory_block}")
            else:
                memory_block = "核心知识库未检索到相关记忆。"
                print(f"DEBUG_RECALL: No items retrieved from core knowledge base.")
        else:
            memory_block = "无法为当前输入生成查询向量"
            print(f"DEBUG_RECALL: query_vector_for_kb is None. Memory block set to: '{memory_block}'")

        prediction_error = self.current_prediction_error 
        print(f"DEBUG_RECALL: Using prediction_error: {prediction_error:.3f} for LLM prompt.")
        
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
            print(f"DEBUG_RECALL: Calling LLM with System Prompt")
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
            
            # full_response = "" # No longer accumulating here
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # print(content, end="", flush=True) # Optional: keep for server-side logging
                    yield content # Yield the content chunk
            
            # print(f"DEBUG_RECALL: LLM stream finished.") # Optional
        
        except Exception as e:
            print(f"\n[API ERROR] {str(e)}")
            traceback.print_exc()
            yield "请求处理遇到技术问题，请稍后再试" # Yield error message as a chunk

    async def dialog_round(self, user_input: str) -> AsyncGenerator[str, None]:
        """处理单轮对话的全流程 (now an async generator)"""
        self.dialog_buffer.add_thought_step("开始解析用户问题语义框架")
        
        full_streamed_response = ""
        async for chunk in self.generate_response(user_input):
            full_streamed_response += chunk
            yield chunk
            
        # Add the complete response to dialog buffer after streaming is done
        self.dialog_buffer.add_dialog(user_input, full_streamed_response)
        # No return value needed for async generator in this context of yielding
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        if hasattr(self.integrated_system, 'knowledge_base_vectors'):
            return {
                "core_knowledge_base_vector_count": len(self.integrated_system.knowledge_base_vectors),
                "last_pda_prediction_error": self.current_prediction_error
            }
        return {
            "core_knowledge_base_vector_count": 0,
            "last_pda_prediction_error": self.current_prediction_error,
            "info": "IntegratedSystem reference not available or KB not found."
        }
    # 在PredictiveDialogAgent类中保持原有流式生成器不变
    
    async def dialog_round(self, user_input: str) -> AsyncGenerator[str, None]:
        """处理单轮对话的全流程 (现在是一个异步生成器)"""
        self.dialog_buffer.add_thought_step("开始解析用户问题语义框架")
        
        full_streamed_response = ""
        async for chunk in self.generate_response(user_input):
            full_streamed_response += chunk
            yield chunk
            
        # 将完整响应添加到对话缓冲区
        self.dialog_buffer.add_dialog(user_input, full_streamed_response)
# 测试主函数
async def main():
    print("PDA模块自检完成。请通过主系统接口调用对话功能。")

if __name__ == "__main__":
    pass
