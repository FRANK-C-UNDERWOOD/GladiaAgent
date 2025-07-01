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

# --------------- 基础记忆组件 (伪代码示意，实际需完整实现) ---------------
class MemoryAnchorUpdater:
    def __init__(self):
        self.shared_anchors = {}
    
    def update_anchor(self, key: str, weight: float):
        self.shared_anchors[key] = weight
    
    def get_shared_anchors(self) -> Dict[str, float]:
        return self.shared_anchors.copy()

class GraphMemoryNode:
    __slots__ = ('id', 'embedding', 'content', 'last_accessed')
    def __init__(self, id: str, emb: torch.Tensor, content: str):
        self.id = id
        self.embedding = emb
        self.content = content
        self.last_accessed = torch.tensor(0.0)

class GraphMemoryBank:
    def __init__(self):
        self.graph_nodes: Dict[str, GraphMemoryNode] = {}
        self.graph_edges: Dict[str, List[Tuple[str, str]]] = {}
    
    def add_node(self, node: GraphMemoryNode):
        self.graph_nodes[node.id] = node
    
    def add_edge(self, from_id: str, to_id: str, rel: str):
        if from_id not in self.graph_edges:
            self.graph_edges[from_id] = []
        self.graph_edges[from_id].append((to_id, rel))
    
    def save_all(self, path_prefix: str):
        """保存记忆图到JSON文件。"""
        nodes_file = f"{path_prefix}_nodes.json"
        edges_file = f"{path_prefix}_edges.json"

        serializable_nodes = []
        for node_id, node in self.graph_nodes.items():
            serializable_nodes.append({
                'id': node.id,
                'embedding': node.embedding.cpu().tolist(), # Ensure tensor is on CPU and convert to list
                'content': node.content,
                'last_accessed': node.last_accessed.cpu().item() # Ensure tensor is on CPU and get scalar value
            })
        
        try:
            with open(nodes_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_nodes, f, indent=4, ensure_ascii=False)
            print(f"🧠 Nodes saved to {nodes_file}")
        except Exception as e:
            print(f"Error saving nodes: {e}")

        try:
            with open(edges_file, 'w', encoding='utf-8') as f:
                json.dump(self.graph_edges, f, indent=4, ensure_ascii=False)
            print(f"🔗 Edges saved to {edges_file}")
        except Exception as e:
            print(f"Error saving edges: {e}")
    
    def load(self, path_prefix: str):
        """从JSON文件加载记忆图。"""
        nodes_file = f"{path_prefix}_nodes.json"
        edges_file = f"{path_prefix}_edges.json"

        if os.path.exists(nodes_file):
            try:
                with open(nodes_file, 'r', encoding='utf-8') as f:
                    loaded_nodes_data = json.load(f)
                
                self.graph_nodes.clear()
                for node_data in loaded_nodes_data:
                    node = GraphMemoryNode(
                        id=node_data['id'],
                        emb=torch.tensor(node_data['embedding'], dtype=torch.float32), # Specify dtype
                        content=node_data['content']
                    )
                    node.last_accessed = torch.tensor(node_data['last_accessed'], dtype=torch.float32) # Specify dtype
                    self.graph_nodes[node.id] = node
                print(f"🧠 Nodes loaded from {nodes_file}: {len(self.graph_nodes)} nodes")
            except Exception as e:
                print(f"Error loading nodes: {e}")
        else:
            print(f"Node file {nodes_file} not found. Starting with an empty node bank.")

        if os.path.exists(edges_file):
            try:
                with open(edges_file, 'r', encoding='utf-8') as f:
                    self.graph_edges = json.load(f)
                print(f"🔗 Edges loaded from {edges_file}: {len(self.graph_edges)} edge groups")
            except Exception as e:
                print(f"Error loading edges: {e}")
                self.graph_edges = {} # Reset if loading failed
        else:
            print(f"Edge file {edges_file} not found. Starting with an empty edge bank.")

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

class PredictiveCodingCore:
    """预测编码核心处理器"""
    def __init__(self, memory: GraphMemoryBank):
        self.memory = memory
        self.current_prediction_error = 0.0
    
    def predict(self, triplet: tuple) -> float:
        """预测三元组的认知匹配度"""
        # 伪代码：基于记忆图计算预测误差
        self.current_prediction_error = 0.1  # 模拟计算值
        return self.current_prediction_error
    
    def encode_input(self, text: str) -> torch.Tensor:
        """编码输入信息（简化实现）"""
        return torch.randn(1, 64)
    
    def update_memory(self, triplet: tuple, embedding: torch.Tensor):
        """更新记忆系统"""
        # 生成唯一记忆ID
        mem_id = hashlib.md5(str(triplet).encode()).hexdigest()
        
        # 创建记忆节点
        node = GraphMemoryNode(
            id=mem_id,
            emb=embedding,
            content=f"{triplet[0]} {triplet[1]} {triplet[2]}"
        )
        self.memory.add_node(node)

class MemoryRetriever:
    """记忆检索系统"""
    def __init__(self, memory_bank: GraphMemoryBank):
        self.memory = memory_bank
    
    def query_memory(self, query_text: str, top_k=3) -> List[str]:
        """查询相关记忆（简化实现）"""
        # 实际应实现基于向量的相似度搜索
        return [node.content for node in list(self.memory.graph_nodes.values())[:top_k]]

class PredictiveDialogAgent:
    """预测编码对话代理主类"""
    def __init__(self, 
                 deepseek_api_key: str, 
                 tool_threshold: float = 0.25,
                 memory_threshold: float = 0.15):
        # 核心组件初始化
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.client = AsyncOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=deepseek_api_key
        )
        
        # 记忆系统初始化
        self.memory_bank = GraphMemoryBank()
        self.DSAP = MemoryAnchorUpdater()
        self.MPR = MemoryRetriever(self.memory_bank)
        self.pc_core = PredictiveCodingCore(self.memory_bank)
        
        # 对话管理系统
        self.dialog_buffer = DialogHistoryBuffer(max_len=8)
        
        # 阈值设置
        self.tool_trigger_threshold = tool_threshold
        self.memory_trigger_threshold = memory_threshold
        
        # 记忆锚点（示例个性化设置）
        self._init_identity_anchors()
    
    def _init_identity_anchors(self):
        """初始化身份锚点（领域知识/个性特征）"""
        self.DSAP.update_anchor("身份::实验室助手::歌蕾蒂娅", 1.0)
        self.DSAP.update_anchor("语言风格::学术严谨性::保持专业术语", 0.9)
        self.DSAP.update_anchor("对话偏好::详细解释::提供额外背景知识", 0.85)
    
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
        
        # 2. 记忆检索与预测编码
        memory_results = self.MPR.query_memory(prompt)
        memory_block = "\n".join([f"- {mem}" for mem in memory_results]) or "无相关记忆"
        prediction_error = self.pc_core.predict(("用户输入", "当前语义", prompt))
        
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
        response = await self.generate_response(user_input)
        
        # Step 2: 预测编码记忆决策
        if self.pc_core.current_prediction_error > self.memory_trigger_threshold:
            self.dialog_buffer.add_thought_step("检测到高认知误差，启动记忆更新协议")
            triplet = await self.extract_triplet(user_input)
            
            if triplet:
                # 创建记忆节点
                emb = self.embedder.encode(user_input)
                self.pc_core.update_memory(triplet, emb)
                
                # 添加到思维链
                self.dialog_buffer.add_thought_step(
                    f"新增知识节点: {triplet[0]}→{triplet[1]}→{triplet[2]}"
                )
        
        # Step 3: 更新对话历史
        self.dialog_buffer.add_dialog(user_input, response)
        return response
    
    def save_memory(self, path: str = "agent_memory.gmb"):
        """持久化记忆系统"""
        self.memory_bank.save_all(path)
        print(f"💾 记忆系统已保存到 {path}")
    
    def load_memory(self, path: str = "agent_memory.gmb"):
        """加载记忆系统"""
        if os.path.exists(path):
            self.memory_bank.load(path)
            print(f"🔍 已加载{len(self.memory_bank.graph_nodes)}个记忆节点")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        return {
            "total_nodes": len(self.memory_bank.graph_nodes),
            "total_edges": sum(len(v) for v in self.memory_bank.graph_edges.values()),
            "last_prediction_error": self.pc_core.current_prediction_error
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
