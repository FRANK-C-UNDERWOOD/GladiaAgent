# main.py
from integrated_system import IntegratedSystem
from config import Config
import asyncio

def main():
    # 创建配置
    config = Config()
    
    # 创建集成系统
    system = IntegratedSystem(config)
    
    # 打印系统信息
    print("System Info:", system.get_system_info())
    
    # 单个处理示例
    test_triples = [
        ("Alice", "loves", "Bob"),
        ("Bob", "works_at", "Company"),
        ("Company", "located_in", "City")
    ]
    
    result = system.process("Test input", triples=test_triples)
    
    print(f"Processing Result:")
    print(f"- Triples: {len(result.triples)}")
    print(f"- Compressed vectors shape: {result.compressed_vectors.shape}")
    print(f"- Spatial embeddings shape: {result.spatial_embeddings.shape}")
    print(f"- Predictions: {result.predictions is not None}")
    print(f"- Processing steps: {result.metadata['processing_steps']}")
    
    # 批处理示例
    batch_input = [
        [("X", "relates_to", "Y")],
        [("A", "connects", "B"), ("B", "links", "C")],
        [("User", "interacts", "System")]
    ]
    
    batch_results = system.batch_process(batch_input)
    print(f"\nBatch processing completed: {len(batch_results)} results")

def chat(system, user_input: str, history: list = None) -> str:
    """
    与用户进行对话，history为历史对话（可选），返回AI回复
    """
    if history is None:
        history = []
    # 构建对话上下文
    messages = [
        {"role": "system", "content": "你是一个专业的科研助手AI，擅长学术、科研、数据分析和技术解答。请用专业、简明的语言回答用户问题。"}
    ]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": user_input})

    # 调用deepseek LLM生成回复
    async def call_deepseek():
        try:
            response = await system.deepseek_client.chat.completions.create(
                model=system.config.llm_config.model_name,
                messages=messages,
                temperature=system.config.llm_config.temperature,
                max_tokens=system.config.llm_config.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM对话生成失败: {e}")
            return "很抱歉，AI暂时无法回答您的问题。"
    return asyncio.run(call_deepseek())

if __name__ == "__main__":
    main()