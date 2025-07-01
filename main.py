# main.py
from integrated_system import IntegratedSystem
from config import Config
import asyncio
import os 

def main():
    # 创建配置
    config = Config()

    # !!! 重要: 设置DeepSeek API密钥 !!!
    # 优先从环境变量 DEEPSEEK_API_KEY 获取
    # 如果环境变量未设置，则会提示用户，此时LLM调用可能失败
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        config.llm_config.api_key = api_key
        print("DeepSeek API密钥已从环境变量加载。")
    else:
        print("警告: DEEPSEEK_API_KEY 环境变量未设置。")
        print("AI对话功能可能无法正常工作，除非API密钥在config.py中已硬编码或以其他方式设置。")
        # 你可以在这里添加逻辑来从用户输入读取API密钥，例如:
        user_api_key = input("请输入DeepSeek API密钥 (如果需要): ")
        if user_api_key:
           config.llm_config.api_key = user_api_key
    
    # 创建集成系统
    # IntegratedSystem 会自动初始化 PredictiveDialogAgent
    system = IntegratedSystem(config)
    
    #(可选) 打印系统信息，可以取消注释来查看
    print("System Info:", system.get_system_info())
    
    # (可选) 单个处理示例和批处理示例，可以取消注释来测试系统的核心处理能力
    print("\n--- 演示系统核心处理能力 ---")
    test_triples = [
        ("Alice", "loves", "Bob"),
        ("Bob", "works_at", "Company"),
        ("Company", "located_in", "City")
    ]
    result = system.process("这是一个用于测试核心处理流程的输入文本。", triples=test_triples)
    print(f"单个处理结果:")
    print(f"- 输入文本: {result.original_input}")
    print(f"- 提取的三元组数量: {len(result.triples)}")
    if result.triples:
        print(f"- 第一个三元组示例: {result.triples[0]}")
    if result.compressed_vectors.size > 0:
        print(f"- 压缩向量形状: {result.compressed_vectors.shape}")
    if result.spatial_embeddings.numel() > 0:
        print(f"- 空间嵌入形状: {result.spatial_embeddings.shape}")
    print(f"- 预测结果是否存在: {result.predictions is not None}")
    print(f"- 执行的处理步骤: {result.metadata['processing_steps']}")
    print("--- 核心处理能力演示结束 ---\n")

    print("\nGladiaAgent启动成功!")
    print("系统已尝试加载历史记忆。")
    print("输入 'exit' 或 'quit' 来结束对话。")

    try:
        while True:
            user_input = input("\nDoctor: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("\nGladia: 正在保存记忆模块状态...")
                system.save_all_memory() # 调用IntegratedSystem中的方法保存记忆
                print("Gladia: 记忆已保存。感谢您的使用，再见！")
                break
            
            if not user_input: # 如果用户只输入了空格或什么都没输入
                continue

            # 使用 system.chat_with_agent 进行对话
            # PredictiveDialogAgent 内部会处理流式输出，所以这里不需要 print(end="", flush=True)
            # AI的回复会在 chat_with_agent 方法中直接打印到控制台
            print("\nGladia: ", end="", flush=True) # 提示AI正在响应
            chat_result = system.chat_with_agent(user_input)
                        
            #打印额外的调试信息，例如思维链或记忆统计
            print(f"\n[调试信息] 思维链: {chat_result['thought_chain']}")
            print(f"\n[调试信息] 当前预测误差: {chat_result['prediction_error']:.4f}")
            print(f"\n[调试信息] 记忆统计: {chat_result['memory_stats']}")

    except KeyboardInterrupt:
        print("\n\nGladia: 检测到中断操作。正在紧急保存记忆...")
        system.save_all_memory()
        print("Gladia: 记忆已保存。对话已结束。")
    except Exception as e:
        print(f"\nGladia: 发生了一个意外错误: {e}")
        print("Gladia: 正在尝试保存记忆...")
        system.save_all_memory()
        print("Gladia: 记忆已保存。程序即将退出。")

if __name__ == "__main__":
    main()
