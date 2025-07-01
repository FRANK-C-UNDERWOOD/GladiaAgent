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
        print("AI对话功能可能无法正常工作，除非API密钥在config.py中已硬编码或通过其他方式设置。")
        # Example: Prompt user if you want to allow direct input, but env var is safer.
        user_api_key = input("请输入DeepSeek API密钥 (如果需要，或按Enter跳过): ").strip()
        if user_api_key:
            config.llm_config.api_key = user_api_key
            print("API密钥已设置。")
        elif not config.llm_config.api_key: # Check if it was pre-set in LLMConfig
            print("继续执行，但LLM功能可能受限。")


    # 创建集成系统
    # IntegratedSystem 会自动初始化 PredictiveDialogAgent 和核心知识库
    print("正在初始化集成系统...")
    system = IntegratedSystem(config)
    print("集成系统初始化完成。")
    
    # (可选) 打印系统信息，可以取消注释来查看
    print("--- 系统信息 ---")
    
    system_info = system.get_system_info()
    print(f"  模块: {system_info.get('modules')}")
    print(f"  处理流程步骤: {system_info.get('pipeline_steps')}")
    print(f"  核心知识库向量数量: {system_info.get('knowledge_base_vector_count')}")
    print("--- 系统信息结束 ---")
    
    # (可选) 演示核心处理功能，可以取消注释以测试
    # print("--- 演示系统核心处理能力 ---")
    # demo_triples = [
    #     ("地球", "是", "行星"),
    #     ("太阳", "是", "恒星")
    # ]
    # demo_text_input = "木星是太阳系中最大的行星。"
    # print(f"处理演示文本: '{demo_text_input}'")
    # # Processing will also attempt to update the core knowledge base
    # result = system.process(demo_text_input, triples=demo_triples) 
    # print(f"演示处理结果:")
    # print(f"- 输入文本: {result.original_input if result.original_input else 'N/A'}")
    # print(f"- 初始三元组数量: {len(result.triples)}")
    # if result.triples:
    #     print(f"- 第一个三元组示例: {result.triples[0]}")
    # if result.compressed_vectors.size > 0:
    #      # Assuming compressed_vectors is numpy array as per ProcessingResult
    #     print(f"- 压缩向量形状: {result.compressed_vectors.shape}")
    # if result.spatial_embeddings.numel() > 0:
    #     print(f"- 空间嵌入形状: {result.spatial_embeddings.shape}")
    # print(f"- 预测结果是否存在: {result.predictions is not None}")
    # print(f"- 执行的处理步骤: {result.metadata['processing_steps']}")
    # print(f"当前核心知识库向量数量: {system.knowledge_base_vectors.__len__()}")
    # print("--- 核心处理能力演示结束 ---")

    print("\nGladiaAgent启动成功")
    print("系统已尝试加载核心知识库。")
    print("输入 'exit' 或 'quit' 来结束对话。")

    try:
        while True:
            user_input = input("\nDoctor: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("\nGladia: 正在保存核心知识库状态...")
                system.save_all_memory() # This now saves only the core TN/seRNN knowledge
                print("Gladia: 核心知识库已保存。感谢您的使用，再见！")
                break
            
            if not user_input:
                continue

            print("\nGladia: ", end="", flush=True) # PDA's response will be streamed
            chat_response = system.chat_with_agent(user_input)
            
            # chat_response dictionary contains:
            # "reply": AI's textual response (already streamed by PDA)
            # "pda_memory_stats": Dict from PDA.get_memory_stats()
            # "core_kb_vector_count": Count from IntegratedSystem
            # "pda_thought_chain": String of PDA's thought chain
            # "pda_prediction_error": Float, PDA's current_prediction_error
            
            # (可选) 打印额外的调试信息
            print(f"\n[调试信息] PDA内存统计: {chat_response.get('pda_memory_stats')}")
            print(f"[调试信息] PDA思维链: {chat_response.get('pda_thought_chain')}")
            print(f"[调试信息] PDA当前预测误差: {chat_response.get('pda_prediction_error'):.4f}")
            print(f"[调试信息] 核心知识库数量 (从chat_response): {chat_response.get('core_kb_vector_count')}")
            print(f"[调试信息] 核心知识库数量 (直接访问): {system.knowledge_base_vectors.__len__()}")


    except KeyboardInterrupt:
        print("\n\nGladia: 检测到中断操作。正在紧急保存核心知识库...")
        system.save_all_memory()
        print("Gladia: 核心知识库已保存。对话已结束。")
    except Exception as e:
        print(f"\nGladia: 主循环发生了一个意外错误: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        print("Gladia: 正在尝试保存核心知识库...")
        system.save_all_memory()
        print("Gladia: 核心知识库已保存。程序即将退出。")

if __name__ == "__main__":
    main()
