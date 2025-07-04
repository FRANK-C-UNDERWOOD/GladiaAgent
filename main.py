# main.py
from integrated_system import IntegratedSystem
from config import Config
import asyncio
import os
import sys
import argparse
from gladia_gui import run_gui  # 导入GUI启动函数

def main():
    # 创建配置
    config = Config()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GladiaAgent 智能系统')
    parser.add_argument('--gui', action='store_true', help='启用图形用户界面')
    parser.add_argument('--console', action='store_true', help='使用命令行界面')
    parser.add_argument('--api-key', type=str, help='直接设置DeepSeek API密钥')
    args = parser.parse_args()
    
    # 设置API密钥 (优先级: 命令行参数 > 环境变量 > 用户输入 > 默认配置)
    api_key = None
    
    # 1. 检查命令行参数
    if args.api_key:
        api_key = args.api_key
        print("API密钥已从命令行参数加载。")
    
    # 2. 检查环境变量
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            print("DeepSeek API密钥已从环境变量加载。")
    
    # 3. 如果仍未设置，检查用户输入
    if not api_key:
        print("警告: DeepSeek API密钥未设置。")
        user_api_key = input("请输入DeepSeek API密钥 (或按Enter跳过): ").strip()
        if user_api_key:
            api_key = user_api_key
            print("API密钥已设置。")
    
    # 4. 如果设置了API密钥，更新配置
    if api_key:
        config.llm_config.api_key = api_key
    elif not config.llm_config.api_key:
        print("继续执行，但LLM功能可能受限。")

    # 创建集成系统
    print("正在初始化集成系统...")
    system = IntegratedSystem(config)
    print("集成系统初始化完成。")
    
    # 打印系统信息
    print("--- 系统信息 ---")
    system_info = system.get_system_info()
    print(f"  模块: {system_info.get('modules')}")
    print(f"  处理流程步骤: {system_info.get('pipeline_steps')}")
    print(f"  核心知识库向量数量: {system_info.get('knowledge_base_vector_count')}")
    print("--- 系统信息结束 ---")
    
    # 根据参数选择运行模式
    if args.gui:
        print("启动图形用户界面...")
        run_gui(config, system)
    elif args.console:
        run_console(system)
    else:
        # 默认情况下，如果有显示设备则使用GUI，否则使用控制台
        if has_display():
            print("启动图形用户界面...")
            run_gui(config, system)
        else:
            print("未检测到显示设备，使用命令行模式...")
            run_console(system)

def run_console(system):
    """命令行模式运行"""
    print("\nGladiaAgent启动成功")
    print("系统已尝试加载核心知识库。")
    print("输入 'exit' 或 'quit' 来结束对话。")

    try:
        while True:
            user_input = input("\nDoctor: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("\nGladia: 正在保存核心知识库状态...")
                system.save_all_memory()
                print("Gladia: 核心知识库已保存。感谢您的使用，再见！")
                break
            
            if not user_input:
                continue

            print("\nGladia: ", end="", flush=True)
            chat_response = system.chat_with_agent(user_input)
            
            # 打印调试信息
            print(f"\n[调试信息] PDA内存统计: {chat_response.get('pda_memory_stats')}")
            print(f"[调试信息] PDA思维链: {chat_response.get('pda_thought_chain')}")
            print(f"[调试信息] PDA当前预测误差: {chat_response.get('pda_prediction_error', 0):.4f}")
            print(f"[调试信息] 核心知识库数量: {chat_response.get('core_kb_vector_count', 0)}")

    except KeyboardInterrupt:
        print("\n\n🤖 AI: 检测到中断操作。正在紧急保存核心知识库...")
        system.save_all_memory()
        print("🤖 AI: 核心知识库已保存。对话已结束。")
    except Exception as e:
        print(f"\n🤖 AI: 主循环发生了一个意外错误: {e}")
        import traceback
        traceback.print_exc()
        print("🤖 AI: 正在尝试保存核心知识库...")
        system.save_all_memory()
        print("🤖 AI: 核心知识库已保存。程序即将退出。")

def has_display():
    """检测是否有显示设备"""
    # Windows系统
    if sys.platform == 'win32':
        return True
    
    # macOS系统
    if sys.platform == 'darwin':
        return True
    
    # Linux/Unix系统 - 检查DISPLAY环境变量
    if 'DISPLAY' in os.environ:
        return True
    
    # 其他情况默认为无显示设备
    return False

if __name__ == "__main__":
    main()
