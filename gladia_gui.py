from PyQt5.QtWidgets import (QApplication, QMainWindow, QSystemTrayIcon, QMenu, 
                             QMessageBox, QAction)
from PyQt5.QtCore import QUrl, QObject, pyqtSlot, pyqtSignal, QPoint, QTimer, Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
import os
import sys
import threading
import time
import asyncio
import json
import traceback

class AgentBridge(QObject):
    # 定义信号用于从Python发送消息到JavaScript
    addMessage = pyqtSignal(str, str)  # (角色, 消息)
    updateStatus = pyqtSignal(str)
    setCharacterImage = pyqtSignal(str)
    
    # 添加流式信号
    streamMessageChunk = pyqtSignal(str, str)  # (角色, 消息块)
    streamMessageFinished = pyqtSignal(str)    # (角色)
    
    def __init__(self, system, gui):
        super().__init__()
        self.system = system
        self.gui = gui

    @pyqtSlot(str, result=str)
    def processUserInput(self, text):
        """处理来自JS的用户输入，并返回回复文本"""
        # 在新线程中处理请求
        threading.Thread(target=self.process_input_thread, args=(text,)).start()
        return "处理中，请稍候..."

    def process_input_thread(self, text):
        """在新线程中处理用户输入"""
        # 为当前线程创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 显示用户消息
        self.updateStatus.emit("处理中...")
        
        try:
            # 处理用户输入
            start_time = time.time()
            
            # 获取流式响应生成器
            response_gen = self.system.chat_with_agent_stream(text)
            
            # 在当前线程的事件循环中处理流式响应
            loop.run_until_complete(self.handle_streaming_response(response_gen))
            
        except Exception as e:
            self.addMessage.emit("system", f"处理错误: {str(e)}")
            self.updateStatus.emit("错误")
            traceback.print_exc()
        finally:
            # 关闭事件循环
            loop.close()

    async def handle_streaming_response(self, response_gen):
        """处理流式响应"""
        try:
            # 收集流式响应
            response = await self.collect_streaming_response(response_gen)
            
            # 添加AI回复
            self.streamMessageFinished.emit("gladia")
            
            # 添加调试信息
            debug_info = [
                f"处理时间: {response.get('processing_time', 0):.2f}秒",
                f"PDA内存统计: {response.get('pda_memory_stats')}",
                f"PDA思维链: {response.get('pda_thought_chain')}",
                f"PDA预测误差: {response.get('pda_prediction_error', 0):.4f}",
                f"知识库向量数: {response.get('core_kb_vector_count', 0)}"
            ]
            
            for info in debug_info:
                self.addMessage.emit("system", info)
                
            self.updateStatus.emit("就緒")
            
        except Exception as e:
            self.addMessage.emit("system", f"流式处理错误: {str(e)}")
            self.updateStatus.emit("错误")
            traceback.print_exc()

    async def collect_streaming_response(self, response_gen):
        """收集流式响应"""
        full_response = ""
        start_time = time.time()
        
        # 处理每个流式块
        async for chunk in response_gen:
            # 发送消息块到前端
            self.streamMessageChunk.emit("gladia", chunk)
            full_response += chunk
            await asyncio.sleep(0.01)  # 稍微让步，避免阻塞事件循环
        
        processing_time = time.time() - start_time
        
        return {
            "reply": full_response,
            "processing_time": processing_time,
            "pda_memory_stats": self.system.pda_agent.get_memory_stats(),
            "core_kb_vector_count": len(self.system.knowledge_base_vectors),
            "pda_thought_chain": self.system.pda_agent.dialog_buffer.chain_text(),
            "pda_prediction_error": self.system.pda_agent.current_prediction_error
        }
    
    @pyqtSlot()
    def saveKnowledgeBase(self):
        """保存知识库"""
        try:
            self.updateStatus.emit("保存知识库...")
            self.system.save_all_memory()
            self.addMessage.emit("system", "核心知识库已保存")
            self.updateStatus.emit("就緒")
        except Exception as e:
            self.addMessage.emit("system", f"保存失败: {str(e)}")
            self.updateStatus.emit("错误")
    
    @pyqtSlot()
    def exitApplication(self):
        """退出应用"""
        self.saveKnowledgeBase()
        self.addMessage.emit("system", "感谢使用，再见！")
        QTimer.singleShot(1000, QApplication.instance().quit)

def create_gladia_icon():
    """创建Gladia图标"""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    
    # 绘制简单的图标
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(100, 180, 255))
    painter.setPen(QColor(150, 200, 255))
    
    # 绘制圆形和三角形组合
    painter.drawEllipse(12, 12, 40, 40)
    
    # 绘制三角形
    points = [
        QPoint(32, 18),
        QPoint(48, 48),
        QPoint(16, 48)
    ]
    painter.drawPolygon(points)
    
    painter.end()
    return QIcon(pixmap)

def run_gui(config, system):
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = QMainWindow()
    window.setWindowTitle("GladiaAgent - 阿戈尔深海界面")
    # 移除固定大小设置，允许调整窗口大小
    # window.setFixedSize(1000, 750)  # 原始固定大小设置
    window.resize(1000, 750)  # 设置初始大小但允许调整
    
    # 创建Web视图
    view = QWebEngineView()
    window.setCentralWidget(view)
    
    # 设置JS-Python通信桥
    channel = QWebChannel()
    bridge = AgentBridge(system, window)
    channel.registerObject("AgentBridge", bridge)
    view.page().setWebChannel(channel)
    
    # 加载HTML界面
    html_path = os.path.abspath("ui/aegor_ui.html")
    if os.path.exists(html_path):
        view.load(QUrl.fromLocalFile(html_path))
    else:
        # 如果文件不存在，显示错误
        QMessageBox.critical(window, "错误", f"UI文件未找到: {html_path}")
        sys.exit(1)
    
    # 创建系统托盘
    tray_icon = QSystemTrayIcon(window)
    tray_icon.setIcon(create_gladia_icon())
    
    # 创建托盘菜单
    tray_menu = QMenu()
    
    # 添加菜单项 - 新增全屏切换选项
    actions = [
        ("显示主窗口", window.show),
        ("隐藏主窗口", window.hide),
        ("切换全屏", lambda: window.showFullScreen() if not window.isFullScreen() else window.showNormal()),
        ("保存知识库", bridge.saveKnowledgeBase),
        ("退出系统", bridge.exitApplication)
    ]
    
    for text, callback in actions:
        action = tray_menu.addAction(text)
        action.triggered.connect(callback)
    
    tray_icon.setContextMenu(tray_menu)
    tray_icon.show()
    
    # 添加全屏切换快捷键 (F11)
    fullscreen_action = QAction(window)
    fullscreen_action.setShortcut("F11")
    fullscreen_action.triggered.connect(
        lambda: window.showFullScreen() if not window.isFullScreen() else window.showNormal()
    )
    window.addAction(fullscreen_action)
    
    # 窗口关闭事件处理
    window.closeEvent = lambda event: bridge.exitApplication()
    
    window.show()
    app.exec_()

# 如果直接运行此文件，用于测试
if __name__ == "__main__":
    from config import Config
    from integrated_system import IntegratedSystem
    
    # 创建测试配置和系统
    config = Config()
    system = IntegratedSystem(config)
    
    # 运行GUI
    run_gui(config, system)ig)
    
    # 运行GUI
    run_gui(config, system)
