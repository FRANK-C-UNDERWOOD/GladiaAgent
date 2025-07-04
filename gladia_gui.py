# gladia_gui.py
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

class AgentBridge(QObject):
    # 定义信号用于从Python发送消息到JavaScript
    addMessage = pyqtSignal(str, str)  # (角色, 消息)
    updateStatus = pyqtSignal(str)
    setCharacterImage = pyqtSignal(str)
    
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
        # 显示用户消息
        self.addMessage.emit("user", text)
        self.updateStatus.emit("处理中...")
        
        try:
            # 处理用户输入
            start_time = time.time()
            response = self.system.chat_with_agent(text)
            processing_time = time.time() - start_time
            
            # 添加AI回复
            self.addMessage.emit("gladia", response.get("reply", "未收到回复"))
            
            # 添加调试信息
            debug_info = [
                f"处理时间: {processing_time:.2f}秒",
                f"PDA内存统计: {response.get('pda_memory_stats')}",
                f"PDA思维链: {response.get('pda_thought_chain')}",
                f"PDA预测误差: {response.get('pda_prediction_error', 0):.4f}",
                f"知识库向量数: {response.get('core_kb_vector_count', 0)}"
            ]
            
            for info in debug_info:
                self.addMessage.emit("system", info)
                
            self.updateStatus.emit("就绪")
            
        except Exception as e:
            self.addMessage.emit("system", f"处理错误: {str(e)}")
            self.updateStatus.emit("错误")
    
    @pyqtSlot()
    def saveKnowledgeBase(self):
        """保存知识库"""
        try:
            self.updateStatus.emit("保存知识库...")
            self.system.save_all_memory()
            self.addMessage.emit("system", "核心知识库已保存")
            self.updateStatus.emit("就绪")
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
    window.setFixedSize(1000, 750)  # 匹配原始UI尺寸
    
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
    
    # 添加菜单项
    actions = [
        ("显示主窗口", window.show),
        ("隐藏主窗口", window.hide),
        ("保存知识库", bridge.saveKnowledgeBase),
        ("退出系统", bridge.exitApplication)
    ]
    
    for text, callback in actions:
        action = tray_menu.addAction(text)
        action.triggered.connect(callback)
    
    tray_icon.setContextMenu(tray_menu)
    tray_icon.show()
    
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
    run_gui(config, system)
