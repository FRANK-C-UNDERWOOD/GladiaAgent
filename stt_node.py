"""
语音识别节点 - 流式语音识别，支持推理中断与逐步生成
"""
import asyncio
import numpy as np
from typing import Optional, List, Set
from queue import Queue, Empty
import threading
import logging
import re

from ..core.datatypes import AudioFrame, TranscriptPiece, Event
from ..core.event_bus import EventBus

class STTNode:
    def __init__(
        self,
        model_name: str = "whisper-base",
        language: Optional[str] = None,
        enable_attention_tagging: bool = True,
        attention_keywords: Optional[List[str]] = None,
        auto_interrupt: bool = True,
        event_bus: Optional[EventBus] = None
    ):
        """
        初始化语音识别节点
        
        Args:
            model_name: 模型名称 (whisper-base, whisper-large等)
            language: 指定语言 (None为自动检测)
            enable_attention_tagging: 是否启用注意力标记
            attention_keywords: 注意力关键词列表
            auto_interrupt: 是否启用自动打断
            event_bus: 事件总线实例
        """
        self.model_name = model_name
        self.language = language
        self.enable_attention_tagging = enable_attention_tagging
        self.attention_keywords = set(attention_keywords or [])
        self.auto_interrupt = auto_interrupt
        self.event_bus = event_bus or EventBus()
        
        # 音频缓冲区
        self.audio_buffer = Queue(maxsize=1000)
        self.text_buffer = Queue(maxsize=100)
        
        # 处理状态
        self.is_processing = False
        self.should_interrupt = False
        
        # 模型相关
        self.model = None
        self._init_model()
        
        # 处理线程
        self.processing_thread = None
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def _init_model(self):
        """初始化语音识别模型"""
        try:
            # 这里可以集成实际的语音识别模型，如Whisper
            # 为了示例，我们使用模拟的模型
            self.model = MockSTTModel(self.model_name)
            self.logger.info(f"STT模型 {self.model_name} 初始化成功")
        except Exception as e:
            self.logger.error(f"STT模型初始化失败: {e}")
            raise
    
    def start_processing(self):
        """开始处理"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.should_interrupt = False
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("STT处理已开始")
    
    def stop_processing(self):
        """停止处理"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        self.should_interrupt = True
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info("STT处理已停止")
    
    async def push_audio(self, audio: AudioFrame) -> None:
        """推送音频数据"""
        if not self.is_processing:
            return
        
        try:
            # 只处理包含语音的帧
            if audio.is_voiced:
                self.audio_buffer.put(audio, block=False)
        except:
            # 缓冲区满了，丢弃旧数据
            try:
                self.audio_buffer.get_nowait()
                self.audio_buffer.put(audio, block=False)
            except Empty:
                pass
    
    async def pull_text(self) -> Optional[TranscriptPiece]:
        """拉取识别结果"""
        try:
            return self.text_buffer.get_nowait()
        except Empty:
            return None
    
    def _processing_loop(self):
        """处理循环"""
        audio_chunk = []
        chunk_start_time = None
        
        while self.is_processing:
            try:
                # 检查是否需要中断
                if self.should_interrupt:
                    break
                
                # 获取音频帧
                try:
                    frame = self.audio_buffer.get(timeout=0.1)
                except Empty:
                    continue
                
                # 记录开始时间
                if chunk_start_time is None:
                    chunk_start_time = frame.timestamp
                
                # 积累音频数据
                audio_chunk.append(frame.pcm)
                
                # 检查是否达到处理条件
                if len(audio_chunk) >= 10:  # 积累足够的音频帧
                    self._process_audio_chunk(audio_chunk, chunk_start_time)
                    audio_chunk = []
                    chunk_start_time = None
                
            except Exception as e:
                self.logger.error(f"处理循环错误: {e}")
        
        # 处理剩余数据
        if audio_chunk:
            self._process_audio_chunk(audio_chunk, chunk_start_time)
    
    def _process_audio_chunk(self, audio_chunk: List[np.ndarray], start_time: float):
        """处理音频块"""
        try:
            # 合并音频数据
            combined_audio = np.concatenate(audio_chunk)
            
            # 使用模型进行识别
            result = self.model.transcribe(combined_audio)
            
            if result and result.strip():
                # 检测语言
                detected_language = self._detect_language(result)
                
                # 创建转录片段
                transcript = TranscriptPiece(
                    text=result,
                    start_time=start_time,
                    end_time=start_time + len(combined_audio) / 16000,  # 假设16kHz
                    is_final=True,
                    language=detected_language
                )
                
                # 检查注意力标记
                if self.enable_attention_tagging:
                    self._check_attention_tags(result)
                
                # 检查自动打断
                if self.auto_interrupt:
                    self._check_auto_interrupt(result)
                
                # 放入输出队列
                try:
                    self.text_buffer.put(transcript, block=False)
                    self.event_bus.publish("TextFinalized")
                except:
                    # 队列满了，丢弃旧数据
                    try:
                        self.text_buffer.get_nowait()
                        self.text_buffer.put(transcript, block=False)
                    except Empty:
                        pass
        
        except Exception as e:
            self.logger.error(f"处理音频块失败: {e}")
    
    def _detect_language(self, text: str) -> Optional[str]:
        """检测文本语言"""
        # 简单的语言检测逻辑
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_chars > english_chars:
            return "zh"
        elif english_chars > chinese_chars:
            return "en"
        else:
            return None
    
    def _check_attention_tags(self, text: str):
        """检查注意力标记"""
        text_lower = text.lower()
        for keyword in self.attention_keywords:
            if keyword.lower() in text_lower:
                self.event_bus.publish("AttentionTagMatched")
                self.logger.info(f"检测到注意力关键词: {keyword}")
                break
    
    def _check_auto_interrupt(self, text: str):
        """检查自动打断条件"""
        # 定义打断关键词
        interrupt_keywords = ["停止", "暂停", "stop", "pause", "等等", "wait"]
        
        text_lower = text.lower()
        for keyword in interrupt_keywords:
            if keyword in text_lower:
                self.event_bus.publish("Interrupt")
                self.logger.info(f"检测到打断指令: {keyword}")
                return
    
    def interrupt(self):
        """手动中断处理"""
        self.should_interrupt = True
        self.event_bus.publish("Interrupt")
    
    def __del__(self):
        """析构函数"""
        self.stop_processing()


class MockSTTModel:
    """模拟STT模型"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.transcribe_count = 0
    
    def transcribe(self, audio: np.ndarray) -> str:
        """模拟转录功能"""
        self.transcribe_count += 1
        
        # 简单的模拟逻辑
        energy = np.mean(np.abs(audio))
        
        if energy > 0.1:
            # 根据能量返回不同的模拟文本
            if energy > 0.3:
                return f"这是一段高能量的语音内容 {self.transcribe_count}"
            else:
                return f"这是一段普通的语音内容 {self.transcribe_count}"
        
        return ""
