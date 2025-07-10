"""
文本转语音节点 - 支持流式音频块逐步输出，带风格控制
"""
import asyncio
import numpy as np
from typing import Optional, Dict, Any
from queue import Queue, Empty
import threading
import logging
import time

from ..core.datatypes import AudioFrame, VoiceStyle, Event
from ..core.event_bus import EventBus

class TTSNode:
    def __init__(
        self,
        model_name: str = "bark",
        default_style: Optional[VoiceStyle] = None,
        sample_rate: int = 16000,
        streaming: bool = True,
        event_bus: Optional[EventBus] = None
    ):
        """
        初始化文本转语音节点
        
        Args:
            model_name: 模型名称 (bark, xtts等)
            default_style: 默认语音风格
            sample_rate: 采样率
            streaming: 是否启用流式输出
            event_bus: 事件总线实例
        """
        self.model_name = model_name
        self.default_style = default_style or VoiceStyle()
        self.sample_rate = sample_rate
        self.streaming = streaming
        self.event_bus = event_bus or EventBus()
        
        # 文本缓冲区和音频输出队列
        self.text_buffer = Queue(maxsize=100)
        self.audio_output_queue = Queue(maxsize=1000)
        
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
        """初始化TTS模型"""
        try:
            # 这里可以集成实际的TTS模型，如Bark、XTTS等
            # 为了示例，我们使用模拟的模型
            self.model = MockTTSModel(self.model_name, self.sample_rate)
            self.logger.info(f"TTS模型 {self.model_name} 初始化成功")
        except Exception as e:
            self.logger.error(f"TTS模型初始化失败: {e}")
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
        
        self.logger.info("TTS处理已开始")
    
    def stop_processing(self):
        """停止处理"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        self.should_interrupt = True
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info("TTS处理已停止")
    
    async def push_text(
        self, 
        text: str, 
        style: Optional[str] = None, 
        emotion: Optional[str] = None
    ) -> None:
        """推送文本进行合成"""
        if not self.is_processing:
            return
        
        # 创建语音风格配置
        voice_style = VoiceStyle(
            speaker_id=self.default_style.speaker_id,
            style=style or self.default_style.style,
            emotion=emotion or self.default_style.emotion,
            rate=self.default_style.rate
        )
        
        # 创建文本任务
        text_task = {
            'text': text,
            'style': voice_style,
            'timestamp': time.time()
        }
        
        try:
            self.text_buffer.put(text_task, block=False)
            self.logger.info(f"推送文本: {text[:50]}...")
        except:
            # 缓冲区满了，丢弃旧数据
            try:
                self.text_buffer.get_nowait()
                self.text_buffer.put(text_task, block=False)
            except Empty:
                pass
    
    async def pull_audio(self) -> Optional[AudioFrame]:
        """拉取合成的音频"""
        try:
            return self.audio_output_queue.get_nowait()
        except Empty:
            return None
    
    def _processing_loop(self):
        """处理循环"""
        while self.is_processing:
            try:
                # 检查是否需要中断
                if self.should_interrupt:
                    break
                
                # 获取文本任务
                try:
                    text_task = self.text_buffer.get(timeout=0.1)
                except Empty:
                    continue
                
                # 处理文本任务
                self._process_text_task(text_task)
                
            except Exception as e:
                self.logger.error(f"处理循环错误: {e}")
    
    def _process_text_task(self, text_task: Dict[str, Any]):
        """处理文本任务"""
        try:
            text = text_task['text']
            style = text_task['style']
            timestamp = text_task['timestamp']
            
            self.logger.info(f"开始合成文本: {text[:50]}...")
            
            # 使用模型进行语音合成
            if self.streaming:
                # 流式合成
                for audio_chunk in self.model.synthesize_streaming(text, style):
                    if self.should_interrupt:
                        break
                    
                    # 创建音频帧
                    audio_frame = AudioFrame(
                        pcm=audio_chunk,
                        timestamp=timestamp,
                        is_voiced=True,
                        language=self._detect_language(text)
                    )
                    
                    # 放入输出队列
                    try:
                        self.audio_output_queue.put(audio_frame, block=False)
                    except:
                        # 队列满了，丢弃旧数据
                        try:
                            self.audio_output_queue.get_nowait()
                            self.audio_output_queue.put(audio_frame, block=False)
                        except Empty:
                            pass
            else:
                # 非流式合成
                audio_data = self.model.synthesize(text, style)
                
                # 分块输出
                chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
                for i in range(0, len(audio_data), chunk_size):
                    if self.should_interrupt:
                        break
                    
                    chunk = audio_data[i:i+chunk_size]
                    audio_frame = AudioFrame(
                        pcm=chunk,
                        timestamp=timestamp + i / self.sample_rate,
                        is_voiced=True,
                        language=self._detect_language(text)
                    )
                    
                    # 放入输出队列
                    try:
                        self.audio_output_queue.put(audio_frame, block=False)
                    except:
                        try:
                            self.audio_output_queue.get_nowait()
                            self.audio_output_queue.put(audio_frame, block=False)
                        except Empty:
                            pass
            
            self.logger.info(f"完成合成文本: {text[:50]}...")
            
        except Exception as e:
            self.logger.error(f"处理文本任务失败: {e}")
    
    def _detect_language(self, text: str) -> Optional[str]:
        """检测文本语言"""
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_chars > english_chars:
            return "zh"
        elif english_chars > chinese_chars:
            return "en"
        else:
            return None
    
    def interrupt(self):
        """中断当前合成"""
        self.should_interrupt = True
        # 清空缓冲区
        while not self.text_buffer.empty():
            try:
                self.text_buffer.get_nowait()
            except Empty:
                break
        
        self.event_bus.publish("Interrupt")
        self.logger.info("TTS合成已中断")
    
    def flush_buffers(self):
        """清空所有缓冲区"""
        while not self.text_buffer.empty():
            try:
                self.text_buffer.get_nowait()
            except Empty:
                break
        
        while not self.audio_output_queue.empty():
            try:
                self.audio_output_queue.get_nowait()
            except Empty:
                break
    
    def __del__(self):
        """析构函数"""
        self.stop_processing()


class MockTTSModel:
    """模拟TTS模型"""
    
    def __init__(self, model_name: str, sample_rate: int):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.synthesis_count = 0
    
    def synthesize(self, text: str, style: VoiceStyle) -> np.ndarray:
        """同步合成"""
        self.synthesis_count += 1
        
        # 根据文本长度生成相应长度的音频
        duration = len(text) * 0.1  # 假设每个字符0.1秒
        num_samples = int(duration * self.sample_rate)
        
        # 生成模拟音频（正弦波）
        t = np.linspace(0, duration, num_samples)
        frequency = 440.0 * style.rate  # 基础频率乘以语速
        
        # 根据情绪调整频率
        if style.emotion == "高兴":
            frequency *= 1.2
        elif style.emotion == "愤怒":
            frequency *= 0.8
        elif style.emotion == "沉稳":
            frequency *= 0.9
        
        # 生成音频数据
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # 添加一些噪声使其更真实
        noise = np.random.normal(0, 0.05, audio.shape)
        audio = audio + noise
        
        return audio.astype(np.float32)
    
    def synthesize_streaming(self, text: str, style: VoiceStyle):
        """流式合成生成器"""
        # 将文本分成小块进行流式处理
        chunk_size = 10  # 每10个字符一块
        
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i+chunk_size]
            chunk_audio = self.synthesize(chunk_text, style)
            
            # 将音频分成更小的块进行流式输出
            stream_chunk_size = int(self.sample_rate * 0.05)  # 50ms chunks
            
            for j in range(0, len(chunk_audio), stream_chunk_size):
                yield chunk_audio[j:j+stream_chunk_size]
                
                # 模拟处理延迟
                import time
                time.sleep(0.01)
