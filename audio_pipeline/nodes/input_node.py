"""
音频输入节点 - 负责音频采集、缓存、语音活动检测（VAD）、语言检测
"""
import asyncio
import numpy as np
import pyaudio
import webrtcvad
from typing import Optional, AsyncGenerator
from threading import Thread
from queue import Queue, Empty
import time
import logging

from ..core.datatypes import AudioFrame, Event
from ..core.event_bus import EventBus

class AudioInputNode:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_aggressiveness: int = 1,
        enable_language_detection: bool = True,
        device_id: Optional[int] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        初始化音频输入节点
        
        Args:
            sample_rate: 采样率 (默认16000Hz)
            frame_duration_ms: 帧持续时间 (默认30ms)
            vad_aggressiveness: VAD灵敏度 (0~3)
            enable_language_detection: 是否启用语言检测
            device_id: 音频设备ID (None为系统默认)
            event_bus: 事件总线实例
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.vad_aggressiveness = vad_aggressiveness
        self.enable_language_detection = enable_language_detection
        self.device_id = device_id
        self.event_bus = event_bus or EventBus()
        
        # 计算帧大小
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.bytes_per_frame = self.frame_size * 2  # 16-bit audio
        
        # 初始化VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        # 音频流相关
        self.audio = None
        self.stream = None
        self.audio_queue = Queue(maxsize=100)
        self.is_recording = False
        
        # 语言检测相关
        self.language_detector = None
        if enable_language_detection:
            self._init_language_detector()
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def _init_language_detector(self):
        """初始化语言检测器"""
        try:
            # 这里可以集成语言检测库，如langdetect或者专门的语音语言检测
            # 为了示例，我们使用简单的模拟
            self.language_detector = SimpleLanguageDetector()
        except Exception as e:
            self.logger.warning(f"语言检测器初始化失败: {e}")
            self.enable_language_detection = False
    
    def start_recording(self):
        """开始录音"""
        if self.is_recording:
            return
        
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=self.frame_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            self.logger.info("音频录制已开始")
            
        except Exception as e:
            self.logger.error(f"启动录音失败: {e}")
            raise
    
    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        self.logger.info("音频录制已停止")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if not self.is_recording:
            return (None, pyaudio.paComplete)
        
        try:
            # 将音频数据放入队列
            self.audio_queue.put(in_data, block=False)
        except:
            # 队列满了，丢弃旧数据
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put(in_data, block=False)
            except Empty:
                pass
        
        return (None, pyaudio.paContinue)
    
    async def get_audio_frame(self) -> Optional[AudioFrame]:
        """获取音频帧"""
        if not self.is_recording:
            return None
        
        try:
            # 非阻塞获取音频数据
            raw_data = self.audio_queue.get_nowait()
            
            # 转换为numpy数组
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            pcm = audio_data.astype(np.float32) / 32768.0  # 归一化到[-1, 1]
            
            # VAD检测
            is_voiced = self._detect_voice_activity(raw_data)
            
            # 语言检测
            language = None
            if self.enable_language_detection and is_voiced:
                language = self._detect_language(pcm)
            
            # 创建音频帧
            frame = AudioFrame(
                pcm=pcm,
                timestamp=time.time(),
                is_voiced=is_voiced,
                language=language
            )
            
            # 发布事件
            if is_voiced:
                self.event_bus.publish("SpeechStart")
            
            if language:
                self.event_bus.publish("LanguageDetected")
            
            return frame
            
        except Empty:
            return None
        except Exception as e:
            self.logger.error(f"获取音频帧失败: {e}")
            return None
    
    def _detect_voice_activity(self, raw_data: bytes) -> bool:
        """检测语音活动"""
        try:
            # 确保数据长度正确
            if len(raw_data) != self.bytes_per_frame:
                return False
            
            return self.vad.is_speech(raw_data, self.sample_rate)
        except Exception as e:
            self.logger.error(f"VAD检测失败: {e}")
            return False
    
    def _detect_language(self, pcm: np.ndarray) -> Optional[str]:
        """检测语言"""
        if not self.language_detector:
            return None
        
        try:
            return self.language_detector.detect(pcm)
        except Exception as e:
            self.logger.error(f"语言检测失败: {e}")
            return None
    
    async def stream_audio(self) -> AsyncGenerator[AudioFrame, None]:
        """异步生成音频帧流"""
        while self.is_recording:
            frame = await self.get_audio_frame()
            if frame:
                yield frame
            else:
                await asyncio.sleep(0.01)  # 避免CPU占用过高
    
    def __del__(self):
        """析构函数，确保资源释放"""
        self.stop_recording()


class SimpleLanguageDetector:
    """简单的语言检测器示例"""
    
    def __init__(self):
        self.languages = ["zh", "en", "unknown"]
    
    def detect(self, audio_data: np.ndarray) -> str:
        """
        检测音频中的语言
        这里是一个简化的示例，实际实现需要使用专门的语言检测模型
        """
        # 简单的模拟检测逻辑
        energy = np.mean(np.abs(audio_data))
        
        if energy > 0.1:
            # 基于能量的简单判断（实际需要更复杂的算法）
            return "zh" if energy > 0.2 else "en"
        
        return "unknown"
