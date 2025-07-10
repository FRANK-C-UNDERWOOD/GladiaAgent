"""
音频输出节点 - 支持非阻塞播放与队列缓存
"""
import asyncio
import numpy as np
import pyaudio
from typing import Optional, List
from queue import Queue, Empty
import threading
import logging
import time

from ..core.datatypes import AudioFrame, Event
from ..core.event_bus import EventBus

class AudioOutputNode:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device_id: Optional[int] = None,
        buffer_size: int = 1024,
        event_bus: Optional[EventBus] = None
    ):
        """
        初始化音频输出节点
        
        Args:
            sample_rate: 采样率
            channels: 声道数
            device_id: 音频设备ID (None为系统默认)
            buffer_size: 缓冲区大小
            event_bus: 事件总线实例
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_id = device_id
        self.buffer_size = buffer_size
        self.event_bus = event_bus or EventBus()
        
        # 音频播放队列
        self.audio_queue = Queue(maxsize=1000)
        
        # 播放状态
        self.is_playing = False
        self.is_paused = False
        self.should_stop = False
        
        # 音频流相关
        self.audio = None
        self.stream = None
        
        # 播放线程
        self.playback_thread = None
        
        # 统计信息
        self.frames_played = 0
        self.total_duration = 0.0
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def start_playback(self):
        """开始播放"""
        if self.is_playing:
            return
        
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.device_id,
                frames_per_buffer=self.buffer_size
            )
            
            self.is_playing = True
            self.should_stop = False
            
            # 启动播放线程
            self.playback_thread = threading.Thread(target=self._playback_loop)
            self.playback_thread.daemon = True
            self.playback_thread.start()
            
            self.logger.info("音频播放已开始")
            
        except Exception as e:
            self.logger.error(f"启动播放失败: {e}")
            raise
    
    def stop_playback(self):
        """停止播放"""
        if not self.is_playing:
            return
        
        self.should_stop = True
        self.is_playing = False
        
        # 等待播放线程结束
        if self.playback_thread:
            self.playback_thread.join(timeout=5.0)
        
        # 关闭音频流
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        self.logger.info("音频播放已停止")
    
    async def enqueue(self, audio: AudioFrame) -> None:
        """将音频帧加入播放队列"""
        if not self.is_playing:
            return
        
        try:
            self.audio_queue.put(audio, block=False)
        except:
            # 队列满了，丢弃旧数据
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put(audio, block=False)
            except Empty:
                pass
    
    async def flush(self) -> None:
        """清空播放队列"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
        self.logger.info("播放队列已清空")
    
    def pause(self):
        """暂停播放"""
        if self.is_playing and not self.is_paused:
            self.is_paused = True
            self.event_bus.publish("Pause")
            self.logger.info("音频播放已暂停")
    
    def resume(self):
        """恢复播放"""
        if self.is_playing and self.is_paused:
            self.is_paused = False
            self.event_bus.publish("Resume")
            self.logger.info("音频播放已恢复")
    
    def stop(self):
        """停止播放并清空队列"""
        self.flush()
        self.stop_playback()
        self.event_bus.publish("Stop")
    
    def is_playing_audio(self) -> bool:
        """检查是否正在播放音频"""
        return self.is_playing and not self.audio_queue.empty()
    
    def on_interrupt(self):
        """当系统外部指令打断正在播放时触发清除"""
        self.flush()
        self.event_bus.publish("Interrupt")
        self.logger.info("播放被中断，队列已清空")
    
    def _playback_loop(self):
        """播放循环"""
        while self.is_playing and not self.should_stop:
            try:
                # 检查是否暂停
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # 获取音频帧
                try:
                    audio_frame = self.audio_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # 播放音频
                self._play_audio_frame(audio_frame)
                
            except Exception as e:
                self.logger.error(f"播放循环错误: {e}")
    
    def _play_audio_frame(self, audio_frame: AudioFrame):
        """播放音频帧"""
        try:
            # 确保音频数据格式正确
            audio_data = audio_frame.pcm
            
            # 如果是单声道但需要立体声，复制声道
            if self.channels == 2 and len(audio_data.shape) == 1:
                audio_data = np.stack([audio_data, audio_data], axis=-1)
            
            # 限制音频幅度
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # 写入音频流
            self.stream.write(audio_data.astype(np.float32).tobytes())
            
            # 更新统计信息
            self.frames_played += 1
            self.total_duration += len(audio_data) / self.sample_rate
            
        except Exception as e:
            self.logger.error(f"播放音频帧失败: {e}")
    
    def get_playback_stats(self) -> dict:
        """获取播放统计信息"""
        return {
            'frames_played': self.frames_played,
            'total_duration': self.total_duration,
            'queue_size': self.audio_queue.qsize(),
            'is_playing': self.is_playing,
            'is_paused': self.is_paused
        }
    
    def set_volume(self, volume: float):
        """设置音量 (0.0 - 1.0)"""
        # 这里可以实现音量控制
        # 实际实现需要在音频数据上应用音量系数
        volume = max(0.0, min(1.0, volume))
        self.logger.info(f"音量设置为: {volume}")
    
    def __del__(self):
        """析构函数"""
        self.stop_playback()
