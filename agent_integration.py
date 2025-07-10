"""
Agent集成模块 - 将所有音频处理模块整合到Agent系统中
"""
import asyncio
import logging
from typing import Optional, Dict, Any, List

from .core.datatypes import AudioFrame, TranscriptPiece, VoiceStyle, Event
from .core.event_bus import EventBus
from .core.interrupter import InterruptController
from .nodes.input_node import AudioInputNode
from .nodes.stt_node import STTNode
from .nodes.tts_node import TTSNode
from .nodes.output_node import AudioOutputNode

class AudioPipeline:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化音频管道
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 初始化事件总线
        self.event_bus = EventBus()
        
        # 初始化中断控制器
        self.interrupt_controller = InterruptController(self.event_bus)
        
        # 初始化各个节点
        self.input_node = AudioInputNode(
            sample_rate=self.config.get('sample_rate', 16000),
            frame_duration_ms=self.config.get('frame_duration_ms', 30),
            vad_aggressiveness=self.config.get('vad_aggressiveness', 1),
            enable_language_detection=self.config.get('enable_language_detection', True),
            event_bus=self.event_bus
        )
        
        self.stt_node = STTNode(
            model_name=self.config.get('stt_model', 'whisper-base'),
            language=self.config.get('language', None),
            enable_attention_tagging=self.config.get('enable_attention_tagging', True),
            attention_keywords=self.config.get('attention_keywords', []),
            auto_interrupt=self.config.get('auto_interrupt', True),
            event_bus=self.event_bus
        )
        
        self.tts_node = TTSNode(
            model_name=self.config.get('tts_model', 'bark'),
            sample_rate=self.config.get('sample_rate', 16000),
            streaming=self.config.get('tts_streaming', True),
            event_bus=self.event_bus
        )
        
        self.output_node = AudioOutputNode(
            sample_rate=self.config.get('sample_rate', 16000),
            channels=self.config.get('audio_channels', 1),
            event_bus=self.event_bus
        )
        
        # 管道状态
        self.is_running = False
        self.processing_tasks = []
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 设置事件监听
        self._setup_event_listeners()
    
    def _setup_event_listeners(self):
        """设置事件监听器"""
        # 监听中断事件
        self.event_bus.subscribe("Interrupt", self._on_interrupt)
        self.event_bus.subscribe("SystemInterrupt", self._on_system_interrupt)
        self.event_bus.subscribe("SystemResume", self._on_system_resume)
        
        # 监听语音事件
        self.event_bus.subscribe("SpeechStart", self._on_speech_start)
        self.event_bus.subscribe("SpeechEnd", self._on_speech_end)
        self.event_bus.subscribe("TextFinalized", self._on_text_finalized)
        
        # 监听注意力事件
        self.event_bus.subscribe("AttentionTagMatched", self._on_attention_tag_matched)
        self.event_bus.subscribe("LanguageDetected", self._on_language_detected)
    
    async def start(self):
        """启动音频管道"""
        if self.is_running:
            return
        
        self.logger.info("启动音频管道...")
        
        try:
            # 启动各个节点
            self.input_node.start_recording()
            self.stt_node.start_processing()
            self.tts_node.start_processing()
            self.output_node.start_playback()
            
            # 创建异步任务
            self.processing_tasks = [
                asyncio.create_task(self._audio_processing_loop()),
                asyncio.create_task(self._text_processing_loop()),
                asyncio.create_task(self._tts_processing_loop())
            ]
            
            self.is_running = True
            self.logger.info("音频管道启动成功")
            
        except Exception as e:
            self.logger.error(f"启动音频管道失败: {e}")
            raise

    async def stop(self):
        """停止音频管道"""
        if not self.is_running:
            return
        
        self.logger.info("停止音频管道...")
        
        # 停止所有处理任务
        for task in self.processing_tasks:
            task.cancel()
        
        try:
            await asyncio.gather(*self.processing_tasks)
        except asyncio.CancelledError:
            pass
        
        # 停止所有节点
        self.input_node.stop_recording()
        self.stt_node.stop_processing()
        self.tts_node.stop_processing()
        self.output_node.stop_playback()
        
        self.is_running = False
        self.logger.info("音频管道已停止")
    
    async def _audio_processing_loop(self):
        """音频处理循环：从输入获取音频 -> 发送给STT"""
        while self.is_running:
            try:
                audio_frame = await self.input_node.get_audio_frame()
                if audio_frame:
                    await self.stt_node.push_audio(audio_frame)
                else:
                    await asyncio.sleep(0.01)  # 避免CPU占用过高
            except Exception as e:
                self.logger.error(f"音频处理循环错误: {e}")
                await asyncio.sleep(0.1)
    
    async def _text_processing_loop(self):
        """文本处理循环：从STT获取文本 -> 发送给TTS"""
        while self.is_running:
            try:
                transcript = await self.stt_node.pull_text()
                if transcript:
                    # 这里可以添加NLU处理逻辑
                    await self.tts_node.push_text(
                        transcript.text,
                        style="默认",
                        emotion="中性"
                    )
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                self.logger.error(f"文本处理循环错误: {e}")
                await asyncio.sleep(0.1)
    
    async def _tts_processing_loop(self):
        """TTS处理循环：从TTS获取音频 -> 发送给输出"""
        while self.is_running:
            try:
                audio_frame = await self.tts_node.pull_audio()
                if audio_frame:
                    await self.output_node.enqueue(audio_frame)
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                self.logger.error(f"TTS处理循环错误: {e}")
                await asyncio.sleep(0.1)
    
    async def _on_interrupt(self, event: Event, data: Any = None):
        """处理中断事件"""
        self.logger.info(f"接收到中断事件: {event}")
        self.tts_node.interrupt()
        self.output_node.on_interrupt()
    
    async def _on_system_interrupt(self, event: Event, data: Any = None):
        """处理系统中断事件"""
        self.logger.info("系统中断事件触发")
        # 暂停所有节点处理
        self.output_node.pause()
        self.stt_node.stop_processing()
        self.tts_node.stop_processing()
    
    async def _on_system_resume(self, event: Event, data: Any = None):
        """处理系统恢复事件"""
        self.logger.info("系统恢复事件触发")
        # 恢复所有节点处理
        self.output_node.resume()
        self.stt_node.start_processing()
        self.tts_node.start_processing()
    
    async def _on_speech_start(self, event: Event, data: Any = None):
        """处理语音开始事件"""
        self.logger.info("检测到语音开始")
        # 可以在此重置对话状态等
    
    async def _on_speech_end(self, event: Event, data: Any = None):
        """处理语音结束事件"""
        self.logger.info("检测到语音结束")
        # 可以在此触发最终处理逻辑
    
    async def _on_text_finalized(self, event: Event, data: Any = None):
        """处理文本最终化事件"""
        self.logger.info("识别文本已最终化")
    
    async def _on_attention_tag_matched(self, event: Event, data: Any = None):
        """处理注意力标签匹配事件"""
        self.logger.info("检测到注意力标签匹配")
        # 可以在此触发特定响应
    
    async def _on_language_detected(self, event: Event, data: Any = None):
        """处理语言检测事件"""
        self.logger.info("检测到语言变化")
        # 更新STT和TTS的语言设置
        if data and 'language' in data:
            self.stt_node.language = data['language']
            # 可以添加更新TTS语言的逻辑
    
    def get_pipeline_status(self) -> dict:
        """获取管道状态信息"""
        return {
            'is_running': self.is_running,
            'input_status': "运行中" if self.input_node.is_recording else "停止",
            'stt_status': "运行中" if self.stt_node.is_processing else "停止",
            'tts_status': "运行中" if self.tts_node.is_processing else "停止",
            'output_status': "运行中" if self.output_node.is_playing else "停止"
        }
    
    async def push_text_response(self, text: str):
        """直接推送文本响应到TTS"""
        await self.tts_node.push_text(text)
    
    def __del__(self):
        """析构函数，确保资源释放"""
        if self.is_running:
            asyncio.run(self.stop())