"""
音频I/O管道核心数据类型定义
"""
from typing import NamedTuple, Optional, Literal
import numpy as np

# 音频帧数据结构
class AudioFrame(NamedTuple):
    pcm: np.ndarray  # shape: [samples], dtype=float32
    timestamp: float  # 秒
    is_voiced: bool  # 是否含语音片段
    language: Optional[str] = None  # 检测语言（可选）

# 语音识别结果数据结构
class TranscriptPiece(NamedTuple):
    text: str
    start_time: float
    end_time: float
    is_final: bool
    language: Optional[str] = None  # 可返回语言标签

# 事件类型定义
Event = Literal[
    "SpeechStart", "SpeechEnd",
    "TextFinalized", "Interrupt", "Pause", "Resume",
    "LanguageDetected", "AttentionTagMatched"
]

# 语音风格配置
class VoiceStyle(NamedTuple):
    speaker_id: Optional[str] = None
    style: Optional[str] = None  # "正式"、"俏皮"、"激动"等
    emotion: Optional[str] = None  # "高兴"、"愤怒"、"沉稳"
    rate: float = 1.0  # 语速，1.0为正常
