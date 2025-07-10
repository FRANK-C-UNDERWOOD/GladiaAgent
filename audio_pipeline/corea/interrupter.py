"""
中断控制器 - 处理系统中断和恢复逻辑
"""
import asyncio
import time
from typing import Optional, Dict, Any
import logging

from .datatypes import Event
from .event_bus import EventBus

class InterruptController:
    def __init__(self, event_bus: EventBus):
        """
        初始化中断控制器
        
        Args:
            event_bus: 事件总线实例
        """
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # 状态管理
        self.is_interrupted = False
        self.interrupt_start_time = None
        self.saved_state = {}
        
        # 订阅中断相关事件
        self.event_bus.subscribe("Interrupt", self._on_interrupt)
        self.event_bus.subscribe("Resume", self._on_resume)
    
    async def _on_interrupt(self, event: Event, data: Any = None):
        """处理中断事件"""
        if self.is_interrupted:
            return
        
        self.is_interrupted = True
        self.interrupt_start_time = time.time()
        
        self.logger.info("系统中断触发")
        
        # 保存当前状态
        await self._save_current_state()
        
        # 通知所有模块进行中断处理
        self.event_bus.publish("SystemInterrupt", {
            'timestamp': self.interrupt_start_time,
            'reason': data.get('reason', 'unknown') if data else 'unknown'
        })
    
    async def _on_resume(self, event: Event, data: Any = None):
        """处理恢复事件"""
        if not self.is_interrupted:
            return
        
        interrupt_duration = time.time() - self.interrupt_start_time
        self.logger.info(f"系统恢复，中断持续时间: {interrupt_duration:.2f}秒")
        
        # 恢复状态
        await self._restore_state()
        
        self.is_interrupted = False
        self.interrupt_start_time = None
        
        # 通知所有模块恢复
        self.event_bus.publish("SystemResume", {
            'interrupt_duration': interrupt_duration,
            'restored_state': self.saved_state
        })
    
    async def _save_current_state(self):
        """保存当前状态"""
        try:
            # 这里可以保存各个模块的状态
            self.saved_state = {
                'timestamp': time.time(),
                'audio_input_state': {},
                'stt_state': {},
                'tts_state': {},
                'audio_output_state': {}
            }
            
            self.logger.debug("当前状态已保存")
            
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
    
    async def _restore_state(self):
        """恢复状态"""
        try:
            if not self.saved_state:
                return
            
            # 这里可以恢复各个模块的状态
            self.logger.debug("状态已恢复")
            
        except Exception as e:
            self.logger.error(f"恢复状态失败: {e}")
    
    def manual_interrupt(self, reason: str = "manual"):
        """手动触发中断"""
        self.event_bus.publish("Interrupt", {'reason': reason})
    
    def manual_resume(self):
        """手动恢复"""
        self.event_bus.publish("Resume")
    
    def get_interrupt_status(self) -> Dict[str, Any]:
        """获取中断状态"""
        return {
            'is_interrupted': self.is_interrupted,
            'interrupt_start_time': self.interrupt_start_time,
            'interrupt_duration': (
                time.time() - self.interrupt_start_time 
                if self.interrupt_start_time else 0
            ),
            'has_saved_state': bool(self.saved_state)
        }
