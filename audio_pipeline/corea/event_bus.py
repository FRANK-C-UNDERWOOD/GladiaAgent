"""
事件总线 - 用于模块间解耦通信
"""
import asyncio
from typing import Dict, List, Callable, Any
from threading import Lock
import logging

from .datatypes import Event

class EventBus:
    def __init__(self):
        """初始化事件总线"""
        self.subscribers: Dict[Event, List[Callable]] = {}
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)
    
    def subscribe(self, event: Event, callback: Callable):
        """订阅事件"""
        with self.lock:
            if event not in self.subscribers:
                self.subscribers[event] = []
            self.subscribers[event].append(callback)
        
        self.logger.debug(f"订阅事件: {event}")
    
    def unsubscribe(self, event: Event, callback: Callable):
        """取消订阅事件"""
        with self.lock:
            if event in self.subscribers:
                try:
                    self.subscribers[event].remove(callback)
                    if not self.subscribers[event]:
                        del self.subscribers[event]
                except ValueError:
                    pass
        
        self.logger.debug(f"取消订阅事件: {event}")
    
    def publish(self, event: Event, data: Any = None):
        """发布事件"""
        with self.lock:
            callbacks = self.subscribers.get(event, []).copy()
        
        if callbacks:
            self.logger.debug(f"发布事件: {event}, 订阅者数量: {len(callbacks)}")
            
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        # 异步回调
                        asyncio.create_task(callback(event, data))
                    else:
                        # 同步回调
                        callback(event, data)
                except Exception as e:
                    self.logger.error(f"事件回调执行失败: {e}")
    
    def clear_all(self):
        """清空所有订阅"""
        with self.lock:
            self.subscribers.clear()
        
        self.logger.info("所有事件订阅已清空")
    
    def get_subscribers_count(self, event: Event) -> int:
        """获取事件订阅者数量"""
        with self.lock:
            return len(self.subscribers.get(event, []))
    
    def get_all_events(self) -> List[Event]:
        """获取所有已订阅的事件"""
        with self.lock:
            return list(self.subscribers.keys())
