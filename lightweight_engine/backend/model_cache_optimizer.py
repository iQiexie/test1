"""
Оптимизатор кэша моделей для ускорения загрузки Flux моделей
Специально разработан для RunPod Serverless с большим количеством VRAM
"""

import torch
import time
from typing import Dict, Any, Optional
from backend import memory_management


class ModelCacheOptimizer:
    """Оптимизатор для агрессивного кэширования моделей в GPU памяти"""
    
    def __init__(self):
        self.cached_models = {}
        self.model_load_times = {}
        self.total_vram = memory_management.total_vram
        self.is_high_vram = self.total_vram > 50 * 1024  # Больше 50GB
        
    def should_keep_model_loaded(self, model_name: str) -> bool:
        """Определяет, стоит ли держать модель в памяти"""
        if not self.is_high_vram:
            return False
            
        # Для больших GPU держим основные модели всегда загруженными
        critical_models = [
            'UNet', 'TextEncoder', 'VAE', 'CLIP', 'T5XXL'
        ]
        
        return any(critical in model_name for critical in critical_models)
    
    def optimize_memory_allocation(self) -> Dict[str, Any]:
        """Возвращает оптимизированные настройки памяти"""
        if self.is_high_vram:
            return {
                'inference_memory_ratio': 0.25,  # 25% для инференса
                'model_memory_ratio': 0.70,      # 70% для моделей  
                'buffer_memory_ratio': 0.05,     # 5% буфер
                'aggressive_caching': True,
                'pin_memory': True,
                'async_loading': True
            }
        else:
            return {
                'inference_memory_ratio': 0.60,  # Стандартные настройки
                'model_memory_ratio': 0.35,
                'buffer_memory_ratio': 0.05,
                'aggressive_caching': False,
                'pin_memory': False,
                'async_loading': False
            }
    
    def log_model_loading_time(self, model_name: str, load_time: float):
        """Логирует время загрузки модели"""
        self.model_load_times[model_name] = load_time
        print(f"[Cache Optimizer] {model_name} загружена за {load_time:.2f}s")
        
    def get_optimization_stats(self) -> str:
        """Возвращает статистику оптимизации"""
        total_time = sum(self.model_load_times.values())
        model_count = len(self.model_load_times)
        avg_time = total_time / model_count if model_count > 0 else 0
        
        return f"Загружено моделей: {model_count}, Общее время: {total_time:.2f}s, Среднее: {avg_time:.2f}s"


# Глобальный экземпляр оптимизатора
cache_optimizer = ModelCacheOptimizer()


def apply_flux_optimizations():
    """Применяет оптимизации специально для Flux моделей"""
    settings = cache_optimizer.optimize_memory_allocation()
    
    if settings['aggressive_caching']:
        print("[Flux Optimizer] Применяем агрессивные оптимизации для большого GPU")
        
        # Устанавливаем HIGH_VRAM режим принудительно
        memory_management.vram_state = memory_management.VRAMState.HIGH_VRAM
        
        # Включаем пинирование памяти
        memory_management.PIN_SHARED_MEMORY = settings['pin_memory']
        
        # Уменьшаем минимальную память для инференса
        memory_management.current_inference_memory = int(
            cache_optimizer.total_vram * settings['inference_memory_ratio'] * 1024 * 1024
        )
        
        print(f"[Flux Optimizer] Память для инференса: {memory_management.current_inference_memory / (1024*1024):.0f} MB")
        print(f"[Flux Optimizer] Режим VRAM: {memory_management.vram_state.name}")
        print(f"[Flux Optimizer] Пинирование памяти: {memory_management.PIN_SHARED_MEMORY}")
        
        return True
    else:
        print("[Flux Optimizer] Используем стандартные настройки для небольшого GPU")
        return False


def measure_model_loading(model_name: str):
    """Декоратор для измерения времени загрузки модели"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            load_time = end_time - start_time
            cache_optimizer.log_model_loading_time(model_name, load_time)
            
            return result
        return wrapper
    return decorator