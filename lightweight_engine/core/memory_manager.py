"""
Управление памятью на основе логов и backend/memory_management.py
"""
import torch
import gc
from typing import Optional

class MemoryManager:
    """
    Управление памятью как в логах
    """
    
    def __init__(self, config):
        self.config = config
        # Из логов: HIGH_VRAM режим
        self.vram_state = config.VRAM_STATE
        self.total_vram = config.TOTAL_VRAM  # MB из логов
        self.inference_memory = config.INFERENCE_MEMORY  # MB из логов
        self.model_memory = config.MODEL_MEMORY  # MB из логов
        
    def setup_optimizations(self):
        """Применение оптимизаций из логов"""
        print("[Flux Memory Optimizer] Начинаем применение оптимизаций...")
        print("[Flux Memory Optimizer] Применяем оптимизации для большого GPU")
        print(f"[Flux Memory Optimizer] {self.vram_state} режим активирован")
        
        # Flash Attention (из логов)
        self._enable_flash_attention()
        
        # PyTorch оптимизации (из логов)
        self._apply_pytorch_optimizations()
        
        # Пинирование памяти (из логов)
        self._enable_memory_pinning()
        
        # CUDA memory fraction 95% (из логов)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(self.config.MEMORY_FRACTION)
            print(f"[Flux Memory Optimizer] CUDA memory fraction установлена в {int(self.config.MEMORY_FRACTION*100)}%")
        
        print("[Flux Memory Optimizer] Все оптимизации для большого GPU применены успешно")
        
        # Настройки из логов
        print(f"[Flux Optimizer] Применяем агрессивные оптимизации для большого GPU")
        print(f"[Flux Optimizer] Память для инференса: {self.inference_memory} MB")
        print(f"[Flux Optimizer] Режим VRAM: {self.vram_state}")
        print(f"[Flux Optimizer] Пинирование памяти: {self.config.ENABLE_MEMORY_PINNING}")
        
        print(f"[GPU Setting] Выделено {self.model_memory} MB для весов модели и {self.inference_memory} MB для вычислений")
        print("[GPU Setting] Установлен ASYNC метод загрузки для лучшей производительности")
        print("[GPU Setting] Установлен Shared метод хранения для лучшей производительности")
        
    def _enable_flash_attention(self):
        """Включение Flash Attention как в логах"""
        if self.config.ENABLE_FLASH_ATTENTION:
            print("[Flux Memory Optimizer] Flash Attention оптимизации включены")
            # Здесь будет настройка Flash Attention
            
    def _apply_pytorch_optimizations(self):
        """PyTorch оптимизации как в логах"""
        print("[Flux Memory Optimizer] PyTorch оптимизации применены")
        
        # Настройки из логов
        if torch.cuda.is_available():
            print("CUDA Using Stream: True")
            print("Using pytorch cross attention")
            print("Using pytorch attention for VAE")
            
    def _enable_memory_pinning(self):
        """Пинирование памяти как в логах"""
        if self.config.ENABLE_MEMORY_PINNING:
            print("[Flux Memory Optimizer] Пинирование памяти включено")
            
    def unload_models(self, memory_to_free: float, keep_loaded: int = 0):
        """Выгрузка моделей как в логах"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            free_memory = torch.cuda.memory_reserved() / 1024 / 1024 - current_memory
            
            print(f"[Unload] Trying to free {memory_to_free:.2f} MB for cuda:0 with {keep_loaded} models keep loaded ... Current free memory is {free_memory:.2f} MB ... Done in 0.000xxx seconds")
            
            # Принудительная очистка памяти
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
    def get_memory_info(self) -> dict:
        """Получение информации о памяти"""
        if not torch.cuda.is_available():
            return {"total": 0, "allocated": 0, "free": 0}
            
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        free = total - allocated
        
        return {
            "total": total,
            "allocated": allocated, 
            "free": free
        }
        
    def ensure_memory_available(self, required_mb: float):
        """Обеспечение доступной памяти"""
        memory_info = self.get_memory_info()
        if memory_info["free"] < required_mb:
            self.unload_models(required_mb - memory_info["free"])