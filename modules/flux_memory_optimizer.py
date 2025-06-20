"""
Дополнительные оптимизации памяти для Flux моделей в WebUI
"""

import torch
from backend import memory_management
from backend.model_cache_optimizer import cache_optimizer


def optimize_flux_model_loading():
    """Оптимизирует загрузку Flux моделей"""
    
    # Проверяем, что у нас большой GPU
    if memory_management.total_vram > 50 * 1024:  # Больше 50GB
        print("[Flux Memory Optimizer] Применяем оптимизации для большого GPU")
        
        # Устанавливаем более агрессивные настройки
        memory_management.vram_state = memory_management.VRAMState.HIGH_VRAM
        
        # Отключаем принудительную выгрузку моделей
        memory_management.ALWAYS_VRAM_OFFLOAD = False
        
        # Включаем пинирование памяти для быстрого доступа
        memory_management.PIN_SHARED_MEMORY = True
        
        print("[Flux Memory Optimizer] HIGH_VRAM режим активирован")
        print("[Flux Memory Optimizer] Принудительная выгрузка отключена")
        print("[Flux Memory Optimizer] Пинирование памяти включено")
        
        return True
    else:
        print("[Flux Memory Optimizer] Стандартные настройки для небольшого GPU")
        return False


def patch_model_loading_functions():
    """Патчит функции загрузки моделей для оптимизации"""
    
    # Импортируем необходимые модули
    try:
        from modules import sd_models
        from backend.model_cache_optimizer import measure_model_loading
        
        # Патчим функцию загрузки чекпоинта
        original_load_model = sd_models.load_model
        
        @measure_model_loading("Checkpoint")
        def optimized_load_model(*args, **kwargs):
            return original_load_model(*args, **kwargs)
        
        sd_models.load_model = optimized_load_model
        print("[Flux Memory Optimizer] Патч загрузки чекпоинта применен")
        
    except ImportError as e:
        print(f"[Flux Memory Optimizer] Не удалось применить патчи: {e}")


def setup_torch_optimizations():
    """Настраивает оптимизации PyTorch для Flux"""
    
    if torch.cuda.is_available():
        # Включаем оптимизации CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Настраиваем аллокатор памяти для больших моделей
        if memory_management.total_vram > 50 * 1024:
            torch.cuda.set_per_process_memory_fraction(0.95)  # Используем 95% VRAM
            print("[Flux Memory Optimizer] CUDA memory fraction установлена в 95%")
        
        # Включаем оптимизации внимания
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            print("[Flux Memory Optimizer] Flash Attention оптимизации включены")
        
        print("[Flux Memory Optimizer] PyTorch оптимизации применены")


def apply_all_optimizations():
    """Применяет все доступные оптимизации"""
    print("[Flux Memory Optimizer] Начинаем применение оптимизаций...")
    
    # Применяем оптимизации памяти
    memory_optimized = optimize_flux_model_loading()
    
    # Настраиваем PyTorch
    setup_torch_optimizations()
    
    # Применяем патчи
    patch_model_loading_functions()
    
    if memory_optimized:
        print("[Flux Memory Optimizer] Все оптимизации для большого GPU применены успешно")
    else:
        print("[Flux Memory Optimizer] Стандартные оптимизации применены")
    
    return memory_optimized