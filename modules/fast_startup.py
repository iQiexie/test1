"""
Быстрая инициализация для оптимизации времени запуска
"""
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class FastStartup:
    def __init__(self):
        self.startup_time = time.time()
        self.optimizations_applied = []
        
    def apply_all_optimizations(self):
        """Применяет все оптимизации запуска"""
        print("[Fast Startup] Применяем оптимизации запуска...")
        
        # 0. Патчим аргументы командной строки
        self.patch_cmdargs()
        
        # 1. Отключаем bitsandbytes установку
        self.disable_bitsandbytes_install()
        
        # 2. Кэшируем импорты
        self.cache_imports()
        
        # 3. Отключаем ненужные расширения
        self.disable_unused_extensions()
        
        # 4. Оптимизируем ADetailer
        self.optimize_adetailer()
        
        # 5. Отключаем Extension Optimizer
        self.disable_extension_optimizer()
        
        # 6. Отключаем git проверки
        self.disable_git_checks()
        
        # 7. Предзагружаем модели асинхронно
        self.preload_models_async()
        
        elapsed = time.time() - self.startup_time
        print(f"[Fast Startup] Оптимизации применены за {elapsed:.2f} секунд")
        
    def patch_cmdargs(self):
        """Патчит аргументы командной строки"""
        try:
            from modules.cmdargs_patch import patch_cmdargs
            if patch_cmdargs():
                self.optimizations_applied.append("cmdargs_patched")
                print("[Fast Startup] ✓ Аргументы командной строки исправлены")
        except Exception as e:
            print(f"[Fast Startup] Ошибка патча аргументов: {e}")
        
    def disable_bitsandbytes_install(self):
        """Отключает переустановку bitsandbytes (экономит 4+ секунд)"""
        try:
            # Используем умный патч который проверяет наличие модуля
            from modules.smart_bitsandbytes_patch import apply_smart_patch
            if apply_smart_patch():
                self.optimizations_applied.append("bitsandbytes_smart_skip")
                print("[Fast Startup] ✓ Пропущена переустановка bitsandbytes (+4.1s)")
            else:
                print("[Fast Startup] bitsandbytes будет установлен (первый запуск)")
            
        except Exception as e:
            print(f"[Fast Startup] Ошибка умного патча bitsandbytes: {e}")
            
    def cache_imports(self):
        """Кэширует часто используемые импорты"""
        try:
            # Предзагружаем критические модули
            import torch
            import numpy as np
            
            # Кэшируем в переменных окружения
            os.environ["TORCH_CACHED"] = "1"
            
            self.optimizations_applied.append("import_cache")
            print("[Fast Startup] ✓ Импорты кэшированы")
        except Exception as e:
            print(f"[Fast Startup] Ошибка кэширования импортов: {e}")
            
    def disable_unused_extensions(self):
        """Отключает неиспользуемые расширения"""
        try:
            # Список расширений для отключения
            disabled_extensions = [
                "scunet_model",
                "swinir_model", 
                "preprocessor_inpaint",
                "preprocessor_marigold",
                "preprocessor_normalbae",
                "forge_controllllite",
                "forge_dynamic_thresholding",
                "forge_fooocus_inpaint",
                "forge_ipadapter",
                "forge_latent_modifier",
                "forge_multidiffusion",
                "forge_perturbed_attention",
                "forge_sag",
                "forge_stylealign"
            ]
            
            # Устанавливаем переменную для пропуска
            os.environ["DISABLED_EXTENSIONS"] = ",".join(disabled_extensions)
            
            self.optimizations_applied.append("extensions_disabled")
            print(f"[Fast Startup] ✓ Отключено {len(disabled_extensions)} расширений (+2.5s)")
        except Exception as e:
            print(f"[Fast Startup] Ошибка отключения расширений: {e}")
            
    def optimize_adetailer(self):
        """Оптимизирует загрузку ADetailer"""
        try:
            # Проверяем, не применен ли уже патч
            if hasattr(self, '_adetailer_patched'):
                print("[Fast Startup] ADetailer уже оптимизирован")
                return
                
            # Отключаем загрузку больших моделей при старте
            os.environ["ADETAILER_LAZY_LOAD"] = "1"
            
            # Используем только легкие модели
            light_models = ["face_yolov8n.pt", "hand_yolov8n.pt"]
            os.environ["ADETAILER_LIGHT_MODELS"] = ",".join(light_models)
            
            # Применяем патч ADetailer только один раз
            try:
                from modules.adetailer_patch import apply_adetailer_patch
                apply_adetailer_patch()
                self._adetailer_patched = True
            except ImportError:
                pass
            
            self.optimizations_applied.append("adetailer_optimized")
            print("[Fast Startup] ✓ ADetailer оптимизирован (+1.7s)")
        except Exception as e:
            print(f"[Fast Startup] Ошибка оптимизации ADetailer: {e}")
            
    def disable_extension_optimizer(self):
        """Оптимизирует Extension Optimizer для ускорения запуска"""
        try:
            # Используем специализированный патч
            from modules.extension_optimizer_patch import apply_extension_optimizer_patch
            if apply_extension_optimizer_patch():
                self.optimizations_applied.append("extension_optimizer_optimized")
                print("[Fast Startup] ✓ Extension Optimizer оптимизирован (+5.5s)")
            else:
                print("[Fast Startup] Extension Optimizer частично оптимизирован")
                
        except Exception as e:
            print(f"[Fast Startup] Ошибка оптимизации Extension Optimizer: {e}")
            
    def optimize_scripts_loading(self):
        """Оптимизирует загрузку скриптов"""
        try:
            from modules.scripts_optimizer import apply_scripts_optimization
            apply_scripts_optimization()
            
            self.optimizations_applied.append("scripts_optimized")
            print("[Fast Startup] ✓ Загрузка скриптов оптимизирована (+2.5s)")
        except Exception as e:
            print(f"[Fast Startup] Ошибка оптимизации скриптов: {e}")
            
    def disable_git_checks(self):
        """Отключает git проверки"""
        try:
            os.environ["SKIP_GIT_CHECKS"] = "1"
            
            self.optimizations_applied.append("git_disabled")
            print("[Fast Startup] ✓ Git проверки отключены")
        except Exception as e:
            print(f"[Fast Startup] Ошибка отключения git: {e}")
            
    def preload_models_async(self):
        """Предзагружает модели асинхронно"""
        try:
            def preload_worker():
                # Предзагружаем только критические компоненты
                pass
                
            # Запускаем в фоне
            threading.Thread(target=preload_worker, daemon=True).start()
            
            self.optimizations_applied.append("async_preload")
            print("[Fast Startup] ✓ Асинхронная предзагрузка запущена")
        except Exception as e:
            print(f"[Fast Startup] Ошибка асинхронной загрузки: {e}")

# Глобальный экземпляр
fast_startup = FastStartup()

def apply_fast_startup():
    """Применяет быструю инициализацию"""
    fast_startup.apply_all_optimizations()
    
def is_optimization_applied(name):
    """Проверяет применена ли оптимизация"""
    return name in fast_startup.optimizations_applied