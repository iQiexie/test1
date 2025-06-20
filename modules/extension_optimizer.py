"""
Оптимизатор расширений для WebUI Forge
Отключает неиспользуемые расширения для ускорения загрузки
"""

import os
import json
import sys
from pathlib import Path

class ExtensionOptimizer:
    def __init__(self):
        # Путь к конфигурации относительно корня проекта
        import os
        from modules import paths
        self.config_path = os.path.join(paths.script_path, "config_optimized.json")
        self.config = self.load_config()
        
    def load_config(self):
        """Загружает конфигурацию оптимизации"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[Extension Optimizer] Ошибка загрузки конфигурации: {e}")
        
        # Возвращаем пустую конфигурацию если файл не найден
        return {
            "disabled_extensions": [],
            "disabled_scripts": [],
            "adetailer_model_paths": {},
            "esrgan_model_path": "",
            "optimization_settings": {}
        }
    
    def should_disable_extension(self, extension_name):
        """Проверяет, нужно ли отключить расширение"""
        disabled = self.config.get("disabled_extensions", [])
        return any(disabled_ext in extension_name.lower() for disabled_ext in disabled)
    
    def should_disable_script(self, script_name):
        """Проверяет, нужно ли отключить скрипт"""
        disabled = self.config.get("disabled_scripts", [])
        return any(disabled_script in script_name.lower() for disabled_script in disabled)
    
    def get_adetailer_model_path(self, model_name):
        """Возвращает локальный путь к модели ADetailer"""
        from modules import paths
        model_paths = self.config.get("adetailer_model_paths", {})
        relative_path = model_paths.get(model_name, None)
        if relative_path:
            return os.path.join(paths.script_path, relative_path)
        return None
    
    def get_esrgan_model_path(self):
        """Возвращает локальный путь к модели ESRGAN"""
        from modules import paths
        relative_path = self.config.get("esrgan_model_path", "")
        if relative_path:
            return os.path.join(paths.script_path, relative_path)
        return ""
    
    def apply_optimization_settings(self):
        """Применяет настройки оптимизации"""
        settings = self.config.get("optimization_settings", {})
        
        # Устанавливаем переменные окружения для оптимизации
        if settings.get("skip_model_hash_calculation", False):
            os.environ["WEBUI_SKIP_MODEL_HASH"] = "1"
        
        if settings.get("disable_safe_unpickle", False):
            os.environ["WEBUI_DISABLE_SAFE_UNPICKLE"] = "1"
            
        if settings.get("disable_extension_access_control", False):
            os.environ["WEBUI_DISABLE_EXTENSION_ACCESS_CONTROL"] = "1"
            
        if settings.get("skip_version_check", False):
            os.environ["WEBUI_SKIP_VERSION_CHECK"] = "1"
            
        if settings.get("disable_console_progressbars", False):
            os.environ["WEBUI_DISABLE_CONSOLE_PROGRESSBARS"] = "1"
            
        if settings.get("enable_batch_cond_uncond", False):
            os.environ["WEBUI_ENABLE_BATCH_COND_UNCOND"] = "1"
            
        if settings.get("use_cpu_for_conditioning", False):
            os.environ["WEBUI_USE_CPU_FOR_CONDITIONING"] = "1"
    
    def patch_script_loading(self):
        """Патчит загрузку скриптов для отключения неиспользуемых"""
        try:
            # Импортируем модули WebUI
            import modules.scripts as scripts_module
            
            # Сохраняем оригинальную функцию
            original_load_scripts = getattr(scripts_module, 'load_scripts', None)
            
            if original_load_scripts:
                def optimized_load_scripts(*args, **kwargs):
                    print("[Extension Optimizer] Применяем оптимизацию загрузки скриптов...")
                    
                    # Вызываем оригинальную функцию
                    result = original_load_scripts(*args, **kwargs)
                    
                    # Фильтруем скрипты в ScriptRunner объектах
                    if hasattr(scripts_module, 'scripts_txt2img') and scripts_module.scripts_txt2img:
                        if hasattr(scripts_module.scripts_txt2img, 'scripts'):
                            original_scripts = scripts_module.scripts_txt2img.scripts[:]
                            scripts_module.scripts_txt2img.scripts = [
                                script for script in original_scripts
                                if not self.should_disable_script(script.__class__.__name__)
                            ]
                            # Также фильтруем alwayson_scripts и selectable_scripts
                            if hasattr(scripts_module.scripts_txt2img, 'alwayson_scripts'):
                                scripts_module.scripts_txt2img.alwayson_scripts = [
                                    script for script in scripts_module.scripts_txt2img.alwayson_scripts
                                    if not self.should_disable_script(script.__class__.__name__)
                                ]
                            if hasattr(scripts_module.scripts_txt2img, 'selectable_scripts'):
                                scripts_module.scripts_txt2img.selectable_scripts = [
                                    script for script in scripts_module.scripts_txt2img.selectable_scripts
                                    if not self.should_disable_script(script.__class__.__name__)
                                ]
                    
                    if hasattr(scripts_module, 'scripts_img2img') and scripts_module.scripts_img2img:
                        if hasattr(scripts_module.scripts_img2img, 'scripts'):
                            original_scripts = scripts_module.scripts_img2img.scripts[:]
                            scripts_module.scripts_img2img.scripts = [
                                script for script in original_scripts
                                if not self.should_disable_script(script.__class__.__name__)
                            ]
                            # Также фильтруем alwayson_scripts и selectable_scripts
                            if hasattr(scripts_module.scripts_img2img, 'alwayson_scripts'):
                                scripts_module.scripts_img2img.alwayson_scripts = [
                                    script for script in scripts_module.scripts_img2img.alwayson_scripts
                                    if not self.should_disable_script(script.__class__.__name__)
                                ]
                            if hasattr(scripts_module.scripts_img2img, 'selectable_scripts'):
                                scripts_module.scripts_img2img.selectable_scripts = [
                                    script for script in scripts_module.scripts_img2img.selectable_scripts
                                    if not self.should_disable_script(script.__class__.__name__)
                                ]
                    
                    disabled_count = len(self.config.get("disabled_scripts", []))
                    print(f"[Extension Optimizer] Отключено {disabled_count} неиспользуемых скриптов")
                    
                    return result
                
                # Заменяем функцию
                scripts_module.load_scripts = optimized_load_scripts
                
        except Exception as e:
            print(f"[Extension Optimizer] Ошибка патчинга загрузки скриптов: {e}")
    
    def patch_adetailer_models(self):
        """Патчит ADetailer для использования локальных моделей"""
        try:
            # Устанавливаем переменные окружения для ADetailer
            from modules import paths
            adetailer_models_dir = os.path.join(paths.script_path, "models", "adetailer")
            if os.path.exists(adetailer_models_dir):
                os.environ["ADETAILER_MODELS_PATH"] = adetailer_models_dir
                print(f"[Extension Optimizer] Установлен путь к моделям ADetailer: {adetailer_models_dir}")
                
                # Проверяем наличие моделей
                model_count = len([f for f in os.listdir(adetailer_models_dir) if f.endswith('.pt')])
                print(f"[Extension Optimizer] Найдено {model_count} предустановленных моделей ADetailer")
            else:
                print(f"[Extension Optimizer] Директория моделей ADetailer не найдена: {adetailer_models_dir}")
                
        except Exception as e:
            print(f"[Extension Optimizer] Ошибка настройки ADetailer: {e}")
    
    def patch_esrgan_models(self):
        """Патчит ESRGAN для использования локальных моделей"""
        try:
            esrgan_path = self.get_esrgan_model_path()
            if esrgan_path and os.path.exists(esrgan_path):
                os.environ["ESRGAN_MODEL_PATH"] = esrgan_path
                print(f"[Extension Optimizer] Установлен путь к модели ESRGAN: {esrgan_path}")
                
        except Exception as e:
            print(f"[Extension Optimizer] Ошибка настройки ESRGAN: {e}")
    
    def optimize(self):
        """Применяет все оптимизации"""
        print("[Extension Optimizer] Запуск оптимизации расширений...")
        
        # Применяем настройки оптимизации
        self.apply_optimization_settings()
        
        # Настраиваем пути к моделям
        self.patch_adetailer_models()
        self.patch_esrgan_models()
        
        # Патчим загрузку скриптов
        self.patch_script_loading()
        
        print("[Extension Optimizer] Оптимизация завершена")

# Глобальный экземпляр оптимизатора
extension_optimizer = ExtensionOptimizer()

def initialize_extension_optimizer():
    """Инициализирует оптимизатор расширений"""
    extension_optimizer.optimize()

# Автоматическая инициализация отключена - теперь вызывается из initialize.py
# if __name__ != "__main__":
#     initialize_extension_optimizer()