"""
Оптимизатор загрузки скриптов для ускорения инициализации
"""
import os
import sys
from typing import List, Set

class ScriptsOptimizer:
    def __init__(self):
        self.disabled_scripts: Set[str] = set()
        self.load_optimizations_applied = False
        
    def get_disabled_scripts(self) -> List[str]:
        """Возвращает список отключенных скриптов"""
        # Скрипты которые можно безопасно отключить для ускорения
        disabled_scripts = [
            "scunet_model.py",
            "swinir_model.py", 
            "preprocessor_inpaint.py",
            "preprocessor_marigold.py",
            "preprocessor_normalbae.py",
            "preprocessor_recolor.py",
            "preprocessor_revision.py",
            "preprocessor_tile.py",
            "forge_controllllite.py",
            "forge_dynamic_thresholding.py",
            "forge_fooocus_inpaint.py",
            "forge_ipadapter.py",
            "forge_latent_modifier.py",
            "forge_multidiffusion.py",
            "forge_perturbed_attention.py",
            "forge_sag.py",
            "forge_stylealign.py",
            "soft_inpainting.py"
        ]
        
        # Проверяем переменную окружения
        env_disabled = os.environ.get("DISABLED_EXTENSIONS", "")
        if env_disabled:
            disabled_scripts.extend(env_disabled.split(","))
            
        return disabled_scripts
        
    def should_skip_script(self, script_path: str) -> bool:
        """Проверяет нужно ли пропустить скрипт"""
        script_name = os.path.basename(script_path)
        return script_name in self.get_disabled_scripts()
        
    def optimize_script_loading(self):
        """Оптимизирует загрузку скриптов"""
        if self.load_optimizations_applied:
            return
            
        try:
            # Патчим функцию загрузки скриптов
            from modules import scripts
            
            # Сохраняем оригинальную функцию
            original_load_scripts = scripts.load_scripts
            
            def optimized_load_scripts():
                """Оптимизированная загрузка скриптов"""
                print("[Scripts Optimizer] Применяем оптимизированную загрузку скриптов...")
                
                # Временно отключаем ненужные скрипты
                disabled_count = 0
                
                # Получаем список скриптов для отключения
                disabled_scripts = self.get_disabled_scripts()
                
                # Патчим импорт для отключенных скриптов
                for script_name in disabled_scripts:
                    if script_name.endswith('.py'):
                        module_name = script_name[:-3]
                        # Создаем заглушку для модуля
                        import types
                        dummy_module = types.ModuleType(module_name)
                        sys.modules[f'scripts.{module_name}'] = dummy_module
                        disabled_count += 1
                
                # Вызываем оригинальную функцию
                result = original_load_scripts()
                
                print(f"[Scripts Optimizer] Отключено {disabled_count} скриптов для ускорения")
                return result
            
            # Заменяем функцию
            scripts.load_scripts = optimized_load_scripts
            
            self.load_optimizations_applied = True
            print("[Scripts Optimizer] Оптимизация загрузки скриптов применена")
            
        except Exception as e:
            print(f"[Scripts Optimizer] Ошибка оптимизации: {e}")

# Глобальный экземпляр
scripts_optimizer = ScriptsOptimizer()

def apply_scripts_optimization():
    """Применяет оптимизацию загрузки скриптов"""
    scripts_optimizer.optimize_script_loading()
    
def is_script_disabled(script_path: str) -> bool:
    """Проверяет отключен ли скрипт"""
    return scripts_optimizer.should_skip_script(script_path)