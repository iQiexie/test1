"""
Патч для ускорения Extension Optimizer
"""
import os
import sys

def patch_extension_optimizer():
    """Патчит Extension Optimizer для быстрого запуска"""
    try:
        # Патчим на уровне модуля initialize
        import modules.initialize as init_module
        
        # Сохраняем оригинальную функцию
        if hasattr(init_module, 'extension_optimizer') and not hasattr(init_module, '_original_extension_optimizer'):
            init_module._original_extension_optimizer = init_module.extension_optimizer
            
            def fast_extension_optimizer():
                """Быстрая версия extension_optimizer"""
                print("[Extension Optimizer Patch] Быстрая инициализация")
                
                # Выполняем только критические операции
                try:
                    # Устанавливаем базовые пути
                    os.environ.setdefault('ADETAILER_MODEL_PATH', '/src/models/adetailer')
                    os.environ.setdefault('ESRGAN_MODEL_PATH', '/src/models/ESRGAN/ESRGAN_4x.pth')
                    
                    # Пропускаем тяжелые операции
                    print("[Extension Optimizer Patch] Пропущены тяжелые операции инициализации")
                    
                except Exception as e:
                    print(f"[Extension Optimizer Patch] Ошибка быстрой инициализации: {e}")
                    # В случае ошибки вызываем оригинальную функцию
                    return init_module._original_extension_optimizer()
            
            # Заменяем функцию
            init_module.extension_optimizer = fast_extension_optimizer
            print("[Extension Optimizer Patch] ✓ Extension Optimizer пропатчен")
            return True
            
    except ImportError:
        print("[Extension Optimizer Patch] Модуль initialize не найден")
        return False
    except Exception as e:
        print(f"[Extension Optimizer Patch] Ошибка патча: {e}")
        return False
    
    return False

def patch_extension_scripts():
    """Патчит загрузку скриптов расширений"""
    try:
        # Патчим прямые вызовы Extension Optimizer
        try:
            import extensions.extension_optimizer.scripts.extension_optimizer as ext_opt
            
            if hasattr(ext_opt, 'initialize') and not hasattr(ext_opt, '_original_initialize'):
                ext_opt._original_initialize = ext_opt.initialize
                
                def fast_initialize():
                    """Быстрая инициализация расширения"""
                    print("[Extension Optimizer Patch] Быстрая инициализация скриптов")
                    # Пропускаем тяжелые операции
                    return
                
                ext_opt.initialize = fast_initialize
                print("[Extension Optimizer Patch] ✓ Скрипты расширений пропатчены")
                
        except ImportError:
            print("[Extension Optimizer Patch] Скрипты расширений не найдены")
            
    except Exception as e:
        print(f"[Extension Optimizer Patch] Ошибка патча скриптов: {e}")

def apply_extension_optimizer_patch():
    """Применяет все патчи Extension Optimizer"""
    print("[Extension Optimizer Patch] Применяем патчи...")
    
    success1 = patch_extension_optimizer()
    patch_extension_scripts()
    
    if success1:
        print("[Extension Optimizer Patch] ✓ Патчи применены успешно")
        return True
    else:
        print("[Extension Optimizer Patch] Частичное применение патчей")
        return False