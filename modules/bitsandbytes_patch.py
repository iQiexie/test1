"""
Патч для отключения установки bitsandbytes при быстрой инициализации
"""
import os
import sys
import subprocess
import importlib.util
from types import ModuleType

def block_pip_install():
    """Блокирует pip install для bitsandbytes"""
    original_run = subprocess.run
    
    def patched_run(*args, **kwargs):
        # Проверяем команду на установку bitsandbytes
        if args and len(args) > 0:
            cmd = args[0] if isinstance(args[0], list) else [str(args[0])]
            cmd_str = ' '.join(str(x) for x in cmd)
            
            if 'pip install' in cmd_str and 'bitsandbytes' in cmd_str:
                print("[Fast Startup] Блокирована установка bitsandbytes через pip")
                # Возвращаем успешный результат без выполнения
                class FakeResult:
                    returncode = 0
                    stdout = b"Requirement already satisfied: bitsandbytes"
                    stderr = b""
                return FakeResult()
        
        return original_run(*args, **kwargs)
    
    subprocess.run = patched_run

def patch_bitsandbytes_import():
    """Патчит импорт bitsandbytes чтобы избежать установки"""
    if os.environ.get("SKIP_BITSANDBYTES_INSTALL") == "1":
        print("[Fast Startup] Пропускаем установку bitsandbytes")
        
        # Блокируем pip install
        block_pip_install()
        
        # Создаем заглушку для bitsandbytes
        import types
        
        # Создаем фиктивный модуль bitsandbytes
        bitsandbytes = types.ModuleType('bitsandbytes')
        bitsandbytes.__version__ = "0.45.3"
        bitsandbytes.__file__ = "/fake/bitsandbytes/__init__.py"
        bitsandbytes.__package__ = "bitsandbytes"
        
        # Создаем правильный __spec__
        spec = importlib.util.spec_from_loader(
            'bitsandbytes',
            loader=None,
            origin="/fake/bitsandbytes/__init__.py"
        )
        bitsandbytes.__spec__ = spec
        
        # Добавляем необходимые атрибуты
        bitsandbytes.cuda_setup = types.ModuleType('cuda_setup')
        bitsandbytes.cuda_setup.common = types.ModuleType('common')
        bitsandbytes.cuda_setup.common.setup_cuda_paths = lambda: None
        
        # Добавляем поддельные функции квантизации
        def fake_quantize(*args, **kwargs):
            raise NotImplementedError("bitsandbytes отключен для оптимизации")
        
        bitsandbytes.quantize = fake_quantize
        bitsandbytes.Linear8bitLt = type('Linear8bitLt', (), {})
        bitsandbytes.Linear4bit = type('Linear4bit', (), {})
        
        # Регистрируем модуль и подмодули
        sys.modules['bitsandbytes'] = bitsandbytes
        sys.modules['bitsandbytes.cuda_setup'] = bitsandbytes.cuda_setup
        sys.modules['bitsandbytes.cuda_setup.common'] = bitsandbytes.cuda_setup.common
        sys.modules['bitsandbytes.nn'] = bitsandbytes
        sys.modules['bitsandbytes.optim'] = bitsandbytes
        sys.modules['bitsandbytes.functional'] = bitsandbytes
        
        return True
    return False

# Применяем патч при импорте модуля
patch_bitsandbytes_import()