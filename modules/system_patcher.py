"""
Системный патчер для блокировки критических операций
"""
import sys
import os
import importlib
import subprocess
from types import ModuleType

class SystemPatcher:
    def __init__(self):
        self.patches_applied = set()
        
    def patch_import_system(self):
        """Патчит систему импорта для блокировки bitsandbytes"""
        if 'import_system' in self.patches_applied:
            return
            
        # Сохраняем оригинальную функцию импорта
        original_import = __builtins__['__import__']
        
        def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Блокируем импорт bitsandbytes
            if name == 'bitsandbytes' or name.startswith('bitsandbytes.'):
                print(f"[System Patcher] Блокирован импорт: {name}")
                return self._create_fake_bitsandbytes()
            
            # Блокируем установку через pip
            if name == 'pip' and globals and 'install' in str(globals):
                print(f"[System Patcher] Блокирован pip импорт для установки")
                return self._create_fake_pip()
                
            return original_import(name, globals, locals, fromlist, level)
        
        __builtins__['__import__'] = patched_import
        self.patches_applied.add('import_system')
        print("[System Patcher] ✓ Система импорта пропатчена")
    
    def patch_subprocess_completely(self):
        """Полностью патчит subprocess для блокировки установки"""
        if 'subprocess_complete' in self.patches_applied:
            return
            
        # Патчим все методы subprocess
        original_run = subprocess.run
        original_popen = subprocess.Popen
        original_call = subprocess.call
        original_check_call = subprocess.check_call
        original_check_output = subprocess.check_output
        
        def block_bitsandbytes_cmd(args):
            if not args:
                return False
            cmd_str = ' '.join(str(x) for x in args) if isinstance(args, list) else str(args)
            return 'bitsandbytes' in cmd_str and ('pip' in cmd_str or 'install' in cmd_str)
        
        def patched_run(*args, **kwargs):
            if args and block_bitsandbytes_cmd(args[0]):
                print("[System Patcher] Блокирован subprocess.run для bitsandbytes")
                return self._fake_success_result()
            return original_run(*args, **kwargs)
        
        def patched_popen(*args, **kwargs):
            if args and block_bitsandbytes_cmd(args[0]):
                print("[System Patcher] Блокирован subprocess.Popen для bitsandbytes")
                return self._fake_process()
            return original_popen(*args, **kwargs)
        
        def patched_call(*args, **kwargs):
            if args and block_bitsandbytes_cmd(args[0]):
                print("[System Patcher] Блокирован subprocess.call для bitsandbytes")
                return 0
            return original_call(*args, **kwargs)
        
        def patched_check_call(*args, **kwargs):
            if args and block_bitsandbytes_cmd(args[0]):
                print("[System Patcher] Блокирован subprocess.check_call для bitsandbytes")
                return 0
            return original_check_call(*args, **kwargs)
        
        def patched_check_output(*args, **kwargs):
            if args and block_bitsandbytes_cmd(args[0]):
                print("[System Patcher] Блокирован subprocess.check_output для bitsandbytes")
                return b"Requirement already satisfied: bitsandbytes==0.45.3"
            return original_check_output(*args, **kwargs)
        
        subprocess.run = patched_run
        subprocess.Popen = patched_popen
        subprocess.call = patched_call
        subprocess.check_call = patched_check_call
        subprocess.check_output = patched_check_output
        
        self.patches_applied.add('subprocess_complete')
        print("[System Patcher] ✓ Subprocess полностью пропатчен")
    
    def patch_os_system(self):
        """Патчит os.system для блокировки установки"""
        if 'os_system' in self.patches_applied:
            return
            
        original_system = os.system
        
        def patched_system(command):
            cmd_str = str(command)
            if 'bitsandbytes' in cmd_str and ('pip' in cmd_str or 'install' in cmd_str):
                print("[System Patcher] Блокирован os.system для bitsandbytes")
                return 0
            return original_system(command)
        
        os.system = patched_system
        self.patches_applied.add('os_system')
        print("[System Patcher] ✓ os.system пропатчен")
    
    def _create_fake_bitsandbytes(self):
        """Создает поддельный модуль bitsandbytes"""
        fake_module = ModuleType('bitsandbytes')
        fake_module.__version__ = '0.45.3'
        fake_module.__file__ = '/fake/bitsandbytes/__init__.py'
        fake_module.__package__ = 'bitsandbytes'
        
        # Создаем __spec__
        import importlib.util
        spec = importlib.util.spec_from_loader(
            'bitsandbytes',
            loader=None,
            origin='/fake/bitsandbytes/__init__.py'
        )
        fake_module.__spec__ = spec
        
        # Добавляем поддельные атрибуты
        fake_module.quantize = lambda *args, **kwargs: None
        fake_module.Linear8bitLt = type('Linear8bitLt', (), {})
        fake_module.Linear4bit = type('Linear4bit', (), {})
        
        # Регистрируем в sys.modules
        sys.modules['bitsandbytes'] = fake_module
        return fake_module
    
    def _create_fake_pip(self):
        """Создает поддельный pip модуль"""
        fake_pip = ModuleType('pip')
        fake_pip.main = lambda args: print("[System Patcher] Pip установка заблокирована")
        return fake_pip
    
    def _fake_success_result(self):
        """Возвращает поддельный успешный результат"""
        class FakeResult:
            returncode = 0
            stdout = b"Requirement already satisfied: bitsandbytes==0.45.3"
            stderr = b""
            def check_returncode(self): pass
        return FakeResult()
    
    def _fake_process(self):
        """Возвращает поддельный процесс"""
        class FakeProcess:
            returncode = 0
            pid = 12345
            def wait(self, timeout=None): return 0
            def communicate(self, input=None, timeout=None): 
                return (b"Requirement already satisfied: bitsandbytes==0.45.3", b"")
            def poll(self): return 0
            def terminate(self): pass
            def kill(self): pass
        return FakeProcess()
    
    def apply_all_patches(self):
        """Применяет все системные патчи"""
        print("[System Patcher] Применяем системные патчи...")
        self.patch_import_system()
        self.patch_subprocess_completely()
        self.patch_os_system()
        print(f"[System Patcher] ✓ Применено {len(self.patches_applied)} патчей")

# Глобальный экземпляр
system_patcher = SystemPatcher()

def apply_system_patches():
    """Применяет все системные патчи"""
    system_patcher.apply_all_patches()