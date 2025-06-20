"""
Блокировщик pip установки для bitsandbytes
"""
import sys
import subprocess
import os

class PipBlocker:
    def __init__(self):
        self.original_functions = {}
        self.blocked = False
        
    def block_pip_install(self):
        """Блокирует все способы установки через pip"""
        if self.blocked:
            return
            
        try:
            # Блокируем subprocess.run
            if not hasattr(subprocess, '_original_run'):
                subprocess._original_run = subprocess.run
                subprocess.run = self._patched_run
            
            # Блокируем subprocess.Popen
            if not hasattr(subprocess, '_original_popen'):
                subprocess._original_popen = subprocess.Popen
                subprocess.Popen = self._patched_popen
            
            # Блокируем os.system
            if not hasattr(os, '_original_system'):
                os._original_system = os.system
                os.system = self._patched_system
            
            self.blocked = True
            print("[Pip Blocker] Все методы установки pip заблокированы")
            
        except Exception as e:
            print(f"[Pip Blocker] Ошибка блокировки: {e}")
    
    def _patched_run(self, *args, **kwargs):
        """Патченная версия subprocess.run"""
        if self._should_block_command(args):
            print("[Pip Blocker] Заблокирована установка bitsandbytes через subprocess.run")
            return self._fake_success_result()
        return subprocess._original_run(*args, **kwargs)
    
    def _patched_popen(self, *args, **kwargs):
        """Патченная версия subprocess.Popen"""
        if self._should_block_command(args):
            print("[Pip Blocker] Заблокирована установка bitsandbytes через subprocess.Popen")
            return self._fake_process()
        return subprocess._original_popen(*args, **kwargs)
    
    def _patched_system(self, command):
        """Патченная версия os.system"""
        if self._should_block_command([command]):
            print("[Pip Blocker] Заблокирована установка bitsandbytes через os.system")
            return 0
        return os._original_system(command)
    
    def _should_block_command(self, args):
        """Проверяет, нужно ли блокировать команду"""
        if not args:
            return False
            
        cmd_str = str(args[0]) if args else ""
        if isinstance(args[0], list):
            cmd_str = " ".join(str(x) for x in args[0])
        
        return ('pip' in cmd_str and 'install' in cmd_str and 'bitsandbytes' in cmd_str) or \
               ('python' in cmd_str and '-m pip' in cmd_str and 'bitsandbytes' in cmd_str)
    
    def _fake_success_result(self):
        """Возвращает поддельный успешный результат"""
        class FakeResult:
            returncode = 0
            stdout = b"Requirement already satisfied: bitsandbytes==0.45.3"
            stderr = b""
            
            def check_returncode(self):
                pass
        
        return FakeResult()
    
    def _fake_process(self):
        """Возвращает поддельный процесс"""
        class FakeProcess:
            returncode = 0
            pid = 12345
            
            def wait(self, timeout=None):
                return 0
            
            def communicate(self, input=None, timeout=None):
                return (b"Requirement already satisfied: bitsandbytes==0.45.3", b"")
            
            def poll(self):
                return 0
            
            def terminate(self):
                pass
            
            def kill(self):
                pass
        
        return FakeProcess()

# Глобальный экземпляр
pip_blocker = PipBlocker()

def block_bitsandbytes_install():
    """Блокирует установку bitsandbytes"""
    pip_blocker.block_pip_install()