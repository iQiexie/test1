"""
Умный патч bitsandbytes - пропускает установку если уже установлен
"""
import sys
import os
import subprocess
import importlib.util

def check_bitsandbytes_installed():
    """Проверяет установлен ли bitsandbytes"""
    try:
        import bitsandbytes
        print(f"[Smart Patch] bitsandbytes уже установлен: {bitsandbytes.__version__}")
        return True
    except ImportError:
        return False

def patch_installation_only():
    """Патчит только процесс установки, не сам модуль"""
    if check_bitsandbytes_installed():
        print("[Smart Patch] bitsandbytes найден, блокируем переустановку")
        
        # Блокируем только переустановку
        original_run = subprocess.run
        original_popen = subprocess.Popen
        
        def smart_run(*args, **kwargs):
            if args and 'bitsandbytes' in str(args[0]) and 'install' in str(args[0]):
                print("[Smart Patch] Пропущена переустановка bitsandbytes")
                class FakeResult:
                    returncode = 0
                    stdout = b"Requirement already satisfied: bitsandbytes"
                    stderr = b""
                return FakeResult()
            return original_run(*args, **kwargs)
        
        def smart_popen(*args, **kwargs):
            if args and 'bitsandbytes' in str(args[0]) and 'install' in str(args[0]):
                print("[Smart Patch] Пропущена переустановка bitsandbytes через Popen")
                class FakeProcess:
                    returncode = 0
                    def wait(self): return 0
                    def communicate(self): return (b"Requirement already satisfied", b"")
                return FakeProcess()
            return original_popen(*args, **kwargs)
        
        subprocess.run = smart_run
        subprocess.Popen = smart_popen
        return True
    else:
        print("[Smart Patch] bitsandbytes не найден, разрешаем установку")
        return False

def apply_smart_patch():
    """Применяет умный патч"""
    return patch_installation_only()