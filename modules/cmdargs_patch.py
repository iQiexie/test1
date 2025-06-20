"""
Патч для игнорирования неизвестных аргументов командной строки
"""
import os
import sys

def patch_cmdargs():
    """Патчит обработку аргументов командной строки"""
    try:
        # Устанавливаем переменную для игнорирования ошибок аргументов
        os.environ["IGNORE_CMD_ARGS_ERRORS"] = "1"
        
        # Фильтруем проблемные аргументы
        filtered_args = []
        skip_next = False
        
        for i, arg in enumerate(sys.argv):
            if skip_next:
                skip_next = False
                continue
                
            # Пропускаем проблемные аргументы
            if arg.startswith('--await-explicit-shutdown'):
                continue
            elif arg.startswith('--upload-url'):
                continue
            elif arg == '--await-explicit-shutdown':
                skip_next = True
                continue
            elif arg == '--upload-url':
                skip_next = True
                continue
            else:
                filtered_args.append(arg)
        
        # Заменяем sys.argv
        original_count = len(sys.argv)
        sys.argv = filtered_args
        
        print(f"[CMD Args Patch] Отфильтровано аргументов: {len(filtered_args)} из {original_count}")
        return True
        
    except Exception as e:
        print(f"[CMD Args Patch] Ошибка: {e}")
        return False

# Патч применяется явно через вызов функции patch_cmdargs()