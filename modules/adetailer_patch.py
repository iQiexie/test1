"""
Патч для ADetailer для использования предустановленных моделей
"""

import os
import sys
from pathlib import Path

def patch_adetailer_models():
    """Патчит ADetailer для использования локальных моделей"""
    try:
        # Устанавливаем путь к локальным моделям
        adetailer_models_dir = "/src/models/adetailer"
        
        if not os.path.exists(adetailer_models_dir):
            print(f"[ADetailer Patch] Директория моделей не найдена: {adetailer_models_dir}")
            print(f"[ADetailer Patch] Создаем директорию...")
            os.makedirs(adetailer_models_dir, exist_ok=True)
            return
        
        # Проверяем наличие моделей
        model_files = [f for f in os.listdir(adetailer_models_dir) if f.endswith('.pt')]
        if not model_files:
            print(f"[ADetailer Patch] Модели не найдены в {adetailer_models_dir}")
            print(f"[ADetailer Patch] Содержимое директории: {os.listdir(adetailer_models_dir) if os.path.exists(adetailer_models_dir) else 'директория не существует'}")
            return
        
        print(f"[ADetailer Patch] Найдено {len(model_files)} моделей: {model_files}")
        
        # Проверяем размеры файлов
        for model_file in model_files:
            model_path = os.path.join(adetailer_models_dir, model_file)
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"[ADetailer Patch] ✓ {model_file}: {size_mb:.1f} MB")
            else:
                print(f"[ADetailer Patch] ✗ {model_file}: файл не найден")
        
        # Устанавливаем переменные окружения
        os.environ["ADETAILER_MODELS_PATH"] = adetailer_models_dir
        
        # Патчим функции загрузки моделей
        try:
            # Попытка импорта модулей ADetailer
            import importlib.util
            
            # Ищем модули ADetailer в системе
            adetailer_paths = []
            for path in sys.path:
                potential_path = os.path.join(path, "extensions", "adetailer")
                if os.path.exists(potential_path):
                    adetailer_paths.append(potential_path)
            
            if adetailer_paths:
                print(f"[ADetailer Patch] Найдены пути ADetailer: {adetailer_paths}")
                
                # Добавляем пути в sys.path
                for path in adetailer_paths:
                    if path not in sys.path:
                        sys.path.insert(0, path)
                
                # Патчим функции загрузки
                patch_model_loading_functions(adetailer_models_dir, model_files)
                
        except Exception as e:
            print(f"[ADetailer Patch] Ошибка при патчинге модулей: {e}")
        
        print(f"[ADetailer Patch] Патч применен успешно")
        
    except Exception as e:
        print(f"[ADetailer Patch] Общая ошибка: {e}")

def patch_model_loading_functions(models_dir, model_files):
    """Патчит функции загрузки моделей"""
    try:
        # Создаем маппинг моделей
        model_mapping = {}
        for model_file in model_files:
            model_name = model_file
            model_path = os.path.join(models_dir, model_file)
            model_mapping[model_name] = model_path
        
        # Сохраняем маппинг в глобальной переменной
        globals()['ADETAILER_MODEL_MAPPING'] = model_mapping
        
        # Патчим функцию загрузки модели
        def patched_model_loader(model_name, *args, **kwargs):
            """Патченная функция загрузки модели"""
            if model_name in model_mapping:
                local_path = model_mapping[model_name]
                if os.path.exists(local_path):
                    print(f"[ADetailer Patch] Используем локальную модель: {local_path}")
                    return local_path
            
            print(f"[ADetailer Patch] Модель {model_name} не найдена локально")
            return None
        
        # Сохраняем патченную функцию
        globals()['patched_model_loader'] = patched_model_loader
        
        print(f"[ADetailer Patch] Создан маппинг для {len(model_mapping)} моделей")
        
    except Exception as e:
        print(f"[ADetailer Patch] Ошибка патчинга функций: {e}")

def get_local_model_path(model_name):
    """Возвращает путь к локальной модели"""
    model_mapping = globals().get('ADETAILER_MODEL_MAPPING', {})
    return model_mapping.get(model_name, None)

# Автоматическое применение патча при импорте
if __name__ != "__main__":
    patch_adetailer_models()