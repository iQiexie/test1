#!/usr/bin/env python3
"""
Простой тест генерации без LoRA
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine import LightweightEngine

def test_simple_generation():
    """Тест простой генерации без расширений"""
    print("=== Простая генерация без LoRA ===")
    
    # Создаем движок
    engine = LightweightEngine()
    
    # Инициализируем движок
    print("Инициализация движка...")
    engine.setup()
    
    # Параметры генерации
    params = {
        'prompt': 'beautiful landscape with mountains and lake, sunset, photorealistic',
        'negative_prompt': 'blurry, low quality, distorted',
        'width': 512,
        'height': 512,
        'steps': 4,
        'cfg_scale': 1.0,
        'seed': 42,
        'sampler_name': 'euler',
        'scheduler': 'simple',
        # Отключаем все расширения
        'enable_hr': False,
        'adetailer_enabled': False,
        'lora_enabled': False
    }
    
    print(f"Промт: {params['prompt']}")
    print(f"Размер: {params['width']}x{params['height']}")
    print(f"Шаги: {params['steps']}, CFG: {params['cfg_scale']}")
    print("Запуск генерации...")
    
    try:
        # Генерируем
        result = engine.generate(**params)
        
        print(f"Результат генерации: {type(result)}")
        
        # Результат - это список PIL изображений
        if result and isinstance(result, list) and len(result) > 0:
            print("✓ Генерация успешна!")
            print(f"Создано изображений: {len(result)}")
            
            # Сохраним первое изображение для проверки
            if hasattr(result[0], 'save'):
                output_path = "generated_image.png"
                result[0].save(output_path)
                print(f"Изображение сохранено: {output_path}")
                print(f"Размер изображения: {result[0].size}")
                print(f"Режим изображения: {result[0].mode}")
            
            return True
        else:
            print("✗ Генерация не удалась - нет изображений")
            print(f"result: {result}")
            return False
            
    except Exception as e:
        print(f"✗ Ошибка генерации: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_generation()
    if success:
        print("\n🎉 Тест пройден успешно!")
    else:
        print("\n❌ Тест провален")