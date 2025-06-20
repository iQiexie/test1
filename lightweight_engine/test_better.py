#!/usr/bin/env python3
"""
Тест с улучшенными параметрами
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine import LightweightEngine

def test_better_generation():
    """Тест с лучшими параметрами"""
    print("=== Тест с улучшенными параметрами ===")
    
    # Создаем движок
    engine = LightweightEngine()
    
    # Инициализируем движок
    print("Инициализация движка...")
    engine.setup()
    
    # Лучшие параметры для Flux
    params = {
        'prompt': 'a beautiful mountain landscape at sunset, highly detailed, photorealistic',
        'negative_prompt': '',  # Flux не очень хорошо работает с negative prompts
        'width': 1024,  # Flux лучше работает с большими разрешениями
        'height': 1024,
        'steps': 20,  # Больше шагов для лучшего качества
        'cfg_scale': 3.5,  # Оптимальный CFG для Flux
        'seed': 12345,
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
        
        if result and isinstance(result, list) and len(result) > 0:
            print("✓ Генерация успешна!")
            print(f"Создано изображений: {len(result)}")
            
            # Сохраним изображение
            if hasattr(result[0], 'save'):
                output_path = "better_image.png"
                result[0].save(output_path)
                print(f"Изображение сохранено: {output_path}")
                print(f"Размер изображения: {result[0].size}")
                
                # Проверим что изображение не черное
                import numpy as np
                img_array = np.array(result[0])
                mean_brightness = np.mean(img_array)
                print(f"Средняя яркость: {mean_brightness:.2f} (0=черный, 255=белый)")
                
                if mean_brightness > 10:
                    print("✓ Изображение не черное!")
                    return True
                else:
                    print("✗ Изображение слишком темное")
                    return False
            
        else:
            print("✗ Генерация не удалась")
            return False
            
    except Exception as e:
        print(f"✗ Ошибка генерации: {e}")
        return False

if __name__ == "__main__":
    success = test_better_generation()
    if success:
        print("\n🎉 Тест пройден успешно!")
    else:
        print("\n❌ Тест провален")