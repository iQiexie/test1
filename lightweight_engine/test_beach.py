#!/usr/bin/env python3
"""
Тест генерации девушки на пляже
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine import LightweightEngine

def test_beach_generation():
    """Тест генерации девушки на пляже"""
    print("=== Генерация девушки на пляже ===")
    
    # Создаем движок
    engine = LightweightEngine()
    
    # Инициализируем движок
    print("Инициализация движка...")
    engine.setup()
    
    # Параметры генерации
    params = {
        'prompt': 'beautiful girl on tropical beach, summer day, blue ocean, white sand, palm trees, photorealistic, high quality',
        'negative_prompt': 'blurry, low quality, distorted',
        'width': 768,
        'height': 512,
        'steps': 15,
        'cfg_scale': 2.5,
        'seed': 123,
        'sampler_name': 'euler',
        'scheduler': 'simple',
        # Отключаем расширения
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
                output_path = "beach_girl.png"
                result[0].save(output_path)
                print(f"Изображение сохранено: {output_path}")
                print(f"Размер изображения: {result[0].size}")
                
                # Проверим что изображение не черное
                import numpy as np
                img_array = np.array(result[0])
                mean_brightness = np.mean(img_array)
                print(f"Средняя яркость: {mean_brightness:.2f} (0=черный, 255=белый)")
                
                if mean_brightness > 20:
                    print("✓ Изображение создано корректно!")
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
    success = test_beach_generation()
    if success:
        print("\n🏖️ Девушка на пляже создана успешно!")
    else:
        print("\n❌ Тест провален")