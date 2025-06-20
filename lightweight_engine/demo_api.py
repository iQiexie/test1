#!/usr/bin/env python3
"""
Демонстрация API генерации
"""

import requests
import json
import time

def test_api_generation():
    """Тест генерации через API"""
    print("=== Демонстрация API генерации ===")
    
    # URL API (предполагаем что main.py запущен)
    api_url = "http://localhost:8000/generate"
    
    # Параметры генерации
    payload = {
        "prompt": "beautiful anime girl with blue hair, detailed face, studio lighting",
        "width": 512,
        "height": 512,
        "steps": 10,
        "cfg_scale": 2.0,
        "seed": 42
    }
    
    print(f"Отправляем запрос: {payload['prompt']}")
    print(f"Размер: {payload['width']}x{payload['height']}")
    
    try:
        # Отправляем POST запрос
        response = requests.post(api_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ API запрос успешен!")
            print(f"Статус: {result.get('status', 'unknown')}")
            print(f"Время генерации: {result.get('generation_time', 'unknown')} сек")
            
            if 'images' in result:
                print(f"Создано изображений: {len(result['images'])}")
                
                # Сохраняем первое изображение
                if result['images']:
                    import base64
                    img_data = base64.b64decode(result['images'][0])
                    with open('api_generated.png', 'wb') as f:
                        f.write(img_data)
                    print("Изображение сохранено: api_generated.png")
            
            return True
        else:
            print(f"✗ Ошибка API: {response.status_code}")
            print(f"Ответ: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Не удается подключиться к API")
        print("Убедитесь что main.py запущен (python main.py)")
        return False
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False

if __name__ == "__main__":
    success = test_api_generation()
    if success:
        print("\n🎉 API тест пройден!")
    else:
        print("\n❌ API тест провален")