"""
Тесты для легкого движка
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import Config
from core.engine import LightweightEngine

def test_basic_generation():
    """Тест базовой генерации"""
    print("=== Test Basic Generation ===")
    
    config = Config()
    engine = LightweightEngine(config)
    
    try:
        engine.setup()
        print("✓ Engine setup successful")
    except Exception as e:
        print(f"✗ Engine setup failed: {e}")
        return False
    
    try:
        images = engine.generate(
            prompt="test prompt",
            width=512,
            height=512,
            steps=4,  # Быстрый тест
            adetailer=False,
            num_outputs=1
        )
        
        assert len(images) > 0, "No images generated"
        assert images[0].size == (512, 512), f"Wrong image size: {images[0].size}"
        
        print("✓ Basic generation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic generation test failed: {e}")
        return False

def test_hires_fix():
    """Тест HiRes.fix как в логах"""
    print("=== Test HiRes.fix ===")
    
    config = Config()
    engine = LightweightEngine(config)
    
    try:
        engine.setup()
        
        images = engine.generate(
            prompt="test hires",
            width=512,
            height=512,
            steps=4,
            hr_scale=1.5,  # 1.5x upscaling
            hr_steps=2,    # Быстрый тест
            hr_upscaler="4x-UltraSharp",
            adetailer=False,
            num_outputs=1
        )
        
        # После HiRes.fix размер должен увеличиться
        expected_size = (int(512 * 1.5), int(512 * 1.5))
        assert images[0].size == expected_size, f"HiRes.fix failed: expected {expected_size}, got {images[0].size}"
        
        print("✓ HiRes.fix test passed")
        return True
        
    except Exception as e:
        print(f"✗ HiRes.fix test failed: {e}")
        return False

def test_adetailer():
    """Тест ADetailer как в логах"""
    print("=== Test ADetailer ===")
    
    config = Config()
    engine = LightweightEngine(config)
    
    try:
        engine.setup()
        
        # Тестовые аргументы ADetailer как в логах
        face_args = {
            "ad_model": "face_yolov8s.pt",
            "ad_tab_enable": True,
            "ad_prompt": "perfect detailed face, sharp eyes",
            "ad_confidence": 0.7,
            "ad_denoising_strength": 0.1
        }
        
        hand_args = {
            "ad_model": "hand_yolov8s.pt", 
            "ad_tab_enable": True,
            "ad_prompt": "perfect human hands",
            "ad_confidence": 0.28,
            "ad_denoising_strength": 0.65
        }
        
        images = engine.generate(
            prompt="portrait with hands",
            width=512,
            height=512,
            steps=4,
            adetailer=True,
            adetailer_args=face_args,
            adetailer_args_hands=hand_args,
            num_outputs=1
        )
        
        assert len(images) > 0, "No images generated with ADetailer"
        
        print("✓ ADetailer test passed")
        return True
        
    except Exception as e:
        print(f"✗ ADetailer test failed: {e}")
        return False

def test_lora_loading():
    """Тест загрузки LoRA как в логах"""
    print("=== Test LoRA Loading ===")
    
    config = Config()
    engine = LightweightEngine(config)
    
    try:
        engine.setup()
        
        # Тестовый URL (будет ошибка загрузки, но проверим структуру)
        test_urls = ["https://example.com/test.safetensors"]
        test_scales = [1.0]
        
        try:
            images = engine.generate(
                prompt="test with lora",
                width=512,
                height=512,
                steps=4,
                lora_urls=test_urls,
                lora_scales=test_scales,
                adetailer=False,
                num_outputs=1
            )
            print("✓ LoRA structure test passed (download will fail but that's expected)")
            return True
            
        except Exception as e:
            if "requests" in str(e) or "download" in str(e).lower():
                print("✓ LoRA structure test passed (download failed as expected)")
                return True
            else:
                raise e
        
    except Exception as e:
        print(f"✗ LoRA test failed: {e}")
        return False

def test_memory_management():
    """Тест управления памятью"""
    print("=== Test Memory Management ===")
    
    config = Config()
    engine = LightweightEngine(config)
    
    try:
        engine.setup()
        
        if engine.memory_manager:
            memory_info = engine.memory_manager.get_memory_info()
            print(f"Memory info: {memory_info}")
            
            # Тест выгрузки моделей
            engine.memory_manager.unload_models(1000.0, keep_loaded=0)
            
            print("✓ Memory management test passed")
            return True
        else:
            print("✗ Memory manager not initialized")
            return False
            
    except Exception as e:
        print(f"✗ Memory management test failed: {e}")
        return False

def run_all_tests():
    """Запуск всех тестов"""
    print("Starting Lightweight Engine Tests...")
    print("=" * 50)
    
    tests = [
        test_basic_generation,
        test_memory_management,
        test_hires_fix,
        test_adetailer,
        test_lora_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
        print()
    
    print("=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)