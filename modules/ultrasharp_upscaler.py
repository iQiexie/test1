"""
Регистрация 4x-UltraSharp апскейлера для WebUI/Forge
"""
import os


class UpscalerUltraSharp:
    def __init__(self, dirname):
        self.name = "4x-UltraSharp"
        self.model_path = os.path.join(dirname, "4x-UltraSharp.pth")
        self.user_path = dirname
        self.scalers = []
        
        # Проверяем наличие модели
        if os.path.exists(self.model_path):
            try:
                # Импортируем внутри функции чтобы избежать циклических импортов
                from modules.upscaler import UpscalerData
                
                model_data = UpscalerData(
                    name="4x-UltraSharp",
                    path=self.model_path,
                    upscaler=self,
                    scale=4
                )
                self.scalers = [model_data]
                print(f"[UltraSharp] Зарегистрирован апскейлер: {self.model_path}")
            except ImportError as e:
                print(f"[UltraSharp] Ошибка импорта UpscalerData: {e}")
                self.scalers = []
        else:
            print(f"[UltraSharp] Модель не найдена: {self.model_path}")

    def do_upscale(self, img, selected_model):
        """Выполняет апскейлинг изображения"""
        try:
            from modules.upscaler_utils import upscale_with_model
            from modules import shared
            
            return upscale_with_model(
                selected_model,
                img,
                ESRGAN_tile=getattr(shared.opts, 'ESRGAN_tile', 192),
                ESRGAN_tile_overlap=getattr(shared.opts, 'ESRGAN_tile_overlap', 8),
                upscaler_for_img=self
            )
        except Exception as e:
            print(f"[UltraSharp] Ошибка апскейлинга: {e}")
            # Fallback на стандартный ESRGAN
            try:
                from modules.upscaler_esrgan import UpscalerESRGAN
                fallback = UpscalerESRGAN("/src/models/ESRGAN")
                if fallback.scalers:
                    return fallback.do_upscale(img, fallback.scalers[0])
            except Exception as e2:
                print(f"[UltraSharp] Fallback тоже не сработал: {e2}")
            return img

    def load_model(self, path):
        """Загружает модель апскейлера"""
        try:
            import torch
            return torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[UltraSharp] Ошибка загрузки модели: {e}")
            return None


def register_ultrasharp_upscaler():
    """Регистрирует 4x-UltraSharp апскейлер в системе"""
    try:
        # Импортируем shared внутри функции
        from modules import shared
        
        # Путь к директории с ESRGAN моделями
        esrgan_dir = "/src/models/ESRGAN"
        
        if os.path.exists(esrgan_dir):
            upscaler = UpscalerUltraSharp(esrgan_dir)
            
            # Добавляем в список доступных апскейлеров
            if hasattr(shared, 'sd_upscalers'):
                # Проверяем, не добавлен ли уже
                existing_names = [getattr(u, 'name', '') for u in shared.sd_upscalers]
                if "4x-UltraSharp" not in existing_names:
                    shared.sd_upscalers.append(upscaler)
                    print(f"[UltraSharp] Апскейлер добавлен в shared.sd_upscalers")
                else:
                    print(f"[UltraSharp] Апскейлер уже зарегистрирован")
            else:
                print(f"[UltraSharp] shared.sd_upscalers не найден")
                
        else:
            print(f"[UltraSharp] Директория не найдена: {esrgan_dir}")
            
    except Exception as e:
        print(f"[UltraSharp] Ошибка регистрации: {e}")
        import traceback
        traceback.print_exc()