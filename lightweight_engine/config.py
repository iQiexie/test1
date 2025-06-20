"""
Конфигурация на основе анализа логов генерации
"""
import os
import torch

class Config:
    # Пути к моделям (RTX 4090)
    MODELS_PATH = r"C:\stable-diffusion-webui-forge-main\models"
    FLUX_CHECKPOINT = "flux_dev.safetensors"  # Используем flux_dev.safetensors (23.8 GB)
    CLIP_L = "clip_l.safetensors"
    T5XXL = "t5xxl_fp16.safetensors"
    VAE = "ae.safetensors"
    
    # Полные пути
    FLUX_CHECKPOINT_PATH = os.path.join(MODELS_PATH, "Stable-diffusion", FLUX_CHECKPOINT)
    CLIP_L_PATH = os.path.join(MODELS_PATH, "text_encoder", CLIP_L)
    T5XXL_PATH = os.path.join(MODELS_PATH, "text_encoder", T5XXL)
    VAE_PATH = os.path.join(MODELS_PATH, "VAE", VAE)
    
    # Настройки генерации (из логов)
    DEFAULT_STEPS = 50
    DEFAULT_HR_STEPS = 10
    DEFAULT_CFG = 1.0
    DEFAULT_DISTILLED_CFG = 3.7
    DEFAULT_WIDTH = 768
    DEFAULT_HEIGHT = 1344
    DEFAULT_SAMPLER = "Euler"
    DEFAULT_SCHEDULER = "Simple"
    
    # Память (RTX 4090 - 24GB)
    VRAM_STATE = "HIGH_VRAM"
    TOTAL_VRAM = 24576  # MB для RTX 4090
    INFERENCE_MEMORY = 8192   # MB для RTX 4090
    MODEL_MEMORY = 16384  # MB для RTX 4090
    MEMORY_FRACTION = 0.95
    ENABLE_FLASH_ATTENTION = True
    ENABLE_MEMORY_PINNING = True
    
    # UNet настройки (из логов)
    UNET_STORAGE_DTYPE = "nf4"  # quantization
    VAE_DTYPE = torch.bfloat16  # из логов
    
    # ADetailer (из логов)
    ADETAILER_MODELS = {
        'face': 'face_yolov8s.pt',
        'hand': 'hand_yolov8s.pt',
        'person': 'person_yolov8s-seg.pt'
    }
    
    # Upscalers (из логов)
    UPSCALERS = {
        '4x-UltraSharp': '4x-UltraSharp.pth',
        'ESRGAN_4x': 'ESRGAN_4x.pth'
    }
    UPSCALERS_PATH = os.path.join(MODELS_PATH, "ESRGAN")
    
    # LoRA
    LORA_PATH = os.path.join(MODELS_PATH, "Lora")
    
    # ADetailer модели
    ADETAILER_PATH = os.path.join(MODELS_PATH, "adetailer")
    
    # Устройство
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Режимы работы
    ASYNC_MODE = True  # из логов: "Автоматически включен асинхронный режим"