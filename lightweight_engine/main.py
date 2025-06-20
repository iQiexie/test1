"""
Основной API файл для легкого движка
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64
import io
from PIL import Image

from config import Config
from core.engine import LightweightEngine

# Модели данных для API
class GenerationRequest(BaseModel):
    prompt: str
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    hr_scale: Optional[float] = 1.0
    hr_steps: Optional[int] = None
    hr_upscaler: Optional[str] = None
    lora_urls: Optional[List[str]] = None
    lora_scales: Optional[List[float]] = None
    adetailer: Optional[bool] = True
    adetailer_args: Optional[Dict[str, Any]] = None
    adetailer_args_hands: Optional[Dict[str, Any]] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = -1
    num_outputs: Optional[int] = 1

class GenerationResponse(BaseModel):
    images: List[str]  # base64 encoded images
    metadata: Dict[str, Any]

# Создание FastAPI приложения
app = FastAPI(
    title="Lightweight Stable Diffusion Engine",
    description="Оптимизированный движок на основе анализа логов",
    version="1.0.0"
)

# Глобальный экземпляр движка
engine = None

@app.on_event("startup")
async def startup():
    """Инициализация движка при запуске"""
    global engine
    print("Initializing Lightweight Engine...")
    
    config = Config()
    engine = LightweightEngine(config)
    
    try:
        engine.setup()
        print("Engine initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    """Очистка ресурсов при остановке"""
    global engine
    if engine and engine.memory_manager:
        engine.memory_manager.unload_models(0)
    print("Engine shutdown complete")

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Lightweight Stable Diffusion Engine",
        "version": "1.0.0",
        "status": "ready" if engine and engine.is_setup else "initializing"
    }

@app.get("/health")
async def health():
    """Проверка состояния"""
    if not engine or not engine.is_setup:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    memory_info = engine.memory_manager.get_memory_info() if engine.memory_manager else {}
    
    return {
        "status": "healthy",
        "engine_ready": engine.is_setup,
        "memory": memory_info
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """API как в оригинальном predict()"""
    if not engine or not engine.is_setup:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    try:
        # Генерация изображений
        images = engine.generate(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            hr_scale=request.hr_scale,
            hr_steps=request.hr_steps,
            hr_upscaler=request.hr_upscaler,
            lora_urls=request.lora_urls,
            lora_scales=request.lora_scales,
            adetailer=request.adetailer,
            adetailer_args=request.adetailer_args,
            adetailer_args_hands=request.adetailer_args_hands,
            sampler=request.sampler,
            scheduler=request.scheduler,
            seed=request.seed,
            num_outputs=request.num_outputs
        )
        
        # Конвертация изображений в base64
        encoded_images = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            encoded_images.append(img_str)
        
        # Метаданные
        metadata = {
            "prompt": request.prompt,
            "width": request.width or engine.config.DEFAULT_WIDTH,
            "height": request.height or engine.config.DEFAULT_HEIGHT,
            "steps": request.steps or engine.config.DEFAULT_STEPS,
            "cfg_scale": request.cfg_scale or engine.config.DEFAULT_CFG,
            "sampler": request.sampler or engine.config.DEFAULT_SAMPLER,
            "scheduler": request.scheduler or engine.config.DEFAULT_SCHEDULER,
            "seed": request.seed,
            "num_outputs": len(images)
        }
        
        return GenerationResponse(
            images=encoded_images,
            metadata=metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/config")
async def get_config():
    """Получение конфигурации"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    return {
        "default_steps": engine.config.DEFAULT_STEPS,
        "default_cfg": engine.config.DEFAULT_CFG,
        "default_width": engine.config.DEFAULT_WIDTH,
        "default_height": engine.config.DEFAULT_HEIGHT,
        "default_sampler": engine.config.DEFAULT_SAMPLER,
        "default_scheduler": engine.config.DEFAULT_SCHEDULER,
        "vram_state": engine.config.VRAM_STATE,
        "device": engine.config.DEVICE,
        "available_upscalers": list(engine.config.UPSCALERS.keys()) if engine.hires_fix else [],
        "adetailer_models": engine.config.ADETAILER_MODELS
    }

@app.get("/models/status")
async def models_status():
    """Статус загруженных моделей"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    return {
        "flux_model": engine.model is not None,
        "vae": engine.vae is not None,
        "text_encoders": len(engine.text_encoders),
        "loaded_loras": engine.lora_manager.get_loaded_loras() if engine.lora_manager else {}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)