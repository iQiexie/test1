# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os, sys, json
import shutil
import time
import subprocess  # Для запуска внешних процессов

sys.path.extend(["/stable-diffusion-webui-forge-main"])

from cog import BasePredictor, BaseModel, Input, Path
def download_base_weights(url: str, dest: Path):
    """
    Загружает базовые веса модели.
    
    Args:
        url: URL для загрузки весов
        dest: Путь для сохранения весов
    """
    start = time.time()  # Засекаем время начала загрузки
    print("downloading url: ", url)
    print("downloading to: ", dest)
    # Используем pget для эффективной загрузки файлов
    # Убираем параметр -xf, так как файл не является архивом
    subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)  # Выводим время загрузки

class Predictor(BasePredictor):
    def _move_model_to_sdwebui_dir(self):
        """
        Проверяет наличие модели и загружает ее, если она отсутствует.
        Модель должна быть предварительно загружена во время сборки в cog.yaml.
        """
        target_dir = "/stable-diffusion-webui-forge-main/models/Stable-diffusion"
        model_path = os.path.join(target_dir, "flux1DevHyperNF4Flux1DevBNB_flux1DevHyperNF4.safetensors")
        
        # Проверяем, существует ли уже файл модели
        if os.path.exists(model_path):
            print(f"Модель уже загружена!!: {model_path}")
            return
        
        # Если модель не найдена, загружаем ее
        print("Модель не найдена, загружаем...")
        os.makedirs(target_dir, exist_ok=True)
        download_base_weights(
            "https://civitai.com/api/download/models/819165?type=Model&format=SafeTensor&size=full&fp=nf4&token=18b51174c4d9ae0451a3dedce1946ce3",
            model_path
        )

    def _download_loras(self, lora_urls):
        """
        Загружает LoRA файлы по указанным URL.
        
        Args:
            lora_urls: Список URL для загрузки LoRA файлов
            
        Returns:
            Список путей к загруженным LoRA файлам
        """
        if not lora_urls or lora_urls.strip() == "":
            return []
            
        lora_urls_list = [url.strip() for url in lora_urls.split(",") if url.strip()]
        if not lora_urls_list:
            return []

        import os
        target_dir = "/stable-diffusion-webui-forge-main/models/Lora"
        os.makedirs(target_dir, exist_ok=True)
        
        lora_paths = []
        for i, url in enumerate(lora_urls_list):
            try:
                # Извлекаем имя файла из URL или используем индекс, если не удалось
                filename = os.path.basename(url.split("?")[0])
                if not filename or filename == "":
                    filename = f"lora_{i+1}.safetensors"
                
                # Убедимся, что файл имеет расширение .safetensors
                if not filename.endswith(".safetensors"):
                    filename += ".safetensors"
                
                lora_path = os.path.join(target_dir, filename)
                
                # Проверяем, существует ли уже файл
                if os.path.exists(lora_path):
                    print(f"ТОЧНО ПОСЛЕДНЯЯ ВЕРСИЯ LoRA файл уже существует: {lora_path}")
                    lora_paths.append(lora_path)
                else:
                    # Если файл не существует, загружаем его
                    download_base_weights(url, lora_path)
                    lora_paths.append(lora_path)
                    print(f"LoRA {i+1} успешно загружена: {lora_path}")
            except Exception as e:
                print(f"Ошибка при загрузке LoRA {i+1}: {e}")

        import os
        files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if
                 os.path.isfile(os.path.join(target_dir, f))]
        print(files)
        
        return lora_paths

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Загружаем модель Flux во время сборки, чтобы ускорить генерацию
        target_dir = "/stable-diffusion-webui-forge-main/models/Stable-diffusion"
        os.makedirs(target_dir, exist_ok=True)
        model_path = os.path.join(target_dir, "flux1DevHyperNF4Flux1DevBNB_flux1DevHyperNF4.safetensors")
        
        # Проверяем, существует ли уже файл модели
        if not os.path.exists(model_path):
            print(f"Загружаем модель Flux...")
            download_base_weights(
                "https://civitai.com/api/download/models/819165?type=Model&format=SafeTensor&size=full&fp=nf4&token=18b51174c4d9ae0451a3dedce1946ce3",
                model_path
            )
        else:
            print(f"Модель Flux уже загружена: {model_path}")

        # workaround for replicate since its entrypoint may contain invalid args
        os.environ["IGNORE_CMD_ARGS_ERRORS"] = "1"
        
        # Set the LoRA directory path
        lora_dir = "/stable-diffusion-webui-forge-main/models/Lora"  # You can change this path if needed
        os.makedirs(lora_dir, exist_ok=True)
        sys.argv.extend(["--lora-dir", lora_dir])
        
        # Ensure the LoRA extension is loaded
        sys.path.append("/stable-diffusion-webui-forge-main/extensions-builtin/sd_forge_lora")
        
        from modules import timer
        
        # Безопасный импорт memory_management
        try:
            from backend import memory_management
            self.has_memory_management = True
        except ImportError as e:
            print(f"Предупреждение: Не удалось импортировать memory_management: {e}")
            self.has_memory_management = False
        
        # moved env preparation to build time to reduce the warm-up time
        # from modules import launch_utils

        # with launch_utils.startup_timer.subcategory("prepare environment"):
        #     launch_utils.prepare_environment()

        from modules import initialize_util
        from modules import initialize

        startup_timer = timer.startup_timer
        startup_timer.record("launcher")

        initialize.imports()

        initialize.check_versions()

        initialize.initialize()
        
        # Импортируем shared после initialize.initialize()
        from modules import shared
        
        # Устанавливаем forge_preset на 'flux'
        shared.opts.set('forge_preset', 'flux')
        
        # Устанавливаем чекпоинт
        shared.opts.set('sd_model_checkpoint', 'flux1DevHyperNF4Flux1DevBNB_flux1DevHyperNF4.safetensors')

        # Устанавливаем unet тип на 'Automatic (fp16 LoRA)' для Flux, чтобы LoRA работали правильно
        shared.opts.set('forge_unet_storage_dtype', 'bnb-nf4')
        
        # Оптимизация памяти для лучшего качества и скорости с Flux
        if self.has_memory_management:
            # Выделяем больше памяти для загрузки весов модели (90% для весов, 10% для вычислений)
            total_vram = memory_management.total_vram
            inference_memory = int(total_vram * 0.1)  # 10% для вычислений
            model_memory = total_vram - inference_memory
            
            memory_management.current_inference_memory = inference_memory * 1024 * 1024  # Конвертация в байты
            print(f"[GPU Setting] Выделено {model_memory} MB для весов модели и {inference_memory} MB для вычислений")
            
            # Настройка Swap Method на ASYNC для лучшей производительности
            try:
                from backend import stream
                # Для Flux рекомендуется ASYNC метод, который может быть до 30% быстрее
                stream.stream_activated = True  # True = ASYNC, False = Queue
                print("[GPU Setting] Установлен ASYNC метод загрузки для лучшей производительности")
                
                # Настройка Swap Location на Shared для лучшей производительности
                memory_management.PIN_SHARED_MEMORY = True  # True = Shared, False = CPU
                print("[GPU Setting] Установлен Shared метод хранения для лучшей производительности")
            except ImportError as e:
                print(f"Предупреждение: Не удалось импортировать stream: {e}")
        else:
            print("[GPU Setting] memory_management не доступен, используются настройки по умолчанию")

        from fastapi import FastAPI

        app = FastAPI()
        initialize_util.setup_middleware(app)

        from modules.api.api import Api
        from modules.call_queue import queue_lock
        
        # Create a custom API class that patches the script handling functions
        class CustomApi(Api):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # Patch the get_script function to handle LoRA scripts
                original_get_script = self.get_script
                
                def patched_get_script(script_name, script_runner):
                    try:
                        return original_get_script(script_name, script_runner)
                    except Exception as e:
                        # If the script is not found and it's the LoRA script, handle it specially
                        if script_name in ["lora", "sd_forge_lora"]:
                            print(f"LoRA script '{script_name}' not found in standard scripts, using extra_network_data instead")
                            return None
                        raise e
                
                self.get_script = patched_get_script
                
                # Patch the init_script_args function to handle missing scripts
                original_init_script_args = self.init_script_args
                
                def patched_init_script_args(request, default_script_args, selectable_scripts, selectable_idx, script_runner, *, input_script_args=None):
                    try:
                        return original_init_script_args(request, default_script_args, selectable_scripts, selectable_idx, script_runner, input_script_args=input_script_args)
                    except Exception as e:
                        # If there's an error with alwayson_scripts, try to continue without them
                        if hasattr(request, 'alwayson_scripts') and request.alwayson_scripts:
                            print(f"Error initializing alwayson_scripts: {e}")
                            # Remove problematic scripts
                            for script_name in list(request.alwayson_scripts.keys()):
                                if script_name in ["lora", "sd_forge_lora"]:
                                    print(f"Removing problematic script: {script_name}")
                                    del request.alwayson_scripts[script_name]
                            
                            # Try again without the problematic scripts
                            if not request.alwayson_scripts:
                                request.alwayson_scripts = None
                            
                            return original_init_script_args(request, default_script_args, selectable_scripts, selectable_idx, script_runner, input_script_args=input_script_args)
                        raise e
                
                self.init_script_args = patched_init_script_args
        
        self.api = CustomApi(app, queue_lock)

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        negative_prompt: str = Input(
            description="Negative Prompt (для Flux рекомендуется оставить пустым и использовать Distilled CFG)",
            default="",
        ),
        width: int = Input(
            description="Width of output image", ge=1, le=1280, default=768
        ),
        height: int = Input(
            description="Height of output image", ge=1, le=1280, default=1280
        ),
        num_outputs: int = Input(
            description="Number of images to output", ge=1, le=4, default=1
        ),
        sampler: str = Input(
            description="Sampling method для Flux моделей",
            choices=[
                "[Forge] Flux Realistic",
                "Euler",
                "Euler a",
                "DPM++ 2M",
                "DPM++ SDE",
                "DPM++ 2M SDE",
                "DPM++ 2M SDE Karras",
                "DPM++ 2M SDE Exponential",
                "DPM++ 3M SDE",
                "DPM++ 3M SDE Karras",
                "DPM++ 3M SDE Exponential"
            ],
            default="[Forge] Flux Realistic",
        ),
        scheduler: str = Input(
            description="Schedule type для Flux моделей",
            choices=[
                "Simple",
                "Karras",
                "Exponential",
                "SGM Uniform",
                "SGM Karras",
                "SGM Exponential",
                "Align Your Steps",
                "Align Your Steps 11",
                "Align Your Steps 32",
                "Align Your Steps GITS",
                "KL Optimal",
                "Normal",
                "DDIM",
                "Beta",
                "Turbo"
            ],
            default="Simple",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=50, default=15
        ),
        guidance_scale: float = Input(
            description="CFG Scale (для Flux рекомендуется значение 1.0)", ge=1, le=50, default=1.0
        ),
        distilled_guidance_scale: float = Input(
            description="Distilled CFG Scale (основной параметр для Flux, рекомендуется 3.5)", ge=0, le=30, default=3.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=-1
        ),
        # image: Path = Input(description="Grayscale input image"),
        enable_hr: bool = Input(
            description="Hires. fix",
            default=False,
        ),
        hr_upscaler: str = Input(
            description="Upscaler for Hires. fix",
            choices=[
                "Latent",
                "Latent (antialiased)",
                "Latent (bicubic)",
                "Latent (bicubic antialiased)",
                "Latent (nearest)",
                "Latent (nearest-exact)",
                "None",
                "Lanczos",
                "Nearest",
                "ESRGAN_4x",
                "LDSR",
                "R-ESRGAN 4x+",
                "R-ESRGAN 4x+ Anime6B",
                "ScuNET GAN",
                "ScuNET PSNR",
                "SwinIR 4x",
            ],
            default="Latent",
        ),
        hr_steps: int = Input(
            description="Inference steps for Hires. fix", ge=0, le=100, default=20
        ),
        hr_scale: float = Input(
            description="Factor to scale image by", ge=1, le=4, default=2
        ),
        denoising_strength: float = Input(
            description="Denoising strength. 1.0 corresponds to full destruction of information in init image",
            ge=0,
            le=1,
            default=0.5,
        ),
        enable_adetailer: bool = Input(
            description="ADetailer (не рекомендуется для Flux моделей)",
            default=False,
        ),
        lora_urls: str = Input(
            description="Ссылки на LoRA файлы, разделенные запятыми (например, https://example.com/lora1.safetensors,https://example.com/lora2.safetensors)",
            default="",
        ),
    ) -> list[Path]:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        # Загружаем и применяем LoRA файлы, если они указаны

        if lora_urls and lora_urls.strip():
            self._download_loras(lora_urls)

        from modules.extra_networks import ExtraNetworkParams

        payload = {
            # "init_images": [encoded_image],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": num_outputs,
            "steps": num_inference_steps,
            "cfg_scale": guidance_scale,
            "seed": seed,
            "do_not_save_samples": True,
            "sampler_name": sampler,  # Используем выбранный пользователем sampler
            "scheduler": scheduler,    # Устанавливаем scheduler для Flux
            "enable_hr": enable_hr,
            "hr_upscaler": hr_upscaler,
            "hr_second_pass_steps": hr_steps,
            "denoising_strength": denoising_strength if enable_hr else None,
            "hr_scale": hr_scale,
            "distilled_cfg_scale": distilled_guidance_scale,  # Добавляем параметр distilled_cfg_scale для Flux
            "hr_additional_modules": [],  # Добавляем пустой список для hr_additional_modules, чтобы избежать ошибки
        }
        
        # Нет необходимости добавлять их в payload отдельно

        alwayson_scripts = {}

        # Add LoRA to both alwayson_scripts and extra_network_data
        if lora_urls and lora_urls.strip():
            # Get the LoRA filenames without path and extension
            lora_files = []
            target_dir = "/stable-diffusion-webui-forge-main/models/Lora"
            for file in os.listdir(target_dir):
                if file.endswith(".safetensors"):
                    lora_name = os.path.splitext(file)[0]
                    lora_files.append(lora_name)
            
            if lora_files:
                # Format the LoRA arguments properly for the script
                lora_args = []
                for lora_name in lora_files:
                    # Each LoRA needs a name and a weight
                    lora_args.append(lora_name)  # Name
                    lora_args.append(1.0)        # Weight (default to 1.0)
                
                # Use the correct script name: "lora" instead of "sd_forge_lora"
                alwayson_scripts["lora"] = {
                    "args": lora_args
                }
                print(f"Added LoRA files to alwayson_scripts: {lora_args}")

        if enable_adetailer:
            alwayson_scripts["ADetailer"] = {
                "args": [
                    {
                        "ad_model": "face_yolov8n.pt",
                    }
                ],
            }

        # Добавляем все скрипты в payload, если они есть
        if alwayson_scripts:
            payload["alwayson_scripts"] = alwayson_scripts

        from modules.api.models import (
            StableDiffusionTxt2ImgProcessingAPI,
        )

        print(f"Финальный пейлоад: {payload=}")
        req = StableDiffusionTxt2ImgProcessingAPI(**payload)
        # generate
        # Use both extra_network_data and alwayson_scripts to handle LoRA models
        extra_network_data = {"lora": []}
        
        # Add all LoRA files with their weights to extra_network_data
        if lora_files:
            for lora_name in lora_files:
                extra_network_data["lora"].append(ExtraNetworkParams(items=[lora_name, "1"]))
                print(f"LOOK HEREitems=[{lora_name}, '1']")
        
        # Import the necessary modules for script registration
        from modules import scripts
        
        # Make sure the LoRA script is initialized
        if not hasattr(scripts.scripts_txt2img, 'selectable_scripts'):
            scripts.scripts_txt2img.initialize_scripts(False)
        
        # Print available scripts for debugging
        print("Available scripts:", [script.title().lower() for script in scripts.scripts_txt2img.scripts])
        
        # Ensure the model is properly loaded before using LoRA
        from modules import sd_models, shared
        
        # Check if the model is a FakeInitialModel and needs to be loaded
        if isinstance(shared.sd_model, sd_models.FakeInitialModel):
            print("Model is FakeInitialModel, loading Flux model...")
            
            # Set the checkpoint to the Flux model specifically
            flux_checkpoint_name = "flux1DevHyperNF4Flux1DevBNB_flux1DevHyperNF4.safetensors"
            shared.opts.set('sd_model_checkpoint', flux_checkpoint_name)
            shared.opts.set('forge_preset', 'flux')
            # Don't set forge_unet_storage_dtype directly, instead set it in the forge_loading_parameters
            
            # Find the Flux checkpoint
            flux_checkpoint = None
            for checkpoint in sd_models.checkpoints_list.values():
                if checkpoint.filename.endswith(flux_checkpoint_name):
                    flux_checkpoint = checkpoint
                    break
            
            if flux_checkpoint is not None:
                # Set up forge loading parameters - don't use string for dtype
                sd_models.model_data.forge_loading_parameters = {
                    'checkpoint_info': flux_checkpoint,
                    'additional_modules': []
                }
                
                # Set the dynamic args directly instead of using the string
                from backend.args import dynamic_args
                dynamic_args['forge_unet_storage_dtype'] = None  # Let the loader determine the best dtype
                # Load the model
                sd_models.forge_model_reload()
                print(f"Flux model loaded: {type(shared.sd_model)}")
            else:
                print(f"Warning: Could not find Flux checkpoint {flux_checkpoint_name}")
        
        # Directly use the ExtraNetworkParams for LoRA without relying on alwayson_scripts
        # This is the most reliable way to use LoRA with the API
        if hasattr(req, 'alwayson_scripts') and req.alwayson_scripts:
            # Remove the LoRA script from alwayson_scripts to avoid conflicts
            if 'lora' in req.alwayson_scripts:
                del req.alwayson_scripts['lora']
            if 'sd_forge_lora' in req.alwayson_scripts:
                del req.alwayson_scripts['sd_forge_lora']
            
            # If alwayson_scripts is empty, remove it
            if not req.alwayson_scripts:
                req.alwayson_scripts = None
        
        # Now check if the model has forge_objects attribute
        if hasattr(shared.sd_model, 'forge_objects'):
            print("Model has forge_objects, proceeding with LoRA...")
            resp = self.api.text2imgapi(
                txt2imgreq=req,
                extra_network_data=extra_network_data
            )
        else:
            print("Warning: Model does not have forge_objects attribute, proceeding without LoRA")
            # Remove LoRA from extra_network_data to avoid errors
            resp = self.api.text2imgapi(
                txt2imgreq=req,
                extra_network_data=None
            )
        info = json.loads(resp.info)

        from PIL import Image
        import uuid
        import base64
        from io import BytesIO

        outputs = []

        for i, image in enumerate(resp.images):
            seed = info["all_seeds"][i]
            gen_bytes = BytesIO(base64.b64decode(image))
            gen_data = Image.open(gen_bytes)
            filename = "{}-{}.png".format(seed, uuid.uuid1())
            gen_data.save(fp=filename, format="PNG")
            output = Path(filename)
            outputs.append(output)

        return outputs
