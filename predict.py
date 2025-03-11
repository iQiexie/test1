import os, sys, json
import time
import subprocess
from datetime import datetime
sys.path.extend(["/stable-diffusion-webui-forge-main"])

from cog import BasePredictor, Input, Path

FLUX_CHECKPOINT_URL = "https://civitai.com/api/download/models/691639?type=Model&format=SafeTensor&size=full&fp=fp32&token=18b51174c4d9ae0451a3dedce1946ce3"


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
    def _move_model_to_sdwebui_dir(self, checkpoint_url):
        """
        Проверяет наличие модели и загружает ее, если она отсутствует.
        Модель должна быть предварительно загружена во время сборки в cog.yaml.
        """
        target_dir = "/stable-diffusion-webui-forge-main/models/Stable-diffusion"
        model_path = os.path.join(target_dir, "flux_checkpoint.safetensors")

        # Проверяем, существует ли уже файл модели
        if os.path.exists(model_path):
            print(f"Модель уже загружена!!: {model_path}")
            return

        # Если модель не найдена, загружаем ее
        print("Модель не найдена, загружаем...")
        os.makedirs(target_dir, exist_ok=True)
        download_base_weights(
            checkpoint_url,
            model_path
        )

    def _download_loras(self, lora_urls: list[str]):
        """
        Загружает LoRA файлы по указанным URL.
        
        Args:
            lora_urls: Список URL для загрузки LoRA файлов
            
        Returns:
            Список путей к загруженным LoRA файлам
        """

        import os
        import tarfile
        import tempfile
        import shutil
        import re

        target_dir = "/stable-diffusion-webui-forge-main/models/Lora"
        os.makedirs(target_dir, exist_ok=True)

        lora_paths = []
        for i, url in enumerate(lora_urls):
            try:
                # Проверяем, является ли URL ссылкой на .tar архив
                is_tar_archive = url.endswith('.tar') or '/trained_model.tar' in url

                if is_tar_archive:
                    # Извлекаем ID из URL для .tar архивов (например, из replicate.delivery)
                    # Пример: https://replicate.delivery/xezq/h1097z5f1FXycCDn31BYqb4fi1o5nfqExf0ZmozqArxFobaRB/trained_model.tar
                    match = re.search(r'/([a-zA-Z0-9]{40,})/trained_model\.tar', url)
                    if match:
                        lora_id = match.group(1)
                        filename = f"{lora_id}.safetensors"
                    else:
                        # Если не удалось извлечь ID, используем индекс
                        filename = f"lora_tar_{i + 1}.safetensors"

                    lora_path = os.path.join(target_dir, filename)

                    # Проверяем, существует ли уже файл
                    if os.path.exists(lora_path):
                        print(f"LoRA файл уже существует: {lora_path}")
                        lora_paths.append(lora_path)
                    else:
                        # Создаем временную директорию для распаковки архива
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Загружаем .tar архив во временный файл
                            tar_path = os.path.join(temp_dir, "archive.tar")
                            download_base_weights(url, tar_path)

                            # Распаковываем архив
                            with tarfile.open(tar_path) as tar:
                                tar.extractall(path=temp_dir)

                            # Ищем lora.safetensors в распакованной структуре
                            # Ожидаемый путь: output/flux_train_replicate/lora.safetensors
                            lora_file_path = None

                            # Проверяем наличие ожидаемой структуры
                            expected_path = os.path.join(
                                temp_dir, "output", "flux_train_replicate", "lora.safetensors"
                                )
                            if os.path.exists(expected_path):
                                lora_file_path = expected_path
                            else:
                                # Если ожидаемой структуры нет, ищем .safetensors файл рекурсивно
                                for root, _, files in os.walk(temp_dir):
                                    for file in files:
                                        if file.endswith('.safetensors'):
                                            lora_file_path = os.path.join(root, file)
                                            break
                                    if lora_file_path:
                                        break

                            if lora_file_path:
                                # Копируем найденный файл в целевую директорию с нужным именем
                                shutil.copy2(lora_file_path, lora_path)
                                lora_paths.append(lora_path)
                                print(f"LoRA {i + 1} успешно извлечена из архива и сохранена: {lora_path}")
                            else:
                                print(f"Ошибка: не удалось найти .safetensors файл в архиве {url}")
                else:
                    # Стандартная обработка для не-tar файлов
                    # Извлекаем имя файла из URL или используем индекс, если не удалось
                    filename = os.path.basename(url.split("?")[0])
                    if not filename or filename == "":
                        filename = f"lora_{i + 1}.safetensors"

                    # Убедимся, что файл имеет расширение .safetensors
                    if not filename.endswith(".safetensors"):
                        filename += ".safetensors"

                    lora_path = os.path.join(target_dir, filename)

                    # Проверяем, существует ли уже файл
                    if os.path.exists(lora_path):
                        print(f"LoRA файл уже существует: {lora_path}")
                        lora_paths.append(lora_path)
                    else:
                        # Если файл не существует, загружаем его
                        download_base_weights(url, lora_path)
                        lora_paths.append(lora_path)
                        print(f"LoRA {i + 1} успешно загружена: {lora_path}")
            except Exception as e:
                print(f"Ошибка при загрузке LoRA {i + 1}: {e}")

        import os
        files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if
                 os.path.isfile(os.path.join(target_dir, f))]
        print(files)

        return lora_paths

    def download_models(self):
        target_dir = "/stable-diffusion-webui-forge-main/models/text_encoder"
        os.makedirs(target_dir, exist_ok=True)
        sys.argv.extend(["--text-encoder-dir", target_dir])

        print("Downloading: clip_l")
        download_base_weights(
            "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true",
            os.path.join(target_dir, "clip_l.safetensors"),
        )

        print("Downloading: t5xxl_fp16")
        download_base_weights(
            "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors?download=true",
            os.path.join(target_dir, "t5xxl_fp16.safetensors"),
        )

        target_dir = "/stable-diffusion-webui-forge-main/models/VAE"
        os.makedirs(target_dir, exist_ok=True)
        sys.argv.extend(["--vae-dir", target_dir])

        print("Downloading: ae")
        download_base_weights(
            "https://ai-photo.fra1.cdn.digitaloceanspaces.com/ae.safetensors",
            os.path.join(target_dir, "ae.safetensors"),
        )

    def setup(self, checkpoint_url: str = None, force: bool = False) -> None:
        if not checkpoint_url:
            checkpoint_url = FLUX_CHECKPOINT_URL
        print("Starting setup...")

        # Download text encoders and VAE if they don't exist
        text_encoder_dir = "/stable-diffusion-webui-forge-main/models/text_encoder"
        vae_dir = "/stable-diffusion-webui-forge-main/models/VAE"

        # Check if files exist and download if needed
        if not os.path.exists(os.path.join(text_encoder_dir, "clip_l.safetensors")) or \
            not os.path.exists(os.path.join(text_encoder_dir, "t5xxl_fp16.safetensors")) or \
            not os.path.exists(os.path.join(vae_dir, "ae.safetensors")):
            print("Downloading required model components...")
            self.download_models()
        else:
            print("All required model components already exist.")
        """Load the model into memory to make running multiple predictions efficient"""
        # Загружаем модель Flux во время сборки, чтобы ускорить генерацию
        target_dir = "/stable-diffusion-webui-forge-main/models/Stable-diffusion"
        os.makedirs(target_dir, exist_ok=True)
        model_path = os.path.join(target_dir, "flux_checkpoint.safetensors")

        # Проверяем, существует ли уже файл модели
        if not os.path.exists(model_path) and force is False:
            print(f"Загружаем модель Flux...")
            download_base_weights(
                checkpoint_url,
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

        # Оптимизация памяти для лучшего качества и скорости с Flux
        if self.has_memory_management:
            # Выделяем больше памяти для загрузки весов модели (90% для весов, 10% для вычислений)
            total_vram = memory_management.total_vram
            inference_memory = int(total_vram * 0.1)  # 10% для вычислений
            model_memory = total_vram - inference_memory

            memory_management.current_inference_memory = inference_memory * 1024 * 1024  # Конвертация в байты
            print(
                f"[GPU Setting] Выделено {model_memory} MB для весов модели и {inference_memory} MB для вычислений"
                )

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
                            print(
                                f"LoRA script '{script_name}' not found in standard scripts, using extra_network_data instead"
                                )
                            return None
                        raise e

                self.get_script = patched_get_script

                # Patch the init_script_args function to handle missing scripts
                original_init_script_args = self.init_script_args

                def patched_init_script_args(
                    request, default_script_args, selectable_scripts, selectable_idx, script_runner, *,
                    input_script_args=None
                    ):
                    try:
                        return original_init_script_args(
                            request, default_script_args, selectable_scripts, selectable_idx, script_runner,
                            input_script_args=input_script_args
                            )
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

                            return original_init_script_args(
                                request, default_script_args, selectable_scripts, selectable_idx,
                                script_runner, input_script_args=input_script_args
                                )
                        raise e

                self.init_script_args = patched_init_script_args

        self.api = CustomApi(app, queue_lock)

    @staticmethod
    def load_model() -> None:
        from modules import sd_models, shared

        print("Loading Flux model...")

        # Set the checkpoint to the Flux model specifically
        flux_checkpoint_name = "flux_checkpoint.safetensors"
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
            # Find the text encoder and VAE files
            text_encoder_dir = "/stable-diffusion-webui-forge-main/models/text_encoder"
            vae_dir = "/stable-diffusion-webui-forge-main/models/VAE"

            clip_path = os.path.join(text_encoder_dir, "clip_l.safetensors")
            t5xxl_path = os.path.join(text_encoder_dir, "t5xxl_fp16.safetensors")
            vae_path = os.path.join(vae_dir, "ae.safetensors")

            # Print directory contents for debugging
            print(
                f"DEBUG: Text encoder directory contents: {os.listdir(text_encoder_dir) if os.path.exists(text_encoder_dir) else 'directory not found'}"
                )
            print(
                f"DEBUG: VAE directory contents: {os.listdir(vae_dir) if os.path.exists(vae_dir) else 'directory not found'}"
            )

            # Check if files exist
            additional_modules = []
            if os.path.exists(clip_path):
                file_size = os.path.getsize(clip_path) / (1024 * 1024)  # Size in MB
                additional_modules.append(clip_path)
                print(f"Adding CLIP text encoder: {clip_path} (Size: {file_size:.2f} MB)")
            else:
                print(f"WARNING: CLIP text encoder not found at {clip_path}")
                # Try to download it if missing
                print("Attempting to download missing CLIP text encoder...")
                os.makedirs(text_encoder_dir, exist_ok=True)
                download_base_weights(
                    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true",
                    clip_path
                )
                if os.path.exists(clip_path):
                    file_size = os.path.getsize(clip_path) / (1024 * 1024)
                    additional_modules.append(clip_path)
                    print(
                        f"Successfully downloaded CLIP text encoder: {clip_path} (Size: {file_size:.2f} MB)"
                    )

            if os.path.exists(t5xxl_path):
                file_size = os.path.getsize(t5xxl_path) / (1024 * 1024)
                additional_modules.append(t5xxl_path)
                print(f"Adding T5XXL text encoder: {t5xxl_path} (Size: {file_size:.2f} MB)")
            else:
                print(f"WARNING: T5XXL text encoder not found at {t5xxl_path}")
                # Try to download it if missing
                print("Attempting to download missing T5XXL text encoder...")
                os.makedirs(text_encoder_dir, exist_ok=True)
                download_base_weights(
                    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors?download=true",
                    t5xxl_path
                )
                if os.path.exists(t5xxl_path):
                    file_size = os.path.getsize(t5xxl_path) / (1024 * 1024)
                    additional_modules.append(t5xxl_path)
                    print(
                        f"Successfully downloaded T5XXL text encoder: {t5xxl_path} (Size: {file_size:.2f} MB)"
                    )

            if os.path.exists(vae_path):
                file_size = os.path.getsize(vae_path) / (1024 * 1024)
                additional_modules.append(vae_path)
                print(f"Adding VAE: {vae_path} (Size: {file_size:.2f} MB)")
            else:
                print(f"WARNING: VAE not found at {vae_path}")
                # Try to download it if missing
                print("Attempting to download missing VAE...")
                os.makedirs(vae_dir, exist_ok=True)
                download_base_weights(
                    "https://ai-photo.fra1.cdn.digitaloceanspaces.com/ae.safetensors",
                    vae_path
                )
                if os.path.exists(vae_path):
                    file_size = os.path.getsize(vae_path) / (1024 * 1024)
                    additional_modules.append(vae_path)
                    print(f"Successfully downloaded VAE: {vae_path} (Size: {file_size:.2f} MB)")

            print(f"DEBUG: Total additional modules to load: {len(additional_modules)}")

            # Set up forge loading parameters - don't use string for dtype
            sd_models.model_data.forge_loading_parameters = {
                'checkpoint_info': flux_checkpoint,
                'additional_modules': additional_modules
            }

            # Set the dynamic args directly instead of using the string
            from backend.args import dynamic_args
            dynamic_args['forge_unet_storage_dtype'] = None  # Let the loader determine the best dtype

            # Add debug info for state dictionaries
            print(f"DEBUG: About to load model with {len(additional_modules)} additional modules")
            for i, module_path in enumerate(additional_modules):
                try:
                    from backend.utils import load_torch_file
                    state_dict = load_torch_file(module_path)
                    if isinstance(state_dict, dict):
                        print(f"DEBUG: Module {i + 1}: {module_path} - State dict has {len(state_dict)} keys")
                        # Print some key names for debugging
                        keys_sample = list(state_dict.keys())[:5]
                        print(f"DEBUG: Sample keys: {keys_sample}")
                    else:
                        print(
                            f"DEBUG: Module {i + 1}: {module_path} - State dict is not a dictionary, type: {type(state_dict)}"
                        )
                except Exception as e:
                    print(f"DEBUG: Error inspecting module {module_path}: {str(e)}")

            # Load the model
            try:
                sd_models.forge_model_reload()
                print("DEBUG: Model loaded successfully")
            except Exception as e:
                print(f"ERROR: Failed to load model: {str(e)}")
                import traceback
                traceback.print_exc()
            print(f"Flux model loaded: {type(shared.sd_model)}")
        else:
            print(f"Warning: Could not find Flux checkpoint {flux_checkpoint_name}")

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
            description="Distilled CFG Scale (основной параметр для Flux, рекомендуется 3.5)", ge=0, le=30,
            default=3.5
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
        forge_unet_storage_dtype: str = Input(
            description="forge_unet_storage_dtype",
            choices=[
                'Automatic',
                'Automatic (fp16 LoRA)',
                'bnb-nf4',
                'bnb-nf4 (fp16 LoRA)',
                'float8-e4m3fn',
                'float8-e4m3fn (fp16 LoRA)',
                'bnb-fp4',
                'bnb-fp4 (fp16 LoRA)',
                'float8-e5m2',
                'float8-e5m2 (fp16 LoRA)',
            ],
            default="Automatic",
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
        lora_urls: list[str] = Input(
            description="Ссылки на LoRA файлы, разделенные запятыми (например, https://example.com/lora1.safetensors,https://example.com/lora2.safetensors)",
            default=[],
        ),
        lora_scales: list[float] = Input(
            description="Lora scales",
            default=[0.8],
        ),
        flux_checkpoint_url: str = Input(
            description="Flux checkpoint URL",
            default=""
        ),

    ) -> list[Path]:
        start = datetime.now()
        self.setup(
            checkpoint_url=flux_checkpoint_url or FLUX_CHECKPOINT_URL,
            force=bool(flux_checkpoint_url),
        )
        print(f"[Time] setup: {(datetime.now() - start).total_seconds()}")
        start = datetime.now()

        # Set up directories for text encoder and VAE
        text_encoder_dir = "/stable-diffusion-webui-forge-main/models/text_encoder"
        vae_dir = "/stable-diffusion-webui-forge-main/models/VAE"

        # Make sure directories exist
        os.makedirs(text_encoder_dir, exist_ok=True)
        os.makedirs(vae_dir, exist_ok=True)

        # Remove any existing arguments to avoid duplicates
        for i in range(len(sys.argv) - 1, -1, -1):
            if sys.argv[i] in ["--text-encoder-dir", "--vae-dir"]:
                if i + 1 < len(sys.argv):
                    sys.argv.pop(i + 1)
                sys.argv.pop(i)

        # Add the arguments
        sys.argv.extend(["--text-encoder-dir", text_encoder_dir])
        sys.argv.extend(["--vae-dir", vae_dir])

        # Set environment variables as well for extra safety
        os.environ["FORGE_TEXT_ENCODER_DIR"] = text_encoder_dir
        os.environ["FORGE_VAE_DIR"] = vae_dir

        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        # Загружаем и применяем LoRA файлы, если они указаны

        # Импортируем shared после initialize.initialize()
        from modules import shared

        # Устанавливаем forge_preset на 'flux'
        shared.opts.set('forge_preset', 'flux')

        # Устанавливаем чекпоинт
        shared.opts.set('sd_model_checkpoint', 'flux_checkpoint.safetensors')

        # Устанавливаем unet тип на 'Automatic (fp16 LoRA)' для Flux, чтобы LoRA работали правильно
        shared.opts.set('forge_unet_storage_dtype', forge_unet_storage_dtype)

        print(f"[Time] defined variables: {(datetime.now() - start).total_seconds()}")
        start = datetime.now()

        if lora_urls:
            self._download_loras(lora_urls)

        print(f"[Time] downloaded loras: {(datetime.now() - start).total_seconds()}")
        start = datetime.now()
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
            "scheduler": scheduler,  # Устанавливаем scheduler для Flux
            "enable_hr": enable_hr,
            "hr_upscaler": hr_upscaler,
            "hr_second_pass_steps": hr_steps,
            "denoising_strength": denoising_strength if enable_hr else None,
            "hr_scale": hr_scale,
            "distilled_cfg_scale": distilled_guidance_scale,
            # Добавляем параметр distilled_cfg_scale для Flux
            "hr_additional_modules": [],
            # Добавляем пустой список для hr_additional_modules, чтобы избежать ошибки
        }

        # Initialize empty alwayson_scripts
        alwayson_scripts = {}

        lora_files = []
        target_dir = "/stable-diffusion-webui-forge-main/models/Lora"
        for file in os.listdir(target_dir):
            if file.endswith(".safetensors"):
                lora_name = os.path.splitext(file)[0]
                lora_files.append(lora_name)

        # Create a proper script_args dictionary for each extension
        # This ensures each extension gets only the arguments it expects
        
        # NeverOOM expects 2 arguments
        neveroom_args = [False, False]  # unet_enabled, vae_enabled
        
        # FreeU expects 7 arguments
        freeu_args = [False, 1.01, 1.02, 0.99, 0.95, 0.0, 1.0]  # enabled, b1, b2, s1, s2, start, end
        
        # Kohya HRFix expects 8 arguments
        kohya_hrfix_args = [False, 3, 2.0, 0.0, 0.35, True, "bicubic", "bicubic"]  # enabled, block_number, etc.
        
        # Latent Modifier expects 21 arguments (simplified here)
        latent_modifier_args = [False] + [0.0] * 20  # enabled + 20 other parameters with default values
        
        # MultiDiffusion expects 6 arguments
        multidiffusion_args = [False, "MultiDiffusion", 768, 768, 64, 4]  # enabled, method, etc.
        
        # Perturbed Attention expects 5 arguments
        perturbed_attention_args = [False, 1.0, 0.0, 0.0, 1.0]  # enabled, scale, etc.
        
        # SAG expects 4 arguments
        sag_args = [False, 0.5, 2.0, 1.0]  # enabled, scale, blur_sigma, threshold
        
        # StyleAlign expects 2 arguments
        stylealign_args = [False, 1.0]  # shared_attention, strength
        
        # ControlNet needs special handling for resize_mode
        # We'll add a resize_mode attribute to the processing object later
        
        # Only add ADetailer if enabled, with proper script args structure
        if enable_adetailer:
            # Use a dictionary with proper structure for ADetailer
            # This matches what the extension expects
            adetailer_args = {
                "ad_model": "face_yolov8n.pt",
                "ad_model_classes": "",
                "ad_tab_enable": True,
                "ad_prompt": "",
                "ad_negative_prompt": "",
                "ad_confidence": 0.3,
                "ad_mask_filter_method": "Area",
                "ad_mask_k": 0,
                "ad_mask_min_ratio": 0.0,
                "ad_mask_max_ratio": 1.0,
                "ad_x_offset": 0,
                "ad_y_offset": 0,
                "ad_dilate_erode": 4,
                "ad_mask_merge_invert": "None",
                "ad_mask_blur": 4,
                "ad_denoising_strength": 0.4,
                "ad_inpaint_only_masked": True,
                "ad_inpaint_only_masked_padding": 32,
                "ad_use_inpaint_width_height": False,
                "ad_inpaint_width": 512,
                "ad_inpaint_height": 512,
                "ad_use_steps": False,
                "ad_steps": 28,
                "ad_use_cfg_scale": False,
                "ad_cfg_scale": 7.0,
                "ad_use_checkpoint": False,
                "ad_checkpoint": "Use same checkpoint",
                "ad_use_vae": False,
                "ad_vae": "Use same VAE",
                "ad_use_sampler": False,
                "ad_sampler": "DPM++ 2M",
                "ad_scheduler": "Use same scheduler",
                "ad_use_noise_multiplier": False,
                "ad_noise_multiplier": 1.0,
                "ad_use_clip_skip": False,
                "ad_clip_skip": 1,
                "ad_restore_face": False,
                "ad_controlnet_model": "None",
                "ad_controlnet_module": "None",
                "ad_controlnet_weight": 1.0,
                "ad_controlnet_guidance_start": 0.0,
                "ad_controlnet_guidance_end": 1.0
            }
            alwayson_scripts["ADetailer"] = {
                "args": [adetailer_args, {"ad_model": "None"}, {"ad_model": "None"}, {"ad_model": "None"}]
            }
            
        # Add script args for each extension - using the exact script names from the available scripts list
        alwayson_scripts["never oom integrated"] = {"args": neveroom_args}
        alwayson_scripts["freeu integrated (sd 1.x, sd 2.x, sdxl)"] = {"args": freeu_args}
        alwayson_scripts["kohya hrfix integrated"] = {"args": kohya_hrfix_args}
        alwayson_scripts["latentmodifier integrated"] = {"args": latent_modifier_args}
        alwayson_scripts["multidiffusion integrated"] = {"args": multidiffusion_args}
        alwayson_scripts["perturbedattentionguidance integrated"] = {"args": perturbed_attention_args}
        alwayson_scripts["selfattentionguidance integrated (sd 1.x, sd 2.x, sdxl)"] = {"args": sag_args}
        alwayson_scripts["stylealign integrated"] = {"args": stylealign_args}
        
        # Add alwayson_scripts to payload
        payload["alwayson_scripts"] = alwayson_scripts

        from modules.api.models import (
            StableDiffusionTxt2ImgProcessingAPI,
        )

        print(f"Финальный пейлоад: {payload=}")
        req = StableDiffusionTxt2ImgProcessingAPI(**payload)
        # generate
        # Use both extra_network_data and alwayson_scripts to handle LoRA models
        extra_network_data = {"lora": []}

        # Handle LoRA files properly - only use extra_network_data, not alwayson_scripts
        for url, scale in zip(lora_urls, lora_scales):
            # Extract the LoRA name from the URL
            if '/' in url:
                lora_name = url.split('/')[-1].split('.')[0]
            else:
                lora_name = url.split('.')[0]
                
            # Add to extra_network_data
            print(f"LOOK HEREitems=['{lora_name}', '{scale}']")
            extra_network_data["lora"].append(ExtraNetworkParams(items=[lora_name, str(scale)]))
            print(f"Adding lora: lora_name='{lora_name}'")
        
        # Don't add LoRA to alwayson_scripts - this causes the script args mismatch

        # Import the necessary modules for script registration
        from modules import scripts

        print(f"[Time] set variables again: {(datetime.now() - start).total_seconds()}")
        start = datetime.now()

        # Make sure the LoRA script is initialized
        if not hasattr(scripts.scripts_txt2img, 'selectable_scripts'):
            scripts.scripts_txt2img.initialize_scripts(False)

        print(f"[Time] initialized scripts: {(datetime.now() - start).total_seconds()}")
        start = datetime.now()
        # Print available scripts for debugging
        print("Available scripts:", [script.title().lower() for script in scripts.scripts_txt2img.scripts])

        # Ensure the model is properly loaded before using LoRA
        from modules import sd_models, shared

        self.load_model()
        print(f"[Time] loaded model: {(datetime.now() - start).total_seconds()}")

        # Now check if the model has forge_objects attribute
        if hasattr(shared.sd_model, 'forge_objects'):
            print(f"Model has forge_objects, proceeding with LoRA... shared.sd_model={shared.sd_model}")
            
            # Make sure req.alwayson_scripts doesn't contain lora to avoid script args mismatch
            if hasattr(req, 'alwayson_scripts') and req.alwayson_scripts and 'lora' in req.alwayson_scripts:
                del req.alwayson_scripts['lora']
            
            # Add a custom process method to handle ControlNet's resize_mode issue
            from modules.processing import StableDiffusionProcessingTxt2Img
            
            # Monkey patch the StableDiffusionProcessingTxt2Img class to add resize_mode attribute
            original_init = StableDiffusionProcessingTxt2Img.__init__
            
            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                # Add resize_mode attribute with default value 0 (INNER_FIT)
                self.resize_mode = 0
            
            # Apply the monkey patch
            StableDiffusionProcessingTxt2Img.__init__ = patched_init
                
            # Use only extra_network_data for LoRA handling
            resp = self.api.text2imgapi(
                txt2imgreq=req,
                extra_network_data=extra_network_data
            )
            
            # Restore the original init method to avoid side effects
            StableDiffusionProcessingTxt2Img.__init__ = original_init
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
