from contextlib import contextmanager
from typing import Callable
from time import perf_counter

@contextmanager
def catchtime(tag: str) -> Callable[[], float]:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f'[Timer: {tag}]: {perf_counter() - start:.3f} seconds')


with catchtime(tag="Imports"):
    import json
    import os
    import re
    import subprocess
    import sys
    import time
    from modules import initialize
    from modules import timer
    from fastapi import FastAPI
    from cog import BasePredictor, Input, Path
    from weights import WeightsDownloadCache


FLUX_CHECKPOINT_URL = "https://civitai.com/api/download/models/691639?type=Model&format=SafeTensor&size=full&fp=fp32&&token=18b51174c4d9ae0451a3dedce1946ce3"
sys.path.extend(["/src"])


ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


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
    subprocess.check_call(["pget", "-f", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)  # Выводим время загрузки


class Predictor(BasePredictor):
    weights_cache = WeightsDownloadCache()

    def _download_loras(self, lora_urls: list[str]):
        lora_paths = []

        for url in lora_urls:
            if re.match(r"^https?://replicate.delivery/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/trained_model.tar", url):
                print(f"Downloading LoRA weights from - Replicate URL: {url}")
                lora_path = self.weights_cache.ensure(
                    url=url,
                    mv_from="output/flux_train_replicate/lora.safetensors",
                )
                print(f"{lora_path=}")
                lora_paths.append(lora_path)
            elif re.match(r"^https?://replicate.delivery/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/flux-lora.tar", url):
                print(f"Downloading LoRA weights from - Replicate URL: {url}")
                lora_path = self.weights_cache.ensure(
                    url=url,
                    mv_from="flux-lora/flux-lora.safetensors",
                )
                print(f"{lora_path=}")
                lora_paths.append(lora_path)
            elif re.match(r"^https?://civitai.com/api/download/models/[0-9]+\?type=Model&format=SafeTensor", url):
                # split url to get first part of the url, everythin before '?type'
                civitai_slug = url.split('?type')[0]
                print(f"Downloading LoRA weights from - Civitai URL: {civitai_slug}")
                lora_path = self.weights_cache.ensure(url, file=True)
                lora_paths.append(lora_path)
            elif url.endswith('.safetensors'):
                print(f"Downloading LoRA weights from - safetensor URL: {url}")
                try:
                    lora_path = self.weights_cache.ensure(url, file=True)
                except Exception as e:
                    print(f"Error downloading LoRA weights: {e}")
                    continue
                print(f"{lora_path=}")
                lora_paths.append(lora_path)

        files = [os.path.join(self.weights_cache.base_dir, f) for f in os.listdir(self.weights_cache.base_dir)]
        print(f'Available loras: {files}')

        return lora_paths

    def _setup_memory_management(self) -> None:
        # Безопасный импорт memory_management
        try:
            from backend import memory_management
            self.has_memory_management = True
        except ImportError as e:
            print(f"Предупреждение: Не удалось импортировать memory_management: {e}")
            self.has_memory_management = False

        if not self.has_memory_management:
            print("[GPU Setting] memory_management не доступен, используются настройки по умолчанию")

        # Выделяем больше памяти для загрузки весов модели (90% для весов, 10% для вычислений)
        total_vram = memory_management.total_vram
        inference_memory = int(total_vram * 0.6)  # 60% для вычислений
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

    def _setup_api(self) -> None:
        from modules.api.api import Api

        with catchtime(tag="init fastapi"):
            app = FastAPI()

        with catchtime(tag="init queue"):
            from modules.call_queue import queue_lock

        self.api = Api(app=app, queue_lock=queue_lock)

    def setup(self, force_download_url: str = None) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Загружаем модель Flux во время сборки, чтобы ускорить генерацию
        target_dir = "/src/models/Stable-diffusion"
        os.makedirs(target_dir, exist_ok=True)
        model_path = os.path.join(target_dir, "flux_checkpoint.safetensors")

        if not os.path.exists(model_path):
            print(f"Загружаем модель Flux...")
            download_base_weights(url=FLUX_CHECKPOINT_URL, dest=model_path)
        elif force_download_url:
            print(f"Загружаем модель Flux... {force_download_url=}")
            download_base_weights(url=force_download_url, dest=model_path)
        else:
            print(f"Модель Flux уже загружена: {model_path}, {os.path.exists(model_path)=}, {force_download_url=}")

        # workaround for replicate since its entrypoint may contain invalid args
        os.environ["IGNORE_CMD_ARGS_ERRORS"] = "1"
        startup_timer = timer.startup_timer
        startup_timer.record("launcher")

        with catchtime(tag="Initialize.*"):
            with catchtime(tag="Initialize.imports"):
                initialize.imports()
            with catchtime(tag="Initialize.check_versions"):
                initialize.check_versions()
            with catchtime(tag="Initialize.initialize"):
                initialize.initialize()

        # Импортируем shared после initialize.initialize()
        from modules import shared

        # Устанавливаем forge_preset на 'flux'
        shared.opts.set('forge_preset', 'flux')
        shared.opts.set('show_progress_every_n_steps', 1)
        shared.parallel_processing_allowed = False

        # Устанавливаем чекпоинт
        shared.opts.set('sd_model_checkpoint', 'flux_checkpoint.safetensors')

        # Оптимизация памяти для лучшего качества и скорости с Flux
        with catchtime(tag="Setup Memory Management"):
            self._setup_memory_management()

        with catchtime(tag="Setup API"):
            self._setup_api()

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        width: int = Input(
            description="Width of output image", ge=0, le=1280, default=0
        ),
        height: int = Input(
            description="Height of output image", ge=0, le=1280, default=0
        ),
        num_outputs: int = Input(
            description="Number of images to output", ge=1, le=4, default=1
        ),
        sampler: str = Input(
            description="Sampling method для Flux моделей",
            choices=[
                "[Forge] Flux Realistic",
                "Euler",
                "DEIS",
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
            default="Euler",
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
            description="Number of denoising steps", ge=1, le=200, default=8
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
        enable_hr: bool = Input(
            description="Hires. fix",
            default=True,
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
                "debug",
            ],
            default="R-ESRGAN 4x+",
        ),
        hr_steps: int = Input(
            description="Inference steps for Hires. fix", ge=0, le=100, default=8
        ),
        hr_scale: float = Input(
            description="Factor to scale image by", ge=1, le=4, default=1.3
        ),
        denoising_strength: float = Input(
            description="Denoising strength. 1.0 corresponds to full destruction of information in init image",
            ge=0,
            le=1,
            default=0.3,
        ),
        debug_flux_checkpoint_url: str = Input(
            description="Flux checkpoint URL. UPDATE 18.05.2025 BREAKS IT BECAUSE OF NEW def calculate_shorthash(self):",
            default=""
        ),
        enable_clip_l: bool = Input(
            description="Enable encoder",
            default=False
        ),
        enable_t5xxl_fp16: bool = Input(
            description="t5xxl_fp16",
            default=False
        ),
        enable_ae: bool = Input(
            description="Enable ae",
            default=False
        ),
        force_model_reload: bool = Input(
            description="Load Flux model from scratch",
            default=False
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
            default="bnb-nf4 (fp16 LoRA)",
        ),
        image: str = Input(
            description="Input image for image to image mode. The aspect ratio of your output will match this image",
            default="",
        ),
        prompt_strength: float = Input(
            description="Prompt strength (or denoising strength) when using image to image. 1.0 corresponds to full destruction of information in image.",
            ge=0, le=1, default=0.8,
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=list(ASPECT_RATIOS.keys()),
            default="9:16"
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        lora_urls: list[str] = Input(
            description="Ссылки на LoRA файлы",
            default=[],
        ),
        lora_scales: list[float] = Input(
            description="Lora scales",
            default=[1],
        ),
        ad_prompt: str = Input(
            default="",
        ),
        ad_hands_prompt: str = Input(
            default="",
        ),
        postback_url: str = Input(
            default="",
        ),
        adetailer: bool = Input(
            description="Enable adetailer",
            default=False
        ),
        adetailer_args: str = Input(
            description="Adetailer arguments",
            default="{}"
        ),
        adetailer_args_hands: str = Input(
            description="Adetailer arguments for hands",
            default="{}"
        ),
    ) -> list[Path]:
        import threading
        import time
        import traceback
        import requests

        adetailer_args = json.loads(adetailer_args)
        adetailer_args_hands = json.loads(adetailer_args_hands)

        def kek():
            print("Running kek")
            while True:
                try:

                    from modules.api.models import (
                        StableDiffusionTxt2ImgProcessingAPI,
                        StableDiffusionImg2ImgProcessingAPI,
                        ProgressRequest
                    )
                    response = self.api.progressapi(ProgressRequest(skip_current_image=False))
                    requests.post(
                        url=postback_url or "https://back-dev.recrea.ai/api/v1/live_preview",
                        json=response.dict(),
                    )
                    time.sleep(1)
                except Exception as e:
                    print(f"[progress] got: {e=}")

        print("Starting kek")
        thread = threading.Thread(target=kek, daemon=True)
        thread.start()

        if image == "runpod":
            print("Setting image to None, because of runpod")
            image = None

        print("Cache version 106")
        """Run a single prediction on the model"""
        from modules.extra_networks import ExtraNetworkParams
        from modules import scripts
        from modules.api.models import (
            StableDiffusionTxt2ImgProcessingAPI,
            StableDiffusionImg2ImgProcessingAPI,
            ProgressRequest
        )
        from PIL import Image
        import uuid
        import base64
        from io import BytesIO
        from modules import shared
        from modules_forge.main_entry import forge_unet_storage_dtype_options
        from backend.args import dynamic_args

        forge_unet_storage_dtype, online_lora = forge_unet_storage_dtype_options.get(
            forge_unet_storage_dtype, (None, False),
        )

        print(f"Setting {forge_unet_storage_dtype=}, {online_lora=}")
        shared.opts.set('forge_unet_storage_dtype', forge_unet_storage_dtype)
        dynamic_args['online_lora'] = online_lora

        if debug_flux_checkpoint_url:
            self.setup(force_download_url=debug_flux_checkpoint_url)

        lora_paths = self._download_loras(lora_urls)

        if (not width) or (not height):
            width, height = ASPECT_RATIOS[aspect_ratio]

        payload = {
            "prompt": prompt,
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
            "hr_additional_modules": [],
        }

        if image:
            payload['denoising_strength'] = prompt_strength
            payload['init_images'] = [image]
            payload['resize_mode'] = 1

        if adetailer:
            payload["alwayson_scripts"] = {
                "ADetailer": {
                    "args": [
                        {
                            "ad_model": adetailer_args.get("ad_model", "face_yolov8n.pt"),
                            "ad_prompt": adetailer_args.get("ad_prompt", ad_prompt),
                            "ad_confidence": adetailer_args.get("ad_confidence", 0.5),
                            "ad_mask_filter_method": adetailer_args.get("ad_mask_filter_method", "Area"),
                            "ad_mask_k": adetailer_args.get("ad_mask_k", 0),
                            "ad_mask_min_ratio": adetailer_args.get("ad_mask_min_ratio", 0),
                            "ad_mask_max_ratio": adetailer_args.get("ad_mask_max_ratio", 1),
                            "ad_x_offset": adetailer_args.get("ad_x_offset", 0),
                            "ad_y_offset": adetailer_args.get("ad_y_offset", 0),
                            "ad_dilate_erode": adetailer_args.get("ad_dilate_erode", 4),
                            "ad_mask_merge_invert": adetailer_args.get("ad_mask_merge_invert", "None"),
                            "ad_mask_blur": adetailer_args.get("ad_mask_blur", 4),
                            "ad_denoising_strength": adetailer_args.get("ad_denoising_strength", 0.1),
                            "ad_inpaint_only_masked": adetailer_args.get("ad_inpaint_only_masked", True),
                            "ad_inpaint_only_masked_padding": adetailer_args.get("ad_inpaint_only_masked_padding", 32),
                            "ad_inpaint_width": adetailer_args.get("ad_inpaint_width", 1024),
                            "ad_inpaint_height": adetailer_args.get("ad_inpaint_height", 1024),
                            "ad_use_steps": adetailer_args.get("ad_use_steps", True),
                            "ad_steps": adetailer_args.get("ad_steps", 8),
                            "ad_use_cfg_scale": adetailer_args.get("ad_use_cfg_scale", False),
                            "ad_cfg_scale": adetailer_args.get("ad_cfg_scale", 7),
                            "ad_use_checkpoint": adetailer_args.get("ad_use_checkpoint", False),
                            "ad_vae": adetailer_args.get("ad_vae", False),
                            "ad_use_sampler": adetailer_args.get("ad_use_sampler", False),
                            "ad_scheduler": adetailer_args.get("ad_scheduler", "Use same scheduler"),
                            "ad_use_noise_multiplier": adetailer_args.get("ad_use_noise_multiplier", False),
                            "ad_noise_multiplier": adetailer_args.get("ad_noise_multiplier", 1),
                            "ad_use_clip_skip": adetailer_args.get("ad_use_clip_skip", False),
                        },
                        {
                            "ad_model": adetailer_args.get("ad_model", "face_yolov8n.pt"),
                            "ad_prompt": adetailer_args.get("ad_prompt", ad_prompt),
                            "ad_confidence": adetailer_args.get("ad_confidence", 0.5),
                            "ad_mask_filter_method": adetailer_args.get("ad_mask_filter_method", "Area"),
                            "ad_mask_k": adetailer_args.get("ad_mask_k", 0),
                            "ad_mask_min_ratio": adetailer_args.get("ad_mask_min_ratio", 0),
                            "ad_mask_max_ratio": adetailer_args.get("ad_mask_max_ratio", 1),
                            "ad_x_offset": adetailer_args.get("ad_x_offset", 0),
                            "ad_y_offset": adetailer_args.get("ad_y_offset", 0),
                            "ad_dilate_erode": adetailer_args.get("ad_dilate_erode", 4),
                            "ad_mask_merge_invert": adetailer_args.get("ad_mask_merge_invert", "None"),
                            "ad_mask_blur": adetailer_args.get("ad_mask_blur", 4),
                            "ad_denoising_strength": adetailer_args.get("ad_denoising_strength", 0.1),
                            "ad_inpaint_only_masked": adetailer_args.get("ad_inpaint_only_masked", True),
                            "ad_inpaint_only_masked_padding": adetailer_args.get("ad_inpaint_only_masked_padding", 32),
                            "ad_inpaint_width": adetailer_args.get("ad_inpaint_width", 1024),
                            "ad_inpaint_height": adetailer_args.get("ad_inpaint_height", 1024),
                            "ad_use_steps": adetailer_args.get("ad_use_steps", True),
                            "ad_steps": adetailer_args.get("ad_steps", 8),
                            "ad_use_cfg_scale": adetailer_args.get("ad_use_cfg_scale", False),
                            "ad_cfg_scale": adetailer_args.get("ad_cfg_scale", 7),
                            "ad_use_checkpoint": adetailer_args.get("ad_use_checkpoint", False),
                            "ad_vae": adetailer_args.get("ad_vae", False),
                            "ad_use_sampler": adetailer_args.get("ad_use_sampler", False),
                            "ad_scheduler": adetailer_args.get("ad_scheduler", "Use same scheduler"),
                            "ad_use_noise_multiplier": adetailer_args.get("ad_use_noise_multiplier", False),
                            "ad_noise_multiplier": adetailer_args.get("ad_noise_multiplier", 1),
                            "ad_use_clip_skip": adetailer_args.get("ad_use_clip_skip", False),
                        }
                    ]
                }
            }

        print(f"Финальный пейлоад: {payload=}")
        print("Available scripts:", [script for script in scripts.scripts_txt2img.scripts])

        req = dict(
            forge_unet_storage_dtype=forge_unet_storage_dtype,
            force_model_reload=force_model_reload,
            extra_network_data={
                "lora": [
                    ExtraNetworkParams(
                        items=[
                            lora_path.split('/')[-1].split('.safetensors')[0],
                            str(lora_scale)
                        ]
                    )
                    for lora_path, lora_scale in zip(lora_paths, lora_scales)
                ]
            },
            additional_modules={
                "clip_l.safetensors": enable_clip_l,
                "t5xxl_fp16.safetensors": enable_t5xxl_fp16,
                "ae.safetensors": enable_ae,
            },
        )

        for lora in req['extra_network_data']['lora']:
            print(f"LoRA: {lora.items=}")

        with catchtime(tag="Total Prediction Time"):
            if image:
                req['img2imgreq'] = StableDiffusionImg2ImgProcessingAPI(**payload)
                resp = self.api.img2imgapi(**req)
            else:
                req['txt2imgreq'] = StableDiffusionTxt2ImgProcessingAPI(**payload)
                resp = self.api.text2imgapi(**req)

        info = json.loads(resp.info)
        outputs = []

        with catchtime(tag="Total Encode Time"):
            for i, image in enumerate(resp.images):
                seed = info["all_seeds"][i]
                gen_bytes = BytesIO(base64.b64decode(image))
                gen_data = Image.open(gen_bytes)
                filename = f"{seed}-{uuid.uuid1()}.{output_format}"

                if output_format != 'png':
                    gen_data.save(fp=filename, format=output_format, quality=100, optimize=True)
                else:
                    gen_data.save(fp=filename, format=output_format)

                output = Path(filename)
                outputs.append(output)

        return outputs
