sergei@DESKTOP-SDULH9A:/mnt/c/Users/Сергей/Desktop/Оптимизированный движок$ cog push r8.im/sergeishapovalov/refaopt
Building Docker image from environment in cog.yaml as r8.im/sergeishapovalov/refaopt...
[+] Building 6.7s (39/39) FINISHED                                                                                                                                  docker:default
 => [internal] load build definition from Dockerfile                                                                                                                          0.0s
 => => transferring dockerfile: 3.88kB                                                                                                                                        0.0s 
 => resolve image config for docker-image://docker.io/docker/dockerfile:1.4                                                                                                   0.5s 
 => CACHED docker-image://docker.io/docker/dockerfile:1.4@sha256:9ba7531bd80fb0a858632727cf7a112fbfd19b17e94c4e84ced81e24ef1a0dbc                                             0.0s
 => [internal] load .dockerignore                                                                                                                                             0.1s 
 => => transferring context: 357B                                                                                                                                             0.0s 
 => [internal] load metadata for r8.im/cog-base:cuda12.4-python3.11                                                                                                           0.4s
 => [stage-0  1/32] FROM r8.im/cog-base:cuda12.4-python3.11@sha256:6ab3f39606455db215113f3a3fd11f739a2c87ef22f3409caafce1405cd07eb4                                           0.0s
 => [internal] load build context                                                                                                                                             4.9s 
 => => transferring context: 259.67kB                                                                                                                                         4.9s 
 => CACHED [stage-0  2/32] COPY .cog/tmp/build20250618164540.1020562803043259/cog-0.13.7-py3-none-any.whl /tmp/cog-0.13.7-py3-none-any.whl                                    0.0s
 => CACHED [stage-0  3/32] RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir /tmp/cog-0.13.7-py3-none-any.whl 'pydantic<2'                            0.0s 
 => CACHED [stage-0  4/32] COPY .cog/tmp/build20250618164540.1020562803043259/requirements.txt /tmp/requirements.txt                                                          0.0s 
 => CACHED [stage-0  5/32] RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt                                                                0.0s 
 => CACHED [stage-0  6/32] RUN echo "Cache 128 - Optimized with preloaded models and disabled extensions"                                                                     0.0s 
 => CACHED [stage-0  7/32] RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64" && chmod +x /usr/local/bin/pget  0.0s 
 => CACHED [stage-0  8/32] RUN curl -L https://github.com/mikefarah/yq/releases/download/v4.40.5/yq_linux_amd64 -o /usr/local/bin/yq && chmod +x /usr/local/bin/yq            0.0s
 => CACHED [stage-0  9/32] RUN mkdir -p /src/models/Stable-diffusion /src/models/text_encoder /src/models/VAE /src/models/RealESRGAN /src/models/ESRGAN /src/models/adetaile  0.0s 
 => CACHED [stage-0 10/32] RUN pget -f "https://civitai.com/api/download/models/691639?type=Model&format=SafeTensor&size=full&fp=fp32&&token=18b51174c4d9ae0451a3dedce1946ce  0.0s 
 => CACHED [stage-0 11/32] RUN pget -f "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors?download=true" "/src/models/text_encode  0.0s 
 => CACHED [stage-0 12/32] RUN pget -f "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true" "/src/models/text_encoder/cl  0.0s
 => CACHED [stage-0 13/32] RUN pget -f "https://weights.replicate.delivery/default/official-models/flux/ae/ae.sft" "/src/models/VAE/ae.safetensors" &                         0.0s 
 => CACHED [stage-0 14/32] RUN wget --content-disposition -P /src/models/RealESRGAN "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"   0.0s 
 => CACHED [stage-0 15/32] RUN wget -O /src/models/ESRGAN/ESRGAN_4x.pth "https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth" &                                    0.0s 
 => CACHED [stage-0 16/32] RUN wget -O /src/models/adetailer/face_yolov8n.pt "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-face.pt" &               0.0s 
 => CACHED [stage-0 17/32] RUN wget -O /src/models/adetailer/hand_yolov8n.pt "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-hand.pt" &               0.0s 
 => CACHED [stage-0 18/32] RUN wget -O /src/models/adetailer/face_yolov8s.pt "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-face.pt" &               0.0s 
 => CACHED [stage-0 19/32] RUN wget -O /src/models/adetailer/person_yolov8n-seg.pt "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt" &          0.0s 
 => CACHED [stage-0 20/32] RUN wget -O /src/models/adetailer/person_yolov8s-seg.pt "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt" &          0.0s 
 => CACHED [stage-0 21/32] RUN wget -O /src/models/adetailer/yolov8x-worldv2.pt "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt" &         0.0s 
 => CACHED [stage-0 22/32] RUN wait                                                                                                                                           0.0s 
 => CACHED [stage-0 23/32] RUN wget --content-disposition -P /src/embeddings "https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors?downlo  0.0s 
 => CACHED [stage-0 24/32] RUN pip install https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip https://github.com/mlfoundations/open_clip/ar  0.0s 
 => CACHED [stage-0 25/32] RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git /src/stable-diffusion-webui-assets                                0.0s 
 => CACHED [stage-0 26/32] RUN git clone https://github.com/lllyasviel/huggingface_guess.git /src/repositories/huggingface_guess                                              0.0s 
 => CACHED [stage-0 27/32] RUN git clone https://github.com/salesforce/BLIP.git /src/repositories/BLIP                                                                        0.0s 
 => CACHED [stage-0 28/32] RUN pip install torchvision==0.21                                                                                                                  0.0s 
 => CACHED [stage-0 29/32] RUN echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True" >> /etc/environment                                       0.0s 
 => CACHED [stage-0 30/32] RUN echo "export CUDA_LAUNCH_BLOCKING=0" >> /etc/environment                                                                                       0.0s 
 => CACHED [stage-0 31/32] WORKDIR /src                                                                                                                                       0.0s 
 => [stage-0 32/32] COPY . /src                                                                                                                                               0.3s 
 => exporting to image                                                                                                                                                        0.2s 
 => => exporting layers                                                                                                                                                       0.2s 
 => => preparing layers for inline cache                                                                                                                                      0.0s 
 => => writing image sha256:1cc3e20a5b796f9f4a6560dc41b5f4012a617f9d5ab636ada6813582b5ca2161                                                                                  0.0s 
 => => naming to r8.im/sergeishapovalov/refaopt                                                                                                                               0.0s 
Validating model schema...
Adding labels to image...
Unable to determine Git tag

Pushing image 'r8.im/sergeishapovalov/refaopt'...
Using default tag: latest
The push refers to repository [r8.im/sergeishapovalov/refaopt]
ba978092ad6e: Preparing                                                                                                                                                            
15f5ec1dc5c7: Preparing
5f70bf18a086: Preparing                                                                                                                                                            
9078f22f9b31: Preparing                                                                                                                                                            
459923d8531b: Preparing                                                                                                                                                            
da55d37f84d3: Preparing                                                                                                                                                            
174d20a3da56: Preparing                                                                                                                                                            
03c7550bf929: Preparing                                                                                                                                                            
ca945d5a0213: Preparing                                                                                                                                                            
ee8036bd0765: Preparing                                                                                                                                                            
709b046151ba: Preparing                                                                                                                                                            
853da558b66f: Preparing                                                                                                                                                            
8949dedf3338: Preparing                                                                                                                                                            
91a8caf8b871: Preparing                                                                                                                                                            
52b0aee2afab: Preparing                                                                                                                                                            
2bc994add269: Preparing                                                                                                                                                            
ea3b7a4b846c: Preparing                                                                                                                                                            
3120c481153e: Preparing                                                                                                                                                            
e68d7e7b1cab: Preparing                                                                                                                                                            
15f5ec1dc5c7: Pushed
ae866f3796fe: Layer already exists
f99a69ccdda5: Layer already exists
f914a7b8ea96: Layer already exists
f8a2e0c5e2bc: Layer already exists
c1a9c1facbe5: Layer already exists
2682c271dc46: Layer already exists
25205b8e84f0: Layer already exists
552b2b859388: Layer already exists
461cf6ba28a4: Layer already exists 
2bbe92c04220: Layer already exists
5b96e3139270: Layer already exists
538b8d249fcb: Layer already exists
3439643961e5: Layer already exists
e30feec8714e: Layer already exists
2b2b5b12f484: Layer already exists
8ec45d846b34: Layer already exists
latest: digest: sha256:ff540cdee08fd1d7e60957ad259c3d4b4ee71282070f50ae1236321537da0e4a size: 10772
Image 'r8.im/sergeishapovalov/refaopt' pushed

Run your model on Replicate:
    https://replicate.com/sergeishapovalov/refaopt
sergei@DESKTOP-SDULH9A:/mnt/c/Users/Сергей/Desktop/Оптимизированный движок$ cd cogworder-main/
sergei@DESKTOP-SDULH9A:/mnt/c/Users/Сергей/Desktop/Оптимизированный движок/cogworder-main$ docker build --platform=linux/amd64 --tag jettongames/runpod-migrate:1 --build-arg COG_REPO=sergeishapovalov --build-arg COG_MODEL=refaopt --build-arg COG_VERSION=ff540cdee08fd1d7e60957ad259c3d4b4ee71282070f50ae1236321537da0e4a . && docker push jettongames/runpod-migrate:1
[+] Building 125.1s (10/10) FINISHED                                                                                                                                docker:default
 => [internal] load build definition from Dockerfile                                                                                                                          0.0s
 => => transferring dockerfile: 945B                                                                                                                                          0.0s 
 => WARN: InvalidDefaultArgInFrom: Default value for ARG r8.im/${COG_REPO}/${COG_MODEL}@sha256:${COG_VERSION} results in empty or invalid base image name (line 5)            0.0s 
 => [internal] load metadata for r8.im/sergeishapovalov/refaopt@sha256:ff540cdee08fd1d7e60957ad259c3d4b4ee71282070f50ae1236321537da0e4a                                       0.0s 
 => [internal] load .dockerignore                                                                                                                                             0.0s 
 => => transferring context: 2B                                                                                                                                               0.0s 
 => [1/5] FROM r8.im/sergeishapovalov/refaopt@sha256:ff540cdee08fd1d7e60957ad259c3d4b4ee71282070f50ae1236321537da0e4a                                                         0.2s 
 => [internal] load build context                                                                                                                                             0.0s 
 => => transferring context: 59B                                                                                                                                              0.0s 
 => [2/5] RUN apt-get update && apt-get upgrade -y &&     apt-get install -y --no-install-recommends software-properties-common curl git openssh-server &&     add-apt-rep  103.6s
 => [3/5] RUN python3 -m venv /opt/venv                                                                                                                                       3.0s
 => [4/5] RUN /opt/venv/bin/pip install runpod                                                                                                                               17.1s
 => [5/5] ADD src/handler.py /rp_handler.py                                                                                                                                   0.0s
 => exporting to image                                                                                                                                                        1.2s
 => => exporting layers                                                                                                                                                       1.2s
 => => writing image sha256:4dda89dcd7fa6dc6bd82c0d53b4869c8af5ee14790063f94e0adc256c6937f9c                                                                                  0.0s
 => => naming to docker.io/jettongames/runpod-migrate:1                                                                                                                       0.0s

 1 warning found (use docker --debug to expand):
 - InvalidDefaultArgInFrom: Default value for ARG r8.im/${COG_REPO}/${COG_MODEL}@sha256:${COG_VERSION} results in empty or invalid base image name (line 5)
The push refers to repository [docker.io/jettongames/runpod-migrate]
a8b4b0118c70: Preparing                                                                                                                                                            
c090a90c6529: Preparing                                                                                                                                                            
f890fd8b67d6: Preparing                                                                                                                                                            
9516691afc57: Preparing                                                                                                                                                            
ba978092ad6e: Preparing                                                                                                                                                            
15f5ec1dc5c7: Preparing                                                                                                                                                            
5f70bf18a086: Preparing                                                                                                                                                            
9078f22f9b31: Preparing                                                                                                                                                            
459923d8531b: Preparing                                                                                                                                                            
da55d37f84d3: Preparing                                                                                                                                                            
174d20a3da56: Preparing                                                                                                                                                            
03c7550bf929: Preparing                                                                                                                                                            
ca945d5a0213: Preparing                                                                                                                                                            
ee8036bd0765: Preparing                                                                                                                                                            
709b046151ba: Preparing                                                                                                                                                            
853da558b66f: Preparing                                                                                                                                                            
8949dedf3338: Preparing                                                                                                                                                            
91a8caf8b871: Preparing                                                                                                                                                            
52b0aee2afab: Preparing                                                                                                                                                            
2bc994add269: Preparing                                                                                                                                                            
ea3b7a4b846c: Preparing                                                                                                                                                            
3120c481153e: Preparing                                                                                                                                                            
e68d7e7b1cab: Preparing                                                                                                                                                            
52b0aee2afab: Pushed
ae866f3796fe: Mounted from jettongames/flux-dev-lora
f99a69ccdda5: Mounted from jettongames/flux-dev-lora
f914a7b8ea96: Mounted from jettongames/flux-dev-lora
f8a2e0c5e2bc: Mounted from jettongames/flux-dev-lora
c1a9c1facbe5: Mounted from jettongames/flux-dev-lora
2682c271dc46: Mounted from jettongames/flux-dev-lora
25205b8e84f0: Mounted from jettongames/flux-dev-lora
552b2b859388: Mounted from jettongames/flux-dev-lora
461cf6ba28a4: Mounted from jettongames/flux-dev-lora
2bbe92c04220: Mounted from jettongames/flux-dev-lora
5b96e3139270: Mounted from jettongames/flux-dev-lora
538b8d249fcb: Mounted from jettongames/flux-dev-lora
3439643961e5: Mounted from jettongames/flux-dev-lora
e30feec8714e: Mounted from jettongames/flux-dev-lora
2b2b5b12f484: Mounted from jettongames/flux-dev-lora
8ec45d846b34: Mounted from jettongames/flux-dev-lora
1: digest: sha256:3ca9a39d6cccf6ebd32df0d55259e49339a89f9f4a7a036dac95cd69952894a8 size: 11616