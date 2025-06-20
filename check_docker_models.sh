#!/bin/bash

echo "=== Проверка предустановленных моделей в Docker образе ==="

echo "1. Проверяем основные модели:"
echo "Flux checkpoint:"
ls -la /src/models/Stable-diffusion/ 2>/dev/null || echo "❌ Директория /src/models/Stable-diffusion/ не найдена"

echo -e "\nText encoders:"
ls -la /src/models/text_encoder/ 2>/dev/null || echo "❌ Директория /src/models/text_encoder/ не найдена"

echo -e "\nVAE:"
ls -la /src/models/VAE/ 2>/dev/null || echo "❌ Директория /src/models/VAE/ не найдена"

echo -e "\nESRGAN:"
ls -la /src/models/ESRGAN/ 2>/dev/null || echo "❌ Директория /src/models/ESRGAN/ не найдена"

echo -e "\nADetailer модели:"
ls -la /src/models/adetailer/ 2>/dev/null || echo "❌ Директория /src/models/adetailer/ не найдена"

echo -e "\n2. Проверяем размеры файлов:"
if [ -f "/src/models/Stable-diffusion/flux_checkpoint.safetensors" ]; then
    size=$(du -h /src/models/Stable-diffusion/flux_checkpoint.safetensors | cut -f1)
    echo "✓ Flux checkpoint: $size"
else
    echo "❌ Flux checkpoint не найден"
fi

if [ -f "/src/models/text_encoder/clip_l.safetensors" ]; then
    size=$(du -h /src/models/text_encoder/clip_l.safetensors | cut -f1)
    echo "✓ CLIP-L: $size"
else
    echo "❌ CLIP-L не найден"
fi

if [ -f "/src/models/text_encoder/t5xxl_fp16.safetensors" ]; then
    size=$(du -h /src/models/text_encoder/t5xxl_fp16.safetensors | cut -f1)
    echo "✓ T5XXL: $size"
else
    echo "❌ T5XXL не найден"
fi

if [ -f "/src/models/VAE/ae.safetensors" ]; then
    size=$(du -h /src/models/VAE/ae.safetensors | cut -f1)
    echo "✓ VAE: $size"
else
    echo "❌ VAE не найден"
fi

echo -e "\n3. Общий размер моделей:"
if [ -d "/src/models" ]; then
    total_size=$(du -sh /src/models | cut -f1)
    echo "Общий размер /src/models: $total_size"
else
    echo "❌ Директория /src/models не найдена"
fi

echo -e "\n4. Свободное место:"
df -h /src

echo -e "\n=== Проверка завершена ==="