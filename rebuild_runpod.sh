#!/bin/bash

# Скрипт для пересборки RunPod образа с новым Cog образом

echo "=== Пересборка RunPod образа ==="

# Новый хэш образа из логов сборки
NEW_COG_VERSION="78ad9ae200425860ee8b155fd72a0b32098ebd9dfe613df7782a9fc256b7ee3b"

echo "Используем новый хэш образа: $NEW_COG_VERSION"

cd cogworder-main/

echo "Собираем новый RunPod образ..."
docker build --platform=linux/amd64 \
  --tag jettongames/runpod-migrate:2 \
  --build-arg COG_REPO=sergeishapovalov \
  --build-arg COG_MODEL=refaopt \
  --build-arg COG_VERSION=$NEW_COG_VERSION . && \
docker push jettongames/runpod-migrate:2

echo "=== Готово! ==="
echo "Новый образ: jettongames/runpod-migrate:2"
echo "Теперь можно тестировать на RunPod"