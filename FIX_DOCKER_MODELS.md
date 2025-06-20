# Исправление проблемы с предустановкой моделей в Docker

## Проблема
Анализ логов показал, что модели НЕ предустанавливаются в Docker образ при сборке:
- Docker образ весит только 11GB, а один Flux чекпоинт весит 24GB
- Все команды загрузки в `cog.yaml` помечены как `CACHED` - используется старый кэш без моделей
- В логах выполнения видно скачивание всех моделей заново (~60 секунд потерь)

## Внесенные исправления

### 1. Обновлен кэш в `cog.yaml`
- Изменен timestamp: `2025-06-18:17-25`
- Изменен номер кэша: `Cache 129`
- Добавлена детальная проверка загруженных моделей

### 2. Улучшена диагностика в `predict.py`
- Добавлен метод `_check_preinstalled_models()` для проверки наличия моделей
- Улучшена логика проверки дополнительных модулей
- Добавлены подробные логи с размерами файлов

### 3. Исправлен `modules/adetailer_patch.py`
- Добавлена проверка размеров файлов
- Улучшена диагностика отсутствующих моделей
- Добавлено создание директорий при необходимости

### 4. Создан скрипт проверки `check_docker_models.sh`
- Проверяет наличие всех предустановленных моделей
- Показывает размеры файлов и общий размер
- Можно запустить внутри контейнера для диагностики

## Команды для исправления

### 1. Принудительная пересборка без кэша
```bash
cog push --no-cache r8.im/sergeishapovalov/refaopt
```

### 2. Проверка результата
После сборки образ должен весить ~35-40GB вместо 11GB.

### 3. Пересборка RunPod образа
```bash
cd cogworder-main/
docker build --platform=linux/amd64 --tag jettongames/runpod-migrate:2 \
  --build-arg COG_REPO=sergeishapovalov \
  --build-arg COG_MODEL=refaopt \
  --build-arg COG_VERSION=<новый_хэш_образа> . && \
docker push jettongames/runpod-migrate:2
```

## Ожидаемый результат

### До исправления:
- Общее время: 378 секунд
- Скачивание моделей: ~60 секунд
- Docker образ: 11GB

### После исправления:
- Общее время: ~120-150 секунд (улучшение на 60-70%)
- Скачивание моделей: 0 секунд (предустановлены)
- Docker образ: ~35-40GB

## Проверка успешности

### В логах сборки должно появиться:
```
=== ПРОВЕРКА ЗАГРУЖЕННЫХ МОДЕЛЕЙ ===
Flux checkpoint:
-rw-r--r-- 1 root root 24G flux_checkpoint.safetensors
Text encoders:
-rw-r--r-- 1 root root 246M clip_l.safetensors
-rw-r--r-- 1 root root 9.8G t5xxl_fp16.safetensors
...
Общий размер моделей: 35G
=== КОНЕЦ ПРОВЕРКИ ===
```

### В логах выполнения должно появиться:
```
[Setup] ✓ Flux checkpoint: /src/models/Stable-diffusion/flux_checkpoint.safetensors (24000.0 MB)
[Setup] ✓ CLIP-L: /src/models/text_encoder/clip_l.safetensors (246.0 MB)
[Setup] ✓ T5XXL: /src/models/text_encoder/t5xxl_fp16.safetensors (9800.0 MB)
[Additional modules] Используем локальный clip_l: /src/models/text_encoder/clip_l.safetensors
```

## Дополнительная диагностика

Если проблема повторится, запустить внутри контейнера:
```bash
bash /src/check_docker_models.sh
```

Это покажет точное состояние предустановленных моделей.