# Phone Calls API: GigaAM

Сервис для обработки WAV-записей телефонных разговоров: транскрибация через GigaAM, диаризация через pyannote и суммаризация через OpenAI-compatible API.

## Что делает сервис

- Принимает WAV-файл через HTTP API.
- Распознает речь через GigaAM `v3_e2e_rnnt`.
- Получает word-level timestamps из GigaAM longform pipeline.
- Определяет спикеров через `pyannote/speaker-diarization-community-1`.
- Собирает реплики по спикерам.
- Отправляет диалог в LLM и возвращает суммаризацию.

## Требования

- Docker и Docker Compose для рекомендуемого запуска.
- Hugging Face token с доступом к pyannote-моделям.
- API-ключ OpenAI-compatible провайдера для суммаризации.
- WAV-аудио на входе. Другие форматы API сейчас отклоняет.

Перед первым запуском нужно принять условия использования pyannote-моделей на Hugging Face:

- `pyannote/speaker-diarization-community-1`
- `pyannote/segmentation-3.0`

## Как Получить Hugging Face Token

1. Зарегистрируйтесь или войдите в аккаунт на Hugging Face: https://huggingface.co.

2. Откройте страницу токенов: https://huggingface.co/settings/tokens.

3. Нажмите `Create new token`.

4. Выберите тип токена `Read`.

5. Скопируйте созданный токен и добавьте его в `.env`:

```env
HF_TOKEN=ваш_huggingface_token
```

6. Откройте страницы pyannote-моделей и примите условия использования:

- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://huggingface.co/pyannote/segmentation-3.0

Токен должен принадлежать тому же аккаунту Hugging Face, на котором приняты условия pyannote-моделей. Без этого сервис не сможет скачать модели VAD и диаризации.

## Быстрый Запуск Через Docker

1. Создайте файл `.env` в корне проекта:

```env
API_KEY=ваш_api_ключ_для_llm
HF_TOKEN=ваш_huggingface_token
```

2. Проверьте `config.yaml`:

```yaml
gigaam:
  model_name: v3_e2e_rnnt
  device: cpu
  pyannote_model: pyannote/speaker-diarization-community-1
  hf_token: null
  num_speakers: null
  min_speakers: 2
  max_speakers: 2
  vad_onset: 0.5
  vad_offset: 0.363
  chunk_size: 30.0
  fr_batch_size: 16
  fr_num_workers: 0
  merge_gap: 1.0
```

Для CPU оставьте `device: cpu`. Для GPU можно указать `device: cuda`, но Dockerfile в этой ветке собран на `python:3.11-slim` и не настраивает CUDA-зависимости.

3. Соберите и запустите сервис:

```bash
docker compose up --build -d
```

4. Проверьте состояние:

```bash
curl http://localhost:8096/health
```

Ожидаемый ответ:

```json
{"status":"healthy"}
```

Первый запуск может занять несколько минут: GigaAM и pyannote скачивают модели в Docker volume `model-cache`.

## Локальный Запуск Без Docker

1. Создайте виртуальное окружение:

```bash
python3 -m venv venv
```

2. Установите зависимости:

```bash
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

3. Установите системные зависимости:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 git
```

4. Создайте `.env`:

```env
API_KEY=ваш_api_ключ_для_llm
HF_TOKEN=ваш_huggingface_token
```

5. Запустите API:

```bash
./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8096
```

## Конфигурация

Основные настройки находятся в `config.yaml`.

Параметры GigaAM:

- `model_name`: модель GigaAM, по умолчанию `v3_e2e_rnnt`.
- `device`: устройство для инференса, `cpu` или `cuda`.
- `pyannote_model`: модель диаризации pyannote.
- `num_speakers`: точное число спикеров, если известно. Если `null`, используются `min_speakers` и `max_speakers`.
- `min_speakers`: минимальное число спикеров для pyannote.
- `max_speakers`: максимальное число спикеров для pyannote.
- `vad_onset`: порог начала речи для VAD, как в WhisperX.
- `vad_offset`: порог окончания речи для VAD, как в WhisperX.
- `chunk_size`: максимальный размер чанка речи в секундах.
- `fr_batch_size`: batch size для longform-инференса GigaAM.
- `fr_num_workers`: число workers для DataLoader.
- `merge_gap`: максимальная пауза для склейки соседних слов одного спикера в одну реплику.

Параметры суммаризации:

- `openai.base_url`: URL OpenAI-compatible API.
- `openai.model`: модель для суммаризации.
- `openai.temperature`: температура генерации.
- `openai.max_tokens`: лимит ответа.
- `prompts.default`: системная инструкция для суммаризации.

Переменные окружения имеют приоритет для ключей:

- `API_KEY`: ключ LLM-провайдера.
- `HF_TOKEN`: токен Hugging Face.
- `CONFIG_PATH`: путь к альтернативному YAML-конфигу, если нужно запустить сервис с другим конфигом.

## API

### Проверка Статуса

```http
GET /health
```

Пример:

```bash
curl http://localhost:8096/health
```

### Транскрибация И Суммаризация

```http
POST /transcribe-and-summarize
```

Поле формы:

- `file`: WAV-файл.

Пример:

```bash
curl -X POST http://localhost:8096/transcribe-and-summarize \
  -F "file=@call.wav"
```

Ответ:

```json
{
  "transcription": "[SPEAKER_01] текст реплики\n[SPEAKER_02] текст реплики",
  "summary": "Название: ...\nСуммаризация: ...\nЭмоциональная окраска: ..."
}
```

### Получить Системный Промпт

```bash
curl http://localhost:8096/summarization/system-prompt
```

### Обновить Системный Промпт

```bash
curl -X PUT http://localhost:8096/summarization/system-prompt \
  -H "Content-Type: application/json" \
  -d '{"system_prompt":"Новая инструкция"}'
```

### Получить Модель Суммаризации

```bash
curl http://localhost:8096/summarization/model
```

### Обновить Модель Суммаризации

```bash
curl -X PUT http://localhost:8096/summarization/model \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-max-preview"}'
```

## Логи И Обслуживание

Посмотреть логи Docker-сервиса:

```bash
docker compose logs -f api
```

Остановить сервис:

```bash
docker compose down
```

Остановить сервис и удалить кеш моделей:

```bash
docker compose down -v
```

## Частые Проблемы

### Нет доступа к pyannote

Симптомы: ошибка загрузки diarization-модели или segmentation-модели.

Что проверить:

- `HF_TOKEN` задан в `.env`.
- Токен имеет доступ к Hugging Face.
- Условия использования pyannote-моделей приняты на Hugging Face.

### Сервис долго стартует

На первом запуске скачиваются модели GigaAM и pyannote. Это нормально. Модели кешируются в Docker volume `model-cache`.

### Ошибка Only WAV files are supported

API принимает только файлы с расширением `.wav` или `.wave`. Конвертируйте аудио заранее:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### Медленная обработка на CPU

CPU-режим значительно медленнее GPU. Для ускорения можно уменьшить размер входных файлов, увеличить ресурсы CPU или настроить запуск на CUDA с совместимым Docker-образом и PyTorch/CUDA-зависимостями.

## Безопасность

Не коммитьте `.env` и реальные значения `API_KEY`/`HF_TOKEN` в репозиторий. В README используйте только шаблоны переменных.
