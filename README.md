# Phone Calls API

Сервис для обработки WAV-записей телефонных разговоров: транскрибация через WhisperX, диаризация через pyannote и суммаризация через OpenAI-compatible API.

## Что делает сервис

- Принимает WAV-файл через HTTP API.
- Распознает речь через WhisperX.
- Выравнивает слова по времени через alignment-модель WhisperX.
- Определяет спикеров через pyannote diarization.
- Формирует текст диалога со спикерами.
- Отправляет текст в LLM и возвращает краткую суммаризацию.

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

Токен должен принадлежать тому же аккаунту Hugging Face, на котором приняты условия pyannote-моделей. Без этого сервис не сможет скачать модели диаризации.

## Быстрый Запуск Через Docker

1. Создайте файл `.env` в корне проекта:

```env
API_KEY=ваш_api_ключ_для_llm
HF_TOKEN=ваш_huggingface_token
```

2. Проверьте `config.yaml`:

```yaml
whisper:
  model_size: medium
  device: cpu
  compute_type: float32
  language: ru
  initial_prompt: >-
    В разговоре может встречаться компания Нетикс. Возможные термины:
    GPON, ONT, роутер, биллинг, лицевой счет, техподдержка.
  hf_token: null
  num_speakers: 2
  min_speakers: 1
  max_speakers: 2
```

Для CPU оставьте `device: cpu` и `compute_type: float32`. Для GPU обычно используют `device: cuda` и `compute_type: float16`, но Dockerfile в этой ветке собран на `python:3.11-slim` и не настраивает CUDA-зависимости.

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

Первый запуск может занять несколько минут: WhisperX и pyannote скачивают модели в Docker volume `model-cache`.

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
sudo apt-get install -y ffmpeg libsndfile1
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

Параметры WhisperX:

- `model_size`: размер Whisper-модели, например `small`, `medium`, `large-v2`, `large-v3`.
- `device`: устройство для инференса, `cpu` или `cuda`.
- `compute_type`: тип вычислений, например `float32`, `float16`, `int8`.
- `language`: язык аудио, для русских звонков `ru`.
- `initial_prompt`: контекстная подсказка для WhisperX. Помогает модели чаще выбирать правильные названия, аббревиатуры и технические термины, но не является жестким словарем замен.
- `num_speakers`: точное число спикеров, если известно.
- `min_speakers`: минимальное число спикеров для pyannote.
- `max_speakers`: максимальное число спикеров для pyannote.

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
  "transcription": "[SPEAKER_00] текст реплики\n[SPEAKER_01] текст реплики",
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
