import os
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class PromptConfig(BaseModel):
    default: str = "Сейчас будет представлен диалог техподдержки компании Нетикс с клиентом." \
        "Tвоя задача собрать целый диалог, суммаризировать диалог в 3-4 предложениях, " \
        "а также дать эмоциональную окраску, был ли клиент грубым или нет. Предоставь ответ тремя блоками: " \
        "название в одно предложение, суммаризация, оценка емоциональной окраски. Не используй эмоджи, пиши строго и формально"
    summary_short: str = "Создай краткое резюме (2-3 предложения):"
    summary_detailed: str = "Создай подробное резюме текста, выделив ключевые моменты:"


class WhisperConfig(BaseModel):
    model_size: str = "medium"
    device: str = "cpu"
    compute_type: str = "float32"
    language: str = "ru"
    hf_token: Optional[str] = None
    num_speakers: int = 2
    min_speakers: int = 1
    max_speakers: int = 2


class OpenAIConfig(BaseModel):
    api_key: Optional[str] = None
    base_url: str = "https://api.aitunnel.ru/v1/"
    model: str = "claude-haiku-4.5"
    temperature: float = 0.3
    max_tokens: int = 1000


class Settings(BaseModel):
    whisper: WhisperConfig = WhisperConfig()
    openai: OpenAIConfig = OpenAIConfig()
    prompts: PromptConfig = PromptConfig()

    @property
    def openai_api_key(self) -> Optional[str]:
        return os.environ.get("API_KEY", self.openai.api_key)

    @property
    def hf_token(self) -> Optional[str]:
        return os.environ.get("HF_TOKEN", self.whisper.hf_token)


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()