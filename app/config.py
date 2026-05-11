import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", Path(__file__).resolve().parent.parent / "config.yaml"))


class PromptConfig(BaseModel):
    default: str


class GigaAMConfig(BaseModel):
    model_name: str
    device: str
    pyannote_model: str
    hf_token: Optional[str]
    num_speakers: Optional[int]
    min_speakers: Optional[int]
    max_speakers: Optional[int]
    vad_onset: float
    vad_offset: float
    chunk_size: float
    fr_batch_size: int
    fr_num_workers: int
    merge_gap: float


class OpenAIConfig(BaseModel):
    api_key: Optional[str]
    base_url: str
    model: str
    temperature: float
    max_tokens: int


class Settings(BaseModel):
    gigaam: GigaAMConfig
    openai: OpenAIConfig
    prompts: PromptConfig

    @property
    def openai_api_key(self) -> Optional[str]:
        return os.environ.get("API_KEY", self.openai.api_key)

    @property
    def hf_token(self) -> Optional[str]:
        return os.environ.get("HF_TOKEN", self.gigaam.hf_token)


@lru_cache()
def get_settings() -> Settings:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    return Settings(**config)


settings = get_settings()
