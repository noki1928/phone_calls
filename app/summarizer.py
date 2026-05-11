from openai import OpenAI
from typing import Optional

from app.config import settings


class SummarizerService:
    def __init__(self):
        api_key = settings.openai_api_key
        base_url = settings.openai.base_url
        if not api_key:
            raise ValueError("API_KEY is not set")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = settings.openai.model
        self.temperature = settings.openai.temperature
        self.max_tokens = settings.openai.max_tokens
        self.system_prompt = settings.prompts.default

    def get_system_prompt(self) -> str:
        return self.system_prompt

    def set_system_prompt(self, system_prompt: str) -> str:
        if not system_prompt.strip():
            raise ValueError("System prompt must not be empty")

        self.system_prompt = system_prompt
        return self.system_prompt

    def summarize(self, text: str, custom_prompt: Optional[str] = None) -> str:
        system_prompt = custom_prompt or self.system_prompt

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content
