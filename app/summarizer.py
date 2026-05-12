"""OpenAI-compatible dialog summarization service."""

from openai import OpenAI

from app.config import settings


class SummarizerService:
    """Manage summarization prompts, model settings, and chat completions."""

    def __init__(self):
        """Initialize the OpenAI-compatible client from application settings."""

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
        """Return the active system prompt."""

        return self.system_prompt

    def set_system_prompt(self, system_prompt: str) -> str:
        """Validate and update the active system prompt."""

        if not system_prompt.strip():
            raise ValueError("System prompt must not be empty")

        self.system_prompt = system_prompt
        return self.system_prompt

    def get_model(self) -> str:
        """Return the active summarization model name."""

        return self.model

    def set_model(self, model: str) -> str:
        """Validate and update the active summarization model name."""

        if not model.strip():
            raise ValueError("Model must not be empty")

        self.model = model.strip()
        return self.model

    def summarize(self, text: str) -> str:
        """Summarize transcribed dialog text using the configured prompt."""

        system_prompt = self.system_prompt
        user_prompt = (
            "Выполни суммаризацию строго по системной инструкции. "
            "Не добавляй Markdown, списки, заголовки с # и пояснения вне заданного формата.\n\n"
            f"Диалог:\n{text}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content
