"""Shared chat-model configuration."""

import inspect
import os
from pathlib import Path
from typing import Any, ClassVar

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

IS_GITHUB_ENV = any(
    os.getenv(name, "").lower() == "true"
    for name in ("GITHUB_ACTIONS", "CODESPACES")
)

if not IS_GITHUB_ENV:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


class ModelProvider:
    """Base configuration for one model provider."""

    model_class: ClassVar[type[BaseChatModel]]
    api_key_env: ClassVar[str]
    models: ClassVar[tuple[str, ...]]
    options: ClassVar[dict[str, Any]] = {}

    @classmethod
    def create(cls, model_name: str | None = None, **options: Any) -> BaseChatModel:
        model_name = model_name or cls.models[-1]
        if model_name not in cls.models:
            choices = ", ".join(cls.models)
            raise ValueError(f"Unknown model {model_name!r}. Choose one of: {choices}.")

        api_key = os.getenv(cls.api_key_env)
        if not api_key:
            raise ValueError(f"{cls.api_key_env} is not set.")

        params = inspect.signature(cls.model_class).parameters
        kwargs = dict(cls.options)
        kwargs.update(options)

        model_key = next((name for name in ("model", "model_name") if name in params), None)
        if model_key is None:
            raise TypeError(f"{cls.model_class.__name__} does not accept a model name parameter.")
        kwargs[model_key] = model_name

        api_key_key = next(
            (
                name
                for name in (
                    "api_key",
                    "openai_api_key",
                    "anthropic_api_key",
                    "google_api_key",
                    "groq_api_key",
                    "xai_api_key",
                )
                if name in params
            ),
            None,
        )
        if api_key_key is not None:
            kwargs[api_key_key] = api_key

        return cls.model_class(**kwargs)


class OpenAIProvider(ModelProvider):
    model_class = ChatOpenAI
    api_key_env = "OPENAI_API_KEY"
    models = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")


class AnthropicProvider(ModelProvider):
    model_class = ChatAnthropic
    api_key_env = "ANTHROPIC_API_KEY"
    models = (
        "claude-haiku-4-5",
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-fable-5",
    )


class GoogleProvider(ModelProvider):
    model_class = ChatGoogleGenerativeAI
    api_key_env = "GEMINI_API_KEY"
    models = ("gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash")


class GroqProvider(ModelProvider):
    model_class = ChatGroq
    api_key_env = "GROQ_API_KEY"
    models = ("openai/gpt-oss-120b",)


class GrokProvider(ModelProvider):
    model_class = ChatOpenAI
    api_key_env = "GROK_API_KEY"
    models = ("grok-4.3", "grok-4.5")
    options = {"base_url": "https://api.x.ai/v1"}


PROVIDERS: dict[str, type[ModelProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "groq": GroqProvider,
    "grok": GrokProvider,
}

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq").lower()
MODEL_NAME = os.getenv("MODEL_NAME")

try:
    provider = PROVIDERS[MODEL_PROVIDER]
except KeyError as error:
    choices = ", ".join(PROVIDERS)
    raise ValueError(
        f"Unknown MODEL_PROVIDER {MODEL_PROVIDER!r}. Choose one of: {choices}."
    ) from error

model = provider.create(MODEL_NAME)