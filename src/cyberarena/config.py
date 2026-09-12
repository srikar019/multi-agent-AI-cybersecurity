"""Configuration loaded from environment / .env file."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


OPENAI_API_KEY = _get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = _get("GOOGLE_API_KEY")
LLM_PROVIDER = (_get("LLM_PROVIDER") or "").strip().lower()
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = _get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.2") or 0.2)
MAX_RED_STEPS = int(_get("MAX_RED_STEPS", "15") or 15)


def resolve_provider() -> str:
    """openai > anthropic > gemini > mock, unless LLM_PROVIDER is set explicitly."""
    if LLM_PROVIDER in {"openai", "anthropic", "gemini", "mock"}:
        return LLM_PROVIDER
    if OPENAI_API_KEY:
        return "openai"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if GOOGLE_API_KEY:
        return "gemini"
    return "mock"