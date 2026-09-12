"""LLM factory: returns a langchain chat model for the configured provider.

Providers:
  - openai     -> langchain_openai.ChatOpenAI
  - anthropic  -> langchain_anthropic.ChatAnthropic
  - gemini     -> langchain_google_genai.ChatGoogleGenerativeAI
  - mock       -> deterministic MockChatModel (no API key, used for tests/CI)

The mock inspects the system prompt to decide what to return, which keeps
the whole graph runnable offline for wiring tests.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from . import config

# --- free-tier rate limiting -------------------------------------------------
# gemini-*-flash-lite free tier allows 15 requests/minute. Agents fire calls
# back-to-back, so we enforce a minimum spacing between request starts and
# retry with backoff when a 429 still slips through.

_RATE_LOCK = threading.Lock()
_LAST_REQUEST = {"ts": 0.0}


def _throttle(min_interval: float) -> None:
    with _RATE_LOCK:
        wait = _LAST_REQUEST["ts"] + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST["ts"] = time.monotonic()


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


class MockChatModel(BaseChatModel):
    """Deterministic stand-in chat model for offline smoke tests.

    Returns canned content based on keywords in the system prompt so each
    pipeline node produces schema-shaped output.
    """

    model: str = "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = "\n".join(
            m.content or "" for m in messages if getattr(m, "type", "") == "system"
        ).lower()

        if "triage" in system:
            content = json.dumps(
                {
                    "is_incident": True,
                    "confidence": 0.85,
                    "focus_areas": ["authentication"],
                    "rationale": "mock triage decision",
                }
            )
        elif "findings" in system or "investigation" in system:
            content = json.dumps([])
        elif "incident report" in system or "report" in system:
            content = "# Mock Incident Report\n\nCanned output from the mock model. "
            "No findings were produced (offline test mode)."
        else:
            content = "mock response"
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
        """Expose tool names as kwargs so create_react_agent wiring works."""
        names = [getattr(t, "name", None) or str(t) for t in tools]
        return self.bind(tools=names, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "mock-chat"


def get_chat_model(
    provider: str | None = None, temperature: float | None = None
) -> BaseChatModel:
    provider = (provider or config.resolve_provider()).lower()
    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE

    if provider == "mock":
        return MockChatModel()

    if provider == "openai":
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set (provider=openai)")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.OPENAI_MODEL, temperature=temperature)

    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (provider=anthropic)")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.ANTHROPIC_MODEL, temperature=temperature)

    if provider == "gemini":
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set (provider=gemini)")
        from langchain_google_genai import ChatGoogleGenerativeAI

        class ThrottledGemini(ChatGoogleGenerativeAI):
            """ChatGoogleGenerativeAI with free-tier RPM throttling + 429 retry.

            The Google SDK retries transient errors internally, but gives up
            quickly on quota 429s; batch runs otherwise die mid-evaluation.
            """

            min_interval: float = 4.2  # ~14.3 req/min < 15 req/min free tier
            max_quota_retries: int = 3

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                for attempt in range(self.max_quota_retries + 1):
                    _throttle(self.min_interval)
                    try:
                        return super()._generate(
                            messages, stop=stop, run_manager=run_manager, **kwargs
                        )
                    except Exception as exc:
                        if not _is_quota_error(exc) or attempt >= self.max_quota_retries:
                            raise
                        time.sleep(45.0 * (attempt + 1))

        # Note: the Google SDK retries transient 429/rate-limit errors internally.
        return ThrottledGemini(
            model=config.GEMINI_MODEL,
            temperature=temperature,
            api_key=config.GOOGLE_API_KEY,
        )

    raise ValueError(f"unknown LLM provider: {provider}")