"""
llm_interface.py

Unified LLM client abstraction supporting Gemini and OpenAI.

This module provides a single interface for all LLM calls in the system,
abstracting away the provider-specific API differences.

Usage:
    from reasoning.llm_interface import LLMClient
    llm = LLMClient()
    
    # Text-only
    response = llm.generate("Describe this scene", system="You are a vision AI")
    
    # Multimodal (with image)
    response = llm.generate_with_image(jpeg_bytes, "What do you see?")
"""

import os  # type: ignore
import json  # type: ignore
import logging  # type: ignore
from dotenv import load_dotenv  # type: ignore

load_dotenv()
logger = logging.getLogger("reasoning.llm")


class LLMClient:
    """
    Unified LLM interface supporting Gemini and OpenAI providers.

    The provider is auto-detected from environment variables:
    - GEMINI_API_KEY → Gemini
    - OPENAI_API_KEY → OpenAI

    All methods return plain text strings. JSON parsing is left to callers.
    """

    def __init__(self, provider: str | None = None):  # type: ignore
        self._provider = provider or self._detect_provider()
        self._client = None
        self._init_client()
        logger.info(f"[LLM] Initialized with provider: {self._provider}")

    def _detect_provider(self) -> str:  # type: ignore
        """Auto-detect LLM provider from env vars."""
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        logger.warning("[LLM] No API key found — defaulting to gemini")
        return "gemini"

    def _init_client(self):  # type: ignore
        """Initialize the provider-specific client."""
        if self._provider == "gemini":
            try:
                from google import genai  # type: ignore
                self._client = genai.Client(
                    api_key=os.environ.get("GEMINI_API_KEY")
                )
            except Exception as e:
                logger.error(f"[LLM] Gemini init failed: {e}")

        elif self._provider == "openai":
            try:
                import openai  # type: ignore
                self._client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY")
                )
            except Exception as e:
                logger.error(f"[LLM] OpenAI init failed: {e}")

    @property
    def provider(self) -> str:  # type: ignore
        return self._provider

    # ─────────────────────────────────────────────────────────────────────
    # TEXT GENERATION
    # ─────────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 256,
        model: str | None = None,
    ) -> str:  # type: ignore
        """
        Generate text from a prompt.

        Args:
            prompt:      User prompt text
            system:      System instruction
            temperature: Creativity (0.0–1.0)
            max_tokens:  Max output length
            model:       Override default model name

        Returns:
            Generated text string, or empty string on error
        """
        try:
            if self._provider == "gemini":
                return self._gemini_generate(prompt, system, temperature, max_tokens, model)
            elif self._provider == "openai":
                return self._openai_generate(prompt, system, temperature, max_tokens, model)
        except Exception as e:
            logger.error(f"[LLM] Generate error: {e}")
        return ""

    # ─────────────────────────────────────────────────────────────────────
    # MULTIMODAL GENERATION (with image)
    # ─────────────────────────────────────────────────────────────────────

    def generate_with_image(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        model: str | None = None,
        mime_type: str = "image/jpeg",
    ) -> str:  # type: ignore
        """
        Generate text from an image + prompt (Vision LLM).

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            prompt:      Text prompt to accompany the image
            system:      System instruction
            temperature: Creativity
            max_tokens:  Max output length
            model:       Override default model name
            mime_type:   Image MIME type

        Returns:
            Generated text string, or empty string on error
        """
        try:
            if self._provider == "gemini":
                return self._gemini_vision(
                    image_bytes, prompt, system, temperature, max_tokens, model, mime_type
                )
            elif self._provider == "openai":
                return self._openai_vision(
                    image_bytes, prompt, system, temperature, max_tokens, model
                )
        except Exception as e:
            logger.error(f"[LLM] Vision generate error: {e}")
        return ""

    # ─────────────────────────────────────────────────────────────────────
    # GEMINI IMPLEMENTATION
    # ─────────────────────────────────────────────────────────────────────

    def _gemini_generate(
        self, prompt, system, temperature, max_tokens, model
    ) -> str:  # type: ignore
        from google.genai import types as gtypes  # type: ignore
        model_name = model or "gemini-2.5-flash"

        response = self._client.models.generate_content(  # type: ignore
            model=model_name,
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=system or None,
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (response.text or "").strip()  # type: ignore

    def _gemini_vision(
        self, image_bytes, prompt, system, temperature, max_tokens, model, mime_type
    ) -> str:  # type: ignore
        from google.genai import types as gtypes  # type: ignore
        model_name = model or "gemini-2.5-flash"

        response = self._client.models.generate_content(  # type: ignore
            model=model_name,
            contents=gtypes.Content(
                role="user",
                parts=[
                    gtypes.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    gtypes.Part(text=prompt),
                ],
            ),
            config=gtypes.GenerateContentConfig(
                system_instruction=system or None,
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (response.text or "").strip()  # type: ignore

    # ─────────────────────────────────────────────────────────────────────
    # OPENAI IMPLEMENTATION
    # ─────────────────────────────────────────────────────────────────────

    def _openai_generate(
        self, prompt, system, temperature, max_tokens, model
    ) -> str:  # type: ignore
        model_name = model or "gpt-4o-mini"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(  # type: ignore
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()  # type: ignore

    def _openai_vision(
        self, image_bytes, prompt, system, temperature, max_tokens, model
    ) -> str:  # type: ignore
        import base64  # type: ignore
        model_name = model or "gpt-4o-mini"
        b64_img = base64.b64encode(image_bytes).decode()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                },
            ],
        })

        response = self._client.chat.completions.create(  # type: ignore
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()  # type: ignore


# ── Singleton ────────────────────────────────────────────────────────────────

_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:  # type: ignore
    """Get the default LLM client singleton."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client  # type: ignore
