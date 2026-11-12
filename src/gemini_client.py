"""Secure Gemini client for the AI Electricity Bill Analyzer."""

from functools import lru_cache

from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL


def _validate_configuration() -> None:
    """Validate required Gemini settings without exposing secret values."""
    if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
        raise RuntimeError(
            "Gemini API key is not configured. Add GEMINI_API_KEY to the "
            "project-root .env file."
        )

    if not GEMINI_MODEL or not GEMINI_MODEL.strip():
        raise RuntimeError(
            "Gemini model is not configured. Set GEMINI_MODEL in config.py "
            "or the project-root .env file."
        )


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Create and reuse one Gemini client for this application process."""
    _validate_configuration()
    return genai.Client(api_key=GEMINI_API_KEY)


def generate_response(prompt: str) -> str:
    """Generate a text response from Gemini for a non-empty prompt."""
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    client = get_gemini_client()

    # Let SDK/API exceptions propagate so the calling application can
    # distinguish authentication, quota, availability, and network failures.
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    response_text = getattr(response, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError(
            "Gemini returned no text. The response may have been blocked "
            "or may not contain a text candidate."
        )

    return response_text.strip()
