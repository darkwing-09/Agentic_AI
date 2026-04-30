"""
tools/llm_client.py
────────────────────
Central wrapper for all OpenAI GPT-4o-mini calls made by the agent.

WHY a central wrapper?
  - Single place to change model, temperature, or provider
  - Consistent error handling and retry logic
  - Easy to mock in tests
  - All nodes import from here — zero duplication

Exports:
  call_llm(system, user, temperature) -> str
  call_llm_structured(system, user, json_schema) -> dict
"""

import os
import json
import time
import logging
from typing import Optional

from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv

load_dotenv()

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Client ────────────────────────────────────────────────────────────────────
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazy singleton — only creates client once."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found. "
                "Copy .env.example to .env and fill in your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ── Core Model Constant ───────────────────────────────────────────────────────
# User specified "gpt-mini4" → this is GPT-4o-mini
LLM_MODEL = "gpt-4o-mini"

# Max retries on rate limit
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 4096
) -> str:
    """
    Basic LLM call → returns raw text response.

    Used by: Node 1, 2, 3, 4, 5, 6 for all generation tasks.

    Parameters
    ----------
    system_prompt : str
        Sets the role/task for the model.
    user_prompt : str
        The actual content/question.
    temperature : float
        0.0 = deterministic, 1.0 = creative. Default 0.7.
    max_tokens : int
        Max response length. Default 4096 (covers full sections).

    Returns
    -------
    str
        The model's text response, stripped of whitespace.

    Raises
    ------
    RuntimeError
        If all retries fail.
    """
    client = _get_client()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content
            return text.strip() if text else ""

        except RateLimitError:
            # Rate limited → wait and retry
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning(f"Rate limited. Waiting {wait}s before retry {attempt+1}...")
                time.sleep(wait)
            else:
                raise RuntimeError("OpenAI rate limit exceeded after all retries.")

        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {str(e)}")


def call_llm_structured(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 6000        # Increased: full paper JSON needs room
) -> dict:
    """
    LLM call where the response MUST be valid JSON.

    Used by: Node 1 (query analysis), Node 6 (diagram planning).

    The system_prompt should instruct the model to return ONLY JSON,
    no markdown fences, no explanation.

    Parameters
    ----------
    system_prompt : str
        Must include instruction: "Respond with ONLY valid JSON. No markdown."
    user_prompt : str
        The task prompt.
    temperature : float
        Lower temperature for structured output. Default 0.3.

    Returns
    -------
    dict
        Parsed JSON response.

    Raises
    ------
    ValueError
        If response cannot be parsed as JSON.
    """
    client = _get_client()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},  # Forces JSON output
            )
            raw = response.choices[0].message.content or "{}"

            # Strip any accidental markdown fences
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            return json.loads(raw)

        except RateLimitError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise RuntimeError("OpenAI rate limit exceeded.")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parse error on attempt {attempt+1}: {e}")
            if attempt == MAX_RETRIES - 1:
                raise ValueError(
                    f"Model returned invalid JSON after {MAX_RETRIES} attempts: {str(e)}"
                )
            time.sleep(1)

        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
