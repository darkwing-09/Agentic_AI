"""
tools/gemini_image.py
──────────────────────
Google Gemini / Imagen 3 image generation for Node 6.

The user requested MCP-server-style integration with their Gemini key.
This module wraps the Gemini REST API directly (which is what an MCP server
would proxy). Same key, same endpoint — just no extra layer.

Supported diagram types (Node 6 picks the right one per section):
  - flowchart          → system architecture / pipeline overview
  - bar_chart          → model performance comparison
  - pie_chart          → dataset / class distribution
  - line_graph         → training curves / performance over epochs
  - heatmap            → confusion matrix / attention weights
  - scatter_plot       → data distribution / clustering
  - block_diagram      → model architecture (encoder-decoder, etc.)
  - table_visualization → formatted comparison table as image
  - gantt_chart        → experimental timeline

Exports:
  generate_diagram_with_gemini(prompt, output_path, diagram_type, api_key) -> str
  generate_diagram_prompt(diagram_type, section_content, topic) -> str
"""

import os
import base64
import logging
from typing import Optional
import requests
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Gemini Imagen 3 API endpoint ─────────────────────────────────────────────
IMAGEN_ENDPOINT = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/imagen-3.0-generate-001:predict"
)

# Fallback: Use gemini-2.0-flash-exp for image generation if Imagen unavailable
GEMINI_IMAGE_ENDPOINT = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-2.0-flash-exp:generateContent"
)


def generate_diagram_with_gemini(
    prompt: str,
    output_path: str,
    diagram_type: str,
    api_key: Optional[str] = None
) -> str:
    """
    Generates a diagram image using Gemini Imagen 3 and saves it as PNG.

    Parameters
    ----------
    prompt : str
        Detailed generation prompt. Should describe:
        - What type of diagram (bar chart, flowchart, etc.)
        - What data/content to visualize
        - Style: "academic, clean, professional, white background, high contrast"
    output_path : str
        Full path where the PNG will be saved.
        e.g. "./output/images/fig_accuracy_comparison.png"
    diagram_type : str
        e.g. "bar_chart", "flowchart" (used for logging/display only)
    api_key : str, optional
        Gemini API key. Falls back to GEMINI_API_KEY env var.

    Returns
    -------
    str
        The output_path where the image was saved.

    Raises
    ------
    RuntimeError
        If image generation fails and no fallback succeeds.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. "
            "Set it in .env file: GEMINI_API_KEY=AIzaSy..."
        )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Enhance prompt with academic style requirements
    full_prompt = (
        f"Create a professional academic research paper diagram. "
        f"Type: {diagram_type}. "
        f"Content: {prompt}. "
        f"Style requirements: clean white background, high contrast colors, "
        f"clear labels with readable font size 14pt minimum, "
        f"IEEE publication quality, no watermarks, no decorative elements, "
        f"suitable for black-and-white printing."
    )

    logger.info(f"Generating {diagram_type} with Gemini Imagen 3...")

    # Try Imagen 3 first
    try:
        image_bytes = _call_imagen3(full_prompt, key)
        _save_image(image_bytes, output_path)
        logger.info(f"✅ Image saved: {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"Imagen 3 failed: {e}. Trying Gemini Flash...")

    # Fallback: Gemini 2.0 Flash with image generation
    try:
        image_bytes = _call_gemini_flash_image(full_prompt, key)
        _save_image(image_bytes, output_path)
        logger.info(f"✅ Image saved via Gemini Flash: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"All Gemini image generation methods failed: {e}")
        raise RuntimeError(
            f"Could not generate {diagram_type} image. "
            f"Error: {str(e)}. "
            f"Please check your GEMINI_API_KEY and ensure you have "
            f"Imagen 3 API access enabled in Google AI Studio."
        )


def _call_imagen3(prompt: str, api_key: str) -> bytes:
    """Calls the Imagen 3 REST endpoint. Returns raw image bytes."""
    params = {"key": api_key}
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "16:9",        # Landscape for paper figures
            "safetyFilterLevel": "block_only_high",
            "personGeneration": "dont_allow"
        }
    }

    response = requests.post(
        IMAGEN_ENDPOINT,
        json=payload,
        params=params,
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    response.raise_for_status()

    data = response.json()
    b64_image = data["predictions"][0]["bytesBase64Encoded"]
    return base64.b64decode(b64_image)


def _call_gemini_flash_image(prompt: str, api_key: str) -> bytes:
    """Fallback: Uses Gemini 2.0 Flash Experimental for image generation."""
    params = {"key": api_key}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]}
    }

    response = requests.post(
        GEMINI_IMAGE_ENDPOINT,
        json=payload,
        params=params,
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    response.raise_for_status()

    data = response.json()
    # Navigate to the inline image data
    parts = data["candidates"][0]["content"]["parts"]
    for part in parts:
        if "inlineData" in part:
            b64_image = part["inlineData"]["data"]
            return base64.b64decode(b64_image)

    raise ValueError("No image data found in Gemini Flash response.")


def _save_image(image_bytes: bytes, output_path: str) -> None:
    """Saves raw image bytes to the given path."""
    with open(output_path, "wb") as f:
        f.write(image_bytes)


def generate_diagram_prompt(
    diagram_type: str,
    section_content: str,
    topic: str
) -> str:
    """
    Generates the optimal Gemini prompt for a given diagram type.

    Called by Node 6 when building the diagram_plan.

    Parameters
    ----------
    diagram_type : str
        One of the supported types (flowchart, bar_chart, etc.)
    section_content : str
        The section text this diagram will visualize (first 300 chars).
    topic : str
        The paper's main topic.

    Returns
    -------
    str
        A detailed, specific generation prompt.
    """
    content_snippet = section_content[:300] if section_content else topic

    prompts = {
        "flowchart": (
            f"A detailed system architecture flowchart showing the pipeline for: {topic}. "
            f"Context: {content_snippet}. "
            f"Include rectangular process boxes, diamond decision nodes, "
            f"arrows showing data flow direction, labeled connections."
        ),
        "bar_chart": (
            f"A grouped bar chart comparing performance metrics across different methods "
            f"for: {topic}. Context: {content_snippet}. "
            f"X-axis: method names, Y-axis: metric value (0-100%), "
            f"different colors per metric group, value labels on bars."
        ),
        "pie_chart": (
            f"A clean pie chart showing distribution/composition relevant to: {topic}. "
            f"Context: {content_snippet}. "
            f"Include percentage labels, legend, distinct colors per slice."
        ),
        "line_graph": (
            f"A multi-line graph showing performance trends over training steps/epochs "
            f"for: {topic}. Context: {content_snippet}. "
            f"X-axis: epochs/steps, Y-axis: accuracy/loss, "
            f"one line per model/method, legend included."
        ),
        "heatmap": (
            f"A confusion matrix heatmap or correlation heatmap for: {topic}. "
            f"Context: {content_snippet}. "
            f"Color scale from white (low) to dark blue (high), "
            f"labeled axes, numerical values in cells."
        ),
        "scatter_plot": (
            f"A scatter plot showing data point distribution or clustering for: {topic}. "
            f"Context: {content_snippet}. "
            f"Different colors per cluster/class, axis labels, "
            f"legend explaining each color group."
        ),
        "block_diagram": (
            f"A clean block diagram showing the model architecture or system components "
            f"for: {topic}. Context: {content_snippet}. "
            f"Rectangular blocks with labels, connecting arrows, "
            f"color-coded by component type (encoder/decoder/attention)."
        ),
        "table_visualization": (
            f"A formatted results comparison table for: {topic}. "
            f"Context: {content_snippet}. "
            f"Clean grid lines, header row highlighted, best results bolded, "
            f"rows=methods, columns=metrics."
        ),
    }

    return prompts.get(diagram_type, (
        f"A professional academic diagram showing key concepts of: {topic}. "
        f"Context: {content_snippet}. Clean, labeled, publication-ready."
    ))
