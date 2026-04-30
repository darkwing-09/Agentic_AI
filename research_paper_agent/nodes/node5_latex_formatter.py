"""
nodes/node5_latex_formatter.py
───────────────────────────────
NODE 5 — LaTeX Formatter

PURPOSE:
  Takes the humanized content and fits it into the IEEE Conference (IEEEtran)
  LaTeX template. Produces a complete, compilable .tex file WITH placeholder
  comments where images will later be inserted by Node 7.

  THREE STEPS:
  1. Convert prose text → LaTeX-safe text (escape special chars)
  2. Build section content (convert [FIGURE: ...] hints → placeholder comments)
  3. Fill the IEEEtran template using fill_latex_template()

READS FROM STATE:
  - humanized_content
  - paper_title
  - paper_keywords

WRITES TO STATE:
  - latex_code
"""

import re
import os
import logging
from pathlib import Path

from state import PaperState
from templates.latex_template import fill_latex_template, DEFAULT_LATEX, REQUIRED_SECTIONS

logger = logging.getLogger(__name__)


def node5_latex_formatter(state: PaperState) -> dict:
    """
    LangGraph node: Converts humanized content into a full LaTeX document.

    Parameters
    ----------
    state : PaperState
        Reads: humanized_content, paper_title, paper_keywords, output_dir

    Returns
    -------
    dict
        State updates: latex_code
    """
    humanized_content = state.get("humanized_content", {})
    paper_title       = state.get("paper_title", "Research Paper")
    paper_keywords    = state.get("paper_keywords", [])
    output_dir        = state.get("output_dir", "./output")

    if not humanized_content:
        logger.error("[Node 5] No humanized content received!")
        return {
            "latex_code": "",
            "error_message": "Node 5: no humanized content to format."
        }

    logger.info("[Node 5] Converting sections to LaTeX-safe format...")

    # ── Step 1: Convert each section to LaTeX-safe text ──────────────────────
    latex_sections = {}

    for section_key in REQUIRED_SECTIONS:
        raw_text = humanized_content.get(section_key, "")
        if not raw_text:
            latex_sections[section_key] = f"% Section {section_key} not provided."
            continue

        # Convert plain text to LaTeX-formatted text
        latex_text = _convert_to_latex(raw_text, section_key)
        latex_sections[section_key] = latex_text

        logger.info(f"[Node 5] Processed section: {section_key} ({len(latex_text)} chars)")

    # ── Step 2: Fill the template ─────────────────────────────────────────────
    logger.info("[Node 5] Filling IEEEtran LaTeX template...")

    try:
        latex_code = fill_latex_template(
            sections=latex_sections,
            title=paper_title,
            keywords=paper_keywords
        )
    except Exception as e:
        logger.error(f"[Node 5] Template fill failed: {e}")
        return {
            "latex_code": "",
            "error_message": f"Node 5: template fill error: {str(e)}"
        }

    # ── Step 3: Save the .tex file (pre-images version) ──────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tex_draft_path = output_path / "paper_draft.tex"
    try:
        with open(tex_draft_path, "w", encoding="utf-8") as f:
            f.write(latex_code)
        logger.info(f"[Node 5] Draft .tex saved: {tex_draft_path}")
    except Exception as e:
        logger.warning(f"[Node 5] Could not save draft: {e}")

    # Count figure placeholders for Node 6
    placeholder_count = latex_code.count("% FIGURE_PLACEHOLDER:")
    logger.info(f"[Node 5] LaTeX ready. {placeholder_count} figure placeholders found.")
    logger.info(f"[Node 5] Total LaTeX size: {len(latex_code)} chars")

    return {"latex_code": latex_code}


# ─────────────────────────────────────────────────────────────────────────────
# Internal conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _convert_to_latex(text: str, section_key: str) -> str:
    """
    Converts plain humanized text into LaTeX-formatted text.

    Handles:
    - Escaping special chars that would break LaTeX
    - Converting [FIGURE: description] hints → % FIGURE comments
    - Converting **bold** markdown → \\textbf{}
    - Converting *italic* markdown → \\textit{}
    - Converting numbered lists → LaTeX enumerate
    - Converting bullet lists → LaTeX itemize
    - Preserving paragraph structure with blank lines
    - Preserving inline math: $...$ stays as-is
    - Preserving block equations: \\begin{equation}...\\end{equation}
    """
    if not text:
        return ""

    # References section: special handling
    if section_key == "references":
        return _convert_references_to_latex(text)

    # ── Remove [FIGURE: ...] inline suggestions ──────────────────────────────
    # Node 6 will plan the actual figures; the template has FIGURE_PLACEHOLDER
    text = re.sub(r'\[FIGURE:\s*[^\]]+\]', '', text)
    text = re.sub(r'\[Figure:\s*[^\]]+\]', '', text)

    # ── Preserve LaTeX commands and math before escaping ─────────────────────
    # We protect all LaTeX commands so the escape step doesn't destroy them.
    protected_blocks = {}
    protect_counter = [0]

    def save_protected(match):
        placeholder = f"PROTECTEDBLOCK{protect_counter[0]}END"
        protected_blocks[placeholder] = match.group(0)
        protect_counter[0] += 1
        return placeholder

    # Protect \cite{...} commands (from citation injector in Node 4)
    text = re.sub(r'\\cite\{[^}]+\}', save_protected, text)

    # Protect \textbf{...} and \textit{...}
    text = re.sub(r'\\text(?:bf|it)\{[^}]+\}', save_protected, text)

    # Protect \bibitem{...}
    text = re.sub(r'\\bibitem\{[^}]+\}', save_protected, text)

    # Protect \begin{...}...\end{...} environments
    text = re.sub(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', save_protected, text, flags=re.DOTALL)

    # Protect display math \[...\]
    text = re.sub(r'\\\[.*?\\\]', save_protected, text, flags=re.DOTALL)

    # Protect inline math $...$
    text = re.sub(r'\$[^$]+\$', save_protected, text)

    # ── Escape LaTeX special characters (outside of protected blocks) ────────
    text = _escape_special_chars(text)

    # ── Convert markdown bold/italic → LaTeX ─────────────────────────────────
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'\*(.+?)\*',     r'\\textit{\1}', text)
    text = re.sub(r'__(.+?)__',     r'\\textbf{\1}', text)
    text = re.sub(r'_([^_]+)_',     r'\\textit{\1}', text)

    # ── Convert markdown headings → LaTeX subsubsections ─────────────────────
    text = re.sub(r'^### (.+)$', r'\\subsubsection{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',  r'\\subsubsection{\1}', text, flags=re.MULTILINE)

    # ── Convert bullet lists → LaTeX itemize ─────────────────────────────────
    text = _convert_bullet_list(text)

    # ── Convert numbered lists → LaTeX enumerate ─────────────────────────────
    text = _convert_numbered_list(text)

    # ── Ensure proper paragraph spacing ──────────────────────────────────────
    text = re.sub(r'\n{3,}', '\n\n', text)

    # ── Restore protected LaTeX blocks ───────────────────────────────────────
    for placeholder, original in protected_blocks.items():
        text = text.replace(placeholder, original)

    return text.strip()


def _escape_special_chars(text: str) -> str:
    """
    Escapes LaTeX-special characters in plain text.
    DOES NOT escape inside MATHBLOCK...END placeholders.
    Must be called AFTER math is extracted.
    """
    replacements = [
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("#",  r"\#"),
        ("_",  r"\_"),       # Fixed: now escaping underscores properly
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
        ("<",  r"\textless{}"),
        (">",  r"\textgreater{}"),
        ("|",  r"\textbar{}"),
    ]

    for char, escaped in replacements:
        text = text.replace(char, escaped)

    return text


def _convert_references_to_latex(text: str) -> str:
    """
    Converts a references section to LaTeX \\bibitem format.

    Handles two input formats:
    1. Already IEEE format: [1] Author... → \\bibitem{b1}...
    2. Already \\bibitem format: pass through
    3. Plain text list: just normalizes spacing
    """
    # If already contains \bibitem, pass through
    if r"\bibitem" in text:
        return text

    lines = text.strip().split("\n")
    output_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if it's a numbered reference [1], [2], etc.
        match = re.match(r'^\[(\d+)\]\s*(.+)$', line)
        if match:
            ref_num  = match.group(1)
            ref_text = match.group(2)
            output_lines.append(f"\\bibitem{{b{ref_num}}}")
            output_lines.append(f"{ref_text}\n")
        else:
            output_lines.append(line)

    return "\n".join(output_lines)


def _convert_bullet_list(text: str) -> str:
    """
    Converts Markdown bullet lists (- item or * item) to LaTeX itemize.
    Groups consecutive bullet lines into a single environment.
    """
    lines = text.split("\n")
    result = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        is_bullet = stripped.startswith("- ") or stripped.startswith("* ")

        if is_bullet and not in_list:
            result.append("\\begin{itemize}")
            in_list = True

        if is_bullet:
            item_text = stripped[2:].strip()
            result.append(f"  \\item {item_text}")
        else:
            if in_list:
                result.append("\\end{itemize}")
                in_list = False
            result.append(line)

    if in_list:
        result.append("\\end{itemize}")

    return "\n".join(result)


def _convert_numbered_list(text: str) -> str:
    """
    Converts Markdown numbered lists (1. item) to LaTeX enumerate.
    Groups consecutive numbered lines into a single environment.
    """
    lines = text.split("\n")
    result = []
    in_list = False

    numbered_pattern = re.compile(r'^\d+\.\s+(.+)$')

    for line in lines:
        stripped = line.strip()
        match = numbered_pattern.match(stripped)

        if match and not in_list:
            result.append("\\begin{enumerate}")
            in_list = True

        if match:
            item_text = match.group(1)
            result.append(f"  \\item {item_text}")
        else:
            if in_list:
                result.append("\\end{enumerate}")
                in_list = False
            result.append(line)

    if in_list:
        result.append("\\end{enumerate}")

    return "\n".join(result)
