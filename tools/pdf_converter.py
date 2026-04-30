"""
tools/pdf_converter.py
───────────────────────
Converts a LaTeX .tex file to PDF using system pdflatex.

Node 7 (pdf_exporter) calls compile_latex_to_pdf() after embedding all images.

Requirements:
  System: pdflatex must be installed
    - Ubuntu/Debian: sudo apt install texlive-full
    - macOS: brew install --cask mactex
    - Windows: Install MiKTeX

  Fallback: If pdflatex is not found, the .tex file is still saved and
  the user is given instructions to compile manually or use Overleaf.

Exports:
  compile_latex_to_pdf(latex_code, output_dir, filename) -> str
"""

import os
import subprocess
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compile_latex_to_pdf(
    latex_code: str,
    output_dir: str,
    filename: str = "research_paper"
) -> str:
    """
    Writes the LaTeX code to a .tex file, then compiles it to PDF
    using pdflatex (runs twice for proper cross-references).

    Parameters
    ----------
    latex_code : str
        The complete .tex file content.
    output_dir : str
        Directory where .tex and .pdf will be saved.
    filename : str
        Base filename (without extension). Default "research_paper".

    Returns
    -------
    str
        Absolute path to the generated PDF file.
        If pdflatex is unavailable, returns the path to the .tex file.

    Notes
    -----
    pdflatex is run TWICE intentionally:
      - First run: generates .aux, .toc, .bbl files
      - Second run: resolves all cross-references correctly
    """
    output_path = Path(output_dir).resolve()  # Always use absolute paths
    output_path.mkdir(parents=True, exist_ok=True)

    tex_path = output_path / f"{filename}.tex"
    pdf_path = output_path / f"{filename}.pdf"

    # ── Step 1: Write the .tex file ──────────────────────────────────────────
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    logger.info(f"📄 LaTeX file saved: {tex_path}")

    # ── Step 2: Check if pdflatex is available ───────────────────────────────
    if not shutil.which("pdflatex"):
        logger.warning(
            "pdflatex not found on system PATH. "
            "Returning .tex file path instead of PDF.\n"
            "To compile manually:\n"
            "  1. Install TeX Live: sudo apt install texlive-full\n"
            "  2. Run: pdflatex research_paper.tex\n"
            "  3. Or upload research_paper.tex to https://overleaf.com"
        )
        return str(tex_path)

    # ── Step 3: Compile with pdflatex (run twice) ────────────────────────────
    # Use just the filename since cwd is set to output_path.
    # This ensures pdflatex resolves relative image paths (e.g., images/fig_01.png)
    # correctly from the output directory.
    compile_command = [
        "pdflatex",
        "-interaction=nonstopmode",   # Don't stop on non-fatal errors
        f"{filename}.tex"             # Just filename — cwd handles the rest
    ]

    for run_number in range(1, 3):  # Run 1 and 2
        logger.info(f"🔨 pdflatex run {run_number}/2...")

        try:
            result = subprocess.run(
                compile_command,
                capture_output=True,
                text=True,
                timeout=120,              # 2 minute timeout per run
                cwd=str(output_path)      # Run from output dir so image paths resolve
            )

            if result.returncode != 0:
                # pdflatex exits non-zero on warnings sometimes
                # Only fail if PDF was not produced
                if not pdf_path.exists():
                    error_log = result.stdout[-2000:] if result.stdout else "No output"
                    logger.error(f"pdflatex failed:\n{error_log}")
                    raise RuntimeError(
                        f"pdflatex compilation failed on run {run_number}.\n"
                        f"Check the .tex file at: {tex_path}\n"
                        f"Last output:\n{error_log}"
                    )

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "pdflatex timed out after 120 seconds. "
                "The LaTeX file may have infinite loops or missing packages."
            )

    # ── Step 4: Verify PDF was created ───────────────────────────────────────
    if not pdf_path.exists():
        logger.warning("PDF not found after compilation. Returning .tex path.")
        return str(tex_path)

    logger.info(f"✅ PDF successfully compiled: {pdf_path}")

    # ── Step 5: Clean up auxiliary files ─────────────────────────────────────
    _cleanup_latex_aux_files(output_path, filename)

    return str(pdf_path)


def _cleanup_latex_aux_files(output_dir: Path, basename: str) -> None:
    """
    Removes intermediate LaTeX files (.aux, .log, .toc, etc.)
    Keeps only the .tex source and .pdf output.
    """
    cleanup_extensions = [".aux", ".log", ".toc", ".out", ".bbl", ".blg", ".lof", ".lot"]

    for ext in cleanup_extensions:
        aux_file = output_dir / f"{basename}{ext}"
        if aux_file.exists():
            try:
                aux_file.unlink()
            except OSError:
                pass  # Non-critical — don't fail on cleanup
