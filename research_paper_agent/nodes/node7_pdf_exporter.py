"""
nodes/node7_pdf_exporter.py
─────────────────────────────
NODE 7 — PDF Exporter (Final Node)

PURPOSE:
  1. Takes the latex_code (with FIGURE_PLACEHOLDER comments)
  2. Replaces each placeholder with a real \includegraphics{} for
     each approved image from Node 6
  3. Compiles the .tex file to PDF via pdflatex
  4. Shows the user the final PDF path and asks: satisfied?

  FINAL HIL:
  - "yes" → END the graph
  - "redo diagrams" → go back to Node 6
  - "start over" → go back to Node 1

READS FROM STATE:
  - latex_code
  - generated_images
  - paper_title
  - output_dir

WRITES TO STATE:
  - final_latex_with_images
  - final_pdf_path
  - user_satisfied
  - restart_from
"""

import os
import logging
from pathlib import Path
from langgraph.types import interrupt

from state import PaperState, GeneratedImage
from templates.latex_template import insert_figure_into_latex, insert_all_figures
from tools.pdf_converter import compile_latex_to_pdf

logger = logging.getLogger(__name__)


def node7_pdf_exporter(state: PaperState) -> dict:
    """
    LangGraph node: Inserts images into LaTeX, compiles PDF, final HIL.

    Parameters
    ----------
    state : PaperState
        Reads: latex_code, generated_images, paper_title, output_dir

    Returns
    -------
    dict
        State updates: final_latex_with_images, final_pdf_path,
                       user_satisfied, restart_from
    """
    latex_code       = state.get("latex_code", "")
    generated_images = state.get("generated_images", [])
    paper_title      = state.get("paper_title", "research_paper")
    output_dir       = state.get("output_dir", "./output")

    if not latex_code:
        logger.error("[Node 7] No LaTeX code received!")
        return {
            "error_message": "Node 7: no LaTeX code available.",
            "user_satisfied": False,
            "restart_from": "node1",
        }

    # ── Step 1: Insert approved images into LaTeX ─────────────────────────────
    approved_images = [img for img in generated_images if img.get("approved", False)]
    logger.info(f"[Node 7] Inserting {len(approved_images)} approved images into LaTeX...")

    final_latex = insert_all_figures(
        latex=latex_code,
        approved_images=approved_images,
        output_dir=output_dir,
    )

    logger.info(f"[Node 7] {len(approved_images)} figures inserted.")

    # ── Step 2: Compile LaTeX to PDF ──────────────────────────────────────────
    logger.info("[Node 7] Compiling LaTeX to PDF...")

    safe_title = _make_safe_filename(paper_title)

    try:
        output_path = compile_latex_to_pdf(
            latex_code=final_latex,
            output_dir=output_dir,
            filename=safe_title
        )
        is_pdf = output_path.endswith(".pdf")
        logger.info(f"[Node 7] Output: {output_path}")
    except Exception as e:
        logger.error(f"[Node 7] PDF compilation error: {e}")
        output_path = str(Path(output_dir) / f"{safe_title}.tex")
        is_pdf = False

    # ── Step 3: Save the final .tex file (with images inserted) ───────────────
    final_tex_path = Path(output_dir) / f"{safe_title}_final.tex"
    try:
        with open(final_tex_path, "w", encoding="utf-8") as f:
            f.write(final_latex)
        logger.info(f"[Node 7] Final .tex saved: {final_tex_path}")
    except Exception as e:
        logger.warning(f"[Node 7] Could not save final .tex: {e}")

    # ── Step 4: Final HIL — ask user if satisfied ─────────────────────────────
    file_type_label = "PDF" if is_pdf else "LaTeX (.tex)"
    file_emoji      = "📄" if is_pdf else "📝"

    if not is_pdf:
        compile_hint = (
            "\n⚠️  pdflatex not found on this system.\n"
            "    To compile to PDF:\n"
            "    Option A: Install TeX Live → sudo apt install texlive-full\n"
            "    Option B: Upload the .tex file to https://overleaf.com\n"
        )
    else:
        compile_hint = ""

    user_response = interrupt({
        "type": "final_review",
        "message": (
            f"\n{'═'*65}\n"
            f"🎓  RESEARCH PAPER COMPLETE!\n"
            f"{'═'*65}\n\n"
            f"📌  Paper: {paper_title}\n"
            f"{file_emoji}  Output: {output_path}\n"
            f"📊  Figures: {len(approved_images)} diagrams embedded\n"
            f"{compile_hint}\n"
            f"{'─'*65}\n"
            f"  What would you like to do?\n\n"
            f"  • 'yes' / 'done'         → All good! End here.\n"
            f"  • 'redo diagrams'         → Regenerate images only (Node 6)\n"
            f"  • 'start over'            → Start fresh from the beginning\n"
            f"  • 'fix content'           → Back to judge review (Node 3)\n"
            f"{'═'*65}"
        ),
        "output_path":  output_path,
        "is_pdf":       is_pdf,
        "figures_count": len(approved_images),
        "paper_title":  paper_title,
    })

    # ── Step 5: Process final user response ──────────────────────────────────
    user_response = str(user_response).strip().lower()

    done_keywords  = ("yes", "y", "done", "satisfied", "ok", "good",
                      "perfect", "finish", "end", "complete", "exit")

    if user_response in done_keywords:
        logger.info("[Node 7] User satisfied. Graph ending.")
        return {
            "final_latex_with_images": final_latex,
            "final_pdf_path":          output_path,
            "user_satisfied":          True,
            "restart_from":            "",
        }

    elif "diagram" in user_response or "image" in user_response or "figure" in user_response:
        logger.info("[Node 7] User wants to redo diagrams. Routing to Node 6.")
        return {
            "final_latex_with_images": final_latex,
            "final_pdf_path":          output_path,
            "user_satisfied":          False,
            "restart_from":            "node6",
            # Reset diagram state for re-run
            "current_diagram_index":   0,
            "generated_images":        [],
            "diagram_plan":            [],
            "diagram_generation_complete": False,
        }

    elif "content" in user_response or "fix" in user_response or "judge" in user_response:
        logger.info("[Node 7] User wants to fix content. Routing to Node 3.")
        return {
            "final_latex_with_images": final_latex,
            "final_pdf_path":          output_path,
            "user_satisfied":          False,
            "restart_from":            "node3",
            "content_approved":        False,
            "judge_iteration":         0,
            "user_correction":         user_response,
        }

    else:
        # "start over" or anything else → full restart
        logger.info(f"[Node 7] User wants to start over: '{user_response}'")
        return {
            "final_latex_with_images": final_latex,
            "final_pdf_path":          output_path,
            "user_satisfied":          False,
            "restart_from":            "node1",
        }


def _make_safe_filename(title: str) -> str:
    """Converts paper title to a safe filename."""
    import re
    safe = title.lower()
    safe = re.sub(r'[^a-z0-9\s_]', '', safe)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:60] or "research_paper"
