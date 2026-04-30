"""
state.py
────────
The single TypedDict that flows through every node of the Research Paper Agent.

Think of this as the "shared whiteboard":
  - Every node READS from it
  - Every node WRITES back to it
  - LangGraph merges the returned dict into this state automatically

Flow:
  raw_query → validated_query → research_prompt → raw_content
  → improved_content → humanized_content → latex_code
  → (images inserted) → final_latex_with_images → final_pdf_path

IEEE Section Structure (15 subsections):
  abstract, intro_background, intro_problem_statement, intro_contributions,
  related_existing_research, related_preliminaries, related_design_considerations,
  framework_architecture, framework_methodology, framework_mitigation,
  eval_qualitative, eval_quantitative, eval_future_work,
  conclusion, references
"""

from typing import TypedDict, List, Dict, Optional


class DiagramPlan(TypedDict):
    """One planned diagram entry."""
    index: int              # 0-based position
    diagram_type: str       # "flowchart" | "bar_chart" | "heatmap" | etc.
    title: str              # e.g., "Accuracy Comparison Across Models"
    section: str            # Which paper section this belongs to
    diagram_spec: dict      # Structured data for matplotlib/mermaid rendering


class GeneratedImage(TypedDict):
    """One generated image entry."""
    index: int
    diagram_type: str
    title: str
    section: str            # Which paper section this belongs to (e.g., "framework_architecture")
    file_path: str          # Absolute path to saved PNG
    latex_label: str        # e.g., "fig:accuracy_comparison"
    approved: bool          # User said yes


class PaperState(TypedDict):
    """
    Complete state shared across ALL nodes.
    Each node reads what it needs and returns ONLY the keys it updated.
    LangGraph merges those updates into this state automatically.
    """

    # ── INPUT ─────────────────────────────────────────────────────────────────
    raw_query: str
    # e.g. "I want a paper on catastrophic forgetting"

    # ── NODE 1: Query Validator ────────────────────────────────────────────────
    validated_query: str
    # The confirmed, clean query used for all downstream processing

    query_sufficient: bool
    # Did the LLM decide the raw_query has enough detail?

    assumed_query: str
    # If query_sufficient=False, the LLM fills this with its best assumption

    # ── NODE 2: Prompt Engineer ────────────────────────────────────────────────
    research_prompt: str
    # The industrial-grade master prompt crafted by the agent

    raw_content: Dict[str, str]
    # IEEE sections → raw text (15 subsection keys)

    paper_title: str
    # Extracted/generated title

    paper_keywords: List[str]
    # e.g. ["catastrophic forgetting", "BERT", "fine-tuning"]

    # ── NODE 3: Judge Researcher (LLM-as-Judge + HIL loop) ────────────────────
    judge_iteration: int
    # How many judge loops have run (capped at MAX_JUDGE_ITERATIONS)

    search_context: str
    # Raw text from Tavily search

    judge_feedback: str
    # What the LLM judge says needs improvement

    improved_content: Dict[str, str]
    # Sections → improved text (after judge applied feedback + search context)

    content_approved: bool
    # Did the user approve the improved_content?

    user_correction: str
    # If content_approved=False, user typed specific correction here

    # ── NODE 4: Humanizer ──────────────────────────────────────────────────────
    humanized_content: Dict[str, str]
    # Sections → deeply humanized text

    # ── NODE 5: LaTeX Formatter ────────────────────────────────────────────────
    latex_code: str
    # Full .tex file content WITH placeholder comments for figures

    # ── NODE 6: Diagram Generator ──────────────────────────────────────────────
    diagram_plan: List[DiagramPlan]
    # All planned diagrams (max MAX_DIAGRAMS, all different types)

    current_diagram_index: int
    # Which diagram we're currently generating/reviewing

    generated_images: List[GeneratedImage]
    # Accumulates as each diagram is approved

    diagram_generation_complete: bool
    # True once all diagrams in diagram_plan have been processed

    # ── NODE 7: PDF Exporter ───────────────────────────────────────────────────
    final_latex_with_images: str
    # latex_code with placeholders replaced by actual \includegraphics{} commands

    final_pdf_path: str
    # Absolute path to the compiled PDF file

    user_satisfied: bool
    # Final HIL: did user say yes to the completed paper?

    restart_from: str
    # If not satisfied: "node1" | "node6"

    # ── CONTROL / METADATA ─────────────────────────────────────────────────────
    output_dir: str
    # Base directory for all saved files (from .env OUTPUT_DIR)

    error_message: str
    # Any non-fatal error message to surface to user
