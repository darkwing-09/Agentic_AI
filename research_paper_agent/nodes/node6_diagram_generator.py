"""
nodes/node6_diagram_generator.py
──────────────────────────────────
NODE 6 — Diagram Generator (Matplotlib + Mermaid + HIL per Image)

PURPOSE:
  Plans up to MAX_DIAGRAMS diagrams for the paper, generates them
  one at a time using matplotlib (charts/plots) or mermaid (flowcharts),
  shows each to the user for approval, and loops (regenerate) if no.

  ALL diagrams are of DIFFERENT types — no two the same.
  Types: flowchart, bar_chart, line_graph, heatmap,
         scatter_plot, pie_chart, block_diagram, confusion_matrix, topk_curve

FLOW (per diagram):
  1. LLM plans what diagrams are needed + structured specs (first call only)
  2. For each diagram in the plan:
     a. Call diagram_engine to render → save as PNG
     b. interrupt() → show image path + description to user
     c. User says "yes" → approve, store in generated_images
     d. User says "no" / "regenerate" → regenerate same diagram
     e. Move to next diagram

READS FROM STATE:
  - humanized_content, paper_title, validated_query, latex_code
  - current_diagram_index, generated_images, diagram_plan, output_dir

WRITES TO STATE:
  - diagram_plan, current_diagram_index, generated_images,
    diagram_generation_complete
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from langgraph.types import interrupt

from state import PaperState, DiagramPlan, GeneratedImage
from tools.llm_client import call_llm_structured
from tools.diagram_engine import (
    generate_diagram, build_diagram_spec, SUPPORTED_DIAGRAM_TYPES
)

load_dotenv()
logger = logging.getLogger(__name__)

MAX_DIAGRAMS = int(os.getenv("MAX_DIAGRAMS", "7"))


# System prompt for diagram planning — NOW returns structured specs
DIAGRAM_PLANNER_SYSTEM = f"""
You are an expert academic data visualization specialist for IEEE papers.

Your task: Analyze a research paper and decide which diagrams will best
communicate the research to readers. For each diagram, provide STRUCTURED
data that matplotlib can render directly.

Rules:
1. Plan between 4 and 7 diagrams total
2. Every diagram must be a DIFFERENT type (no duplicates)
3. Each diagram must directly relate to the paper's content
4. Assign each diagram to the most appropriate section
5. Provide STRUCTURED specs (axis labels, data values, node names) — NOT free-text prompts

Available diagram types: {', '.join(SUPPORTED_DIAGRAM_TYPES)}

Respond ONLY with valid JSON:
{{
  "diagrams": [
    {{
      "index": 0,
      "diagram_type": "flowchart",
      "title": "System Architecture Pipeline",
      "section": "framework_architecture",
      "diagram_spec": {{
        "title": "System Architecture",
        "nodes": ["Input", "Preprocessing", "Retrieval", "Generation", "Output"],
        "mermaid_code": "graph LR\\n  A[Input] --> B[Preprocessing]\\n  B --> C[Retrieval]\\n  C --> D[Generation]\\n  D --> E[Output]"
      }}
    }},
    {{
      "index": 1,
      "diagram_type": "bar_chart",
      "title": "Performance Comparison",
      "section": "eval_quantitative",
      "diagram_spec": {{
        "title": "Performance Comparison Across Methods",
        "xlabel": "Method",
        "ylabel": "Score (%)",
        "categories": ["Baseline", "Proposed", "SOTA"],
        "groups": {{
          "Accuracy": [78.5, 91.2, 89.0],
          "F1": [75.1, 89.3, 87.6]
        }}
      }}
    }},
    ...
  ]
}}
"""


def node6_diagram_generator(state: PaperState) -> dict:
    """
    LangGraph node: Generates one diagram per call (loops via graph routing).

    The graph routes back to this node until diagram_generation_complete=True.

    Parameters
    ----------
    state : PaperState
        Reads: humanized_content, paper_title, validated_query,
               current_diagram_index, diagram_plan, generated_images, output_dir

    Returns
    -------
    dict
        State updates for the current diagram step.
    """
    output_dir      = state.get("output_dir", "./output")
    paper_title     = state.get("paper_title", "Research Paper")
    validated_query = state.get("validated_query", "")
    humanized_content = state.get("humanized_content", {})
    current_index   = state.get("current_diagram_index", 0)
    diagram_plan    = state.get("diagram_plan", [])
    generated_images = list(state.get("generated_images", []))

    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Plan diagrams (only on first entry to this node) ─────────────
    if not diagram_plan:
        logger.info("[Node 6] Planning diagrams for the paper...")

        # Build content summary for planner
        content_summary = _build_content_summary(humanized_content)

        try:
            plan_result = call_llm_structured(
                system_prompt=DIAGRAM_PLANNER_SYSTEM,
                user_prompt=(
                    f"Paper Title: {paper_title}\n"
                    f"Research Topic: {validated_query}\n\n"
                    f"Paper Content Summary:\n{content_summary}\n\n"
                    f"Plan the optimal set of diagrams (max {MAX_DIAGRAMS}, "
                    f"all different types) for this paper. "
                    f"Include detailed structured specs for each diagram."
                ),
                temperature=0.3
            )

            diagram_plan = plan_result.get("diagrams", [])

            # Enforce max diagrams and unique types
            diagram_plan = _enforce_unique_types(diagram_plan)[:MAX_DIAGRAMS]

            # Re-index after filtering
            for i, d in enumerate(diagram_plan):
                d["index"] = i
                # Ensure diagram_spec exists with at least a title
                if "diagram_spec" not in d or not d["diagram_spec"]:
                    d["diagram_spec"] = build_diagram_spec(
                        d["diagram_type"],
                        humanized_content.get(d.get("section", ""), ""),
                        validated_query
                    )

            logger.info(f"[Node 6] Planned {len(diagram_plan)} diagrams:")
            for d in diagram_plan:
                logger.info(f"  [{d['index']}] {d['diagram_type']}: {d['title']}")

        except Exception as e:
            logger.error(f"[Node 6] Diagram planning failed: {e}")
            diagram_plan = _build_default_diagram_plan(validated_query)

    # ── Step 2: Check if all diagrams are done ────────────────────────────────
    if current_index >= len(diagram_plan):
        logger.info(f"[Node 6] All {len(diagram_plan)} diagrams processed.")
        return {
            "diagram_plan":               diagram_plan,
            "current_diagram_index":      current_index,
            "generated_images":           generated_images,
            "diagram_generation_complete": True,
        }

    # ── Step 3: Generate the current diagram ──────────────────────────────────
    current_diagram = diagram_plan[current_index]
    diagram_type    = current_diagram["diagram_type"]
    diagram_title   = current_diagram["title"]
    diagram_section = current_diagram.get("section", "framework_methodology")
    diagram_spec    = current_diagram.get("diagram_spec", {})

    # Ensure we have a valid spec
    if not diagram_spec:
        diagram_spec = build_diagram_spec(
            diagram_type,
            humanized_content.get(diagram_section, ""),
            validated_query
        )

    # File path for this image
    safe_title  = _make_safe_filename(diagram_title)
    image_path  = str(images_dir / f"fig_{current_index:02d}_{safe_title}.png")
    latex_label = f"fig:{safe_title}"

    logger.info(
        f"[Node 6] Generating diagram {current_index + 1}/{len(diagram_plan)}: "
        f"{diagram_type} — '{diagram_title}'"
    )

    # ── Step 4: Call diagram engine to render the image ───────────────────────
    generation_error = None
    try:
        saved_path = generate_diagram(
            diagram_type=diagram_type,
            spec=diagram_spec,
            output_path=image_path,
            topic=validated_query,
        )
        logger.info(f"[Node 6] Image generated: {saved_path}")
    except Exception as e:
        generation_error = str(e)
        logger.error(f"[Node 6] Image generation failed: {e}")
        saved_path = None

    # ── Step 5: HIL — show image to user, ask for approval ───────────────────
    if generation_error:
        interrupt_message = (
            f"\n{'═'*65}\n"
            f"🖼️   DIAGRAM {current_index + 1}/{len(diagram_plan)}\n"
            f"{'═'*65}\n"
            f"Type:    {diagram_type}\n"
            f"Title:   {diagram_title}\n"
            f"Section: {diagram_section}\n\n"
            f"⚠️  Generation Failed: {generation_error}\n\n"
            f"Options:\n"
            f"  • 'retry'  → Try generating again\n"
            f"  • 'skip'   → Skip this diagram\n"
            f"{'═'*65}"
        )
    else:
        # Show spec summary
        spec_preview = json.dumps(diagram_spec, indent=2)[:300]
        interrupt_message = (
            f"\n{'═'*65}\n"
            f"🖼️   DIAGRAM {current_index + 1}/{len(diagram_plan)}\n"
            f"{'═'*65}\n"
            f"Type:    {diagram_type}\n"
            f"Title:   {diagram_title}\n"
            f"Section: {diagram_section}\n"
            f"Engine:  matplotlib/mermaid (local rendering)\n"
            f"Saved:   {saved_path}\n\n"
            f"📌  Diagram Spec:\n"
            f"    {spec_preview}{'...' if len(json.dumps(diagram_spec))>300 else ''}\n\n"
            f"Options:\n"
            f"  • 'yes' / 'approve'    → Accept this diagram\n"
            f"  • 'no' / 'regenerate'  → Generate a new version\n"
            f"  • 'skip'               → Skip this diagram\n"
            f"{'═'*65}"
        )

    user_response = interrupt({
        "type":          "diagram_review",
        "message":       interrupt_message,
        "diagram_index": current_index,
        "diagram_type":  diagram_type,
        "diagram_title": diagram_title,
        "image_path":    saved_path,
        "failed":        generation_error is not None,
    })

    # ── Step 6: Process user response ────────────────────────────────────────
    user_response = str(user_response).strip().lower()

    approve_keywords    = ("yes", "y", "ok", "approve", "approved",
                           "good", "looks good", "perfect", "accept")
    regenerate_keywords = ("no", "n", "regenerate", "retry", "redo",
                           "again", "bad", "wrong")
    skip_keywords       = ("skip", "s", "pass", "ignore", "next")

    if user_response in approve_keywords and saved_path:
        # ✅ Approved → add to generated_images and advance
        logger.info(f"[Node 6] Diagram {current_index} approved.")

        generated_images.append(GeneratedImage(
            index=current_index,
            diagram_type=diagram_type,
            title=diagram_title,
            section=diagram_section,
            file_path=saved_path,
            latex_label=latex_label,
            approved=True,
        ))

        return {
            "diagram_plan":               diagram_plan,
            "current_diagram_index":      current_index + 1,
            "generated_images":           generated_images,
            "diagram_generation_complete": (current_index + 1) >= len(diagram_plan),
        }

    elif user_response in skip_keywords:
        # ⏭ Skip this diagram
        logger.info(f"[Node 6] Diagram {current_index} skipped by user.")
        return {
            "diagram_plan":               diagram_plan,
            "current_diagram_index":      current_index + 1,
            "generated_images":           generated_images,
            "diagram_generation_complete": (current_index + 1) >= len(diagram_plan),
        }

    else:
        # 🔄 Regenerate — stay at same index
        logger.info(f"[Node 6] Regenerating diagram {current_index}...")

        # Stay at same index → graph will re-enter this node and regenerate
        return {
            "diagram_plan":               diagram_plan,
            "current_diagram_index":      current_index,
            "generated_images":           generated_images,
            "diagram_generation_complete": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_content_summary(sections: dict) -> str:
    """Summarizes section content for the diagram planner (keeps it short)."""
    summary_parts = []
    for key, text in sections.items():
        if text:
            snippet = str(text)[:200].replace("\n", " ")
            summary_parts.append(f"[{key}]: {snippet}...")
    return "\n".join(summary_parts)


def _enforce_unique_types(diagrams: list) -> list:
    """Removes duplicate diagram types, keeping the first occurrence."""
    seen_types = set()
    unique = []
    for d in diagrams:
        dtype = d.get("diagram_type", "")
        if dtype not in seen_types and dtype in SUPPORTED_DIAGRAM_TYPES:
            seen_types.add(dtype)
            unique.append(d)
    return unique


def _make_safe_filename(title: str) -> str:
    """
    Converts a diagram title to a safe filename string.
    e.g. "Accuracy Comparison Across Models" → "accuracy_comparison_across_models"
    """
    import re
    safe = title.lower()
    safe = re.sub(r'[^a-z0-9\s_]', '', safe)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:40]


def _build_default_diagram_plan(topic: str) -> list:
    """Fallback diagram plan if LLM planning fails."""
    return [
        {
            "index": 0,
            "diagram_type": "flowchart",
            "title": "System Architecture Overview",
            "section": "framework_architecture",
            "diagram_spec": build_diagram_spec("flowchart", "", topic),
        },
        {
            "index": 1,
            "diagram_type": "bar_chart",
            "title": "Performance Comparison Across Methods",
            "section": "eval_quantitative",
            "diagram_spec": build_diagram_spec("bar_chart", "", topic),
        },
        {
            "index": 2,
            "diagram_type": "line_graph",
            "title": "Training Loss and Accuracy Curves",
            "section": "eval_quantitative",
            "diagram_spec": build_diagram_spec("line_graph", "", topic),
        },
        {
            "index": 3,
            "diagram_type": "confusion_matrix",
            "title": "Classification Confusion Matrix",
            "section": "eval_quantitative",
            "diagram_spec": build_diagram_spec("confusion_matrix", "", topic),
        },
        {
            "index": 4,
            "diagram_type": "block_diagram",
            "title": "Model Architecture Diagram",
            "section": "framework_methodology",
            "diagram_spec": build_diagram_spec("block_diagram", "", topic),
        },
    ]
