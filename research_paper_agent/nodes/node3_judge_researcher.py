"""
nodes/node3_judge_researcher.py
────────────────────────────────
NODE 3 — LLM-as-Judge + Web Researcher + Human-in-the-Loop

PURPOSE:
  This is the quality gating node. It runs in a loop until the user approves.

  THREE THINGS HAPPEN:
  1. Web search (Tavily) → finds recent papers, benchmarks, industry standards
  2. LLM-as-Judge → reviews the content, applies search context, improves it
  3. HIL (interrupt) → shows the judge's assessment to user, asks for approval

LOOP BEHAVIOR:
  - If user approves → proceeds to Node 4
  - If user provides correction → applies correction + loops back to Node 3
  - Max iterations (MAX_JUDGE_ITERATIONS from env) → forces proceed to avoid infinite loop

READS FROM STATE:
  - improved_content (starts as raw_content on first run)
  - validated_query
  - paper_title
  - judge_iteration
  - user_correction (empty on first run)

WRITES TO STATE:
  - search_context
  - judge_feedback
  - improved_content  (updated)
  - judge_iteration   (incremented)
  - content_approved
  - user_correction
"""

import os
import logging
from dotenv import load_dotenv
from langgraph.types import interrupt

from tools.llm_client import call_llm_structured, call_llm
from tools.web_search import search_recent_papers
from state import PaperState

load_dotenv()
logger = logging.getLogger(__name__)

MAX_JUDGE_ITERATIONS = int(os.getenv("MAX_JUDGE_ITERATIONS", "5"))

# IEEE sections that the judge must return
IEEE_SECTIONS = [
    "abstract", "intro_background", "intro_problem_statement",
    "intro_contributions", "related_existing_research",
    "related_preliminaries", "related_design_considerations",
    "framework_architecture", "framework_methodology",
    "framework_mitigation", "eval_qualitative", "eval_quantitative",
    "eval_future_work", "conclusion", "references",
]


# ── LLM-as-Judge System Prompt ────────────────────────────────────────────────
JUDGE_SYSTEM = """
You are a world-class peer reviewer for top-tier IEEE conferences.

Your task: Review a research paper's content and IMPROVE it significantly.

You will receive:
1. The current paper sections (15 IEEE subsections)
2. Recent research context from web search
3. A user correction (if any) from the previous review round

Your job:
A. IDENTIFY weaknesses:
   - Missing related work (specific papers that should be cited)
   - Weak methodology (missing equations, unclear steps)
   - Unsupported claims (need citations or experimental backing)
   - Missing ablation studies or baselines
   - Inconsistent notation or terminology
   - Results that need stronger statistical analysis
   - Sections below target word count

B. IMPROVE every section that needs it:
   - Add concrete citations from the search context
   - Strengthen mathematical formulations
   - Add missing experimental details
   - Improve clarity and technical depth
   - Apply any specific correction the user requested
   - Ensure each section meets the IEEE word count target

C. PRESERVE what is already good.

Respond ONLY with valid JSON:
{
  "overall_score": 7,
  "feedback_summary": "2-3 sentence assessment of what was changed and why",
  "weaknesses_found": ["weakness 1", "weakness 2", ...],
  "improvements_made": ["improvement 1", "improvement 2", ...],
  "improved_sections": {
    "abstract": "improved text...",
    "intro_background": "improved text...",
    "intro_problem_statement": "improved text...",
    "intro_contributions": "improved text...",
    "related_existing_research": "improved text...",
    "related_preliminaries": "improved text...",
    "related_design_considerations": "improved text...",
    "framework_architecture": "improved text...",
    "framework_methodology": "improved text...",
    "framework_mitigation": "improved text...",
    "eval_qualitative": "improved text...",
    "eval_quantitative": "improved text...",
    "eval_future_work": "improved text...",
    "conclusion": "improved text...",
    "references": "improved text..."
  }
}
"""


def node3_judge_researcher(state: PaperState) -> dict:
    """
    LangGraph node: LLM-as-Judge with web search and HIL approval loop.

    Parameters
    ----------
    state : PaperState
        Reads: improved_content, validated_query, paper_title,
               judge_iteration, user_correction

    Returns
    -------
    dict
        State updates: search_context, judge_feedback, improved_content,
                       judge_iteration, content_approved, user_correction
    """
    iteration      = state.get("judge_iteration", 0)
    current_content = state.get("improved_content") or state.get("raw_content", {})
    validated_query = state.get("validated_query", "")
    paper_title     = state.get("paper_title", "")
    user_correction = state.get("user_correction", "")

    logger.info(f"[Node 3] Judge iteration {iteration + 1}/{MAX_JUDGE_ITERATIONS}")

    # ── Max iterations guard ──────────────────────────────────────────────────
    if iteration >= MAX_JUDGE_ITERATIONS:
        logger.warning(
            f"[Node 3] Max judge iterations ({MAX_JUDGE_ITERATIONS}) reached. "
            f"Auto-approving to proceed."
        )
        return {
            "content_approved": True,
            "judge_iteration": iteration,
        }

    # ── Step 1: Web Search for recent papers ─────────────────────────────────
    logger.info("[Node 3] Searching for recent papers and benchmarks...")

    search_query = f"{validated_query} recent advances 2024 2025"
    search_context = search_recent_papers(search_query, max_results=5)

    # Also search for specific aspects if this is a follow-up iteration
    if iteration > 0 and user_correction:
        extra_context = search_recent_papers(
            f"{user_correction} {validated_query}",
            max_results=3
        )
        search_context = search_context + "\n\n" + extra_context

    logger.info(f"[Node 3] Search context length: {len(search_context)} chars")

    # ── Step 2: Build the judge prompt ───────────────────────────────────────
    import json as _json

    # Build a compact summary of each section for the judge
    # Show section name + word count + first 400 chars (enough context)
    section_summaries = {}
    for k, v in current_content.items():
        text = str(v)
        word_count = len(text.split())
        preview = text[:400]
        if len(text) > 400:
            preview += "...[truncated]"
        section_summaries[k] = f"[{word_count} words] {preview}"

    # Truncate search context too
    search_context_trimmed = search_context[:2000] if len(search_context) > 2000 else search_context

    judge_user_prompt = (
        f"=== PAPER TITLE ===\n{paper_title}\n\n"
        f"=== TOPIC ===\n{validated_query}\n\n"
        f"=== CURRENT PAPER CONTENT (summaries with word counts) ===\n"
        f"{_json.dumps(section_summaries, indent=2)}\n\n"
        f"=== RECENT RESEARCH CONTEXT ===\n"
        f"{search_context_trimmed}\n\n"
    )

    # Add user correction if this is a follow-up iteration
    if user_correction:
        judge_user_prompt += (
            f"=== USER CORRECTION REQUEST ===\n"
            f"The user specifically asked you to: {user_correction}\n"
            f"Make sure to address this explicitly.\n\n"
        )

    judge_user_prompt += (
        f"Now review and improve ALL 15 sections. Be thorough and rigorous. "
        f"Each section should meet its IEEE word count target. "
        f"Return the complete improved paper in the JSON format specified."
    )

    # ── Step 3: LLM-as-Judge improves the content ─────────────────────────────
    logger.info("[Node 3] Running LLM judge...")

    try:
        judge_result = call_llm_structured(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=judge_user_prompt,
            temperature=0.4
        )

        improved_sections = judge_result.get("improved_sections", current_content)
        feedback_summary  = judge_result.get("feedback_summary", "Review complete.")
        weaknesses        = judge_result.get("weaknesses_found", [])
        improvements      = judge_result.get("improvements_made", [])
        overall_score     = judge_result.get("overall_score", 7)

        # Ensure all original sections are preserved if judge missed any
        for key in current_content:
            if key not in improved_sections or not improved_sections[key]:
                improved_sections[key] = current_content[key]

        logger.info(f"[Node 3] Judge score: {overall_score}/10")
        logger.info(f"[Node 3] Weaknesses found: {len(weaknesses)}")
        logger.info(f"[Node 3] Improvements made: {len(improvements)}")

    except Exception as e:
        logger.error(f"[Node 3] Judge LLM call failed: {e}")
        improved_sections = current_content
        feedback_summary  = f"Review failed: {str(e)}"
        weaknesses        = []
        improvements      = []
        overall_score     = 0

    # ── Step 4: Format feedback summary for user ──────────────────────────────
    display_feedback = _format_feedback_for_display(
        feedback_summary, weaknesses, improvements, overall_score, iteration
    )

    # ── Step 5: HIL — Show results to user, ask for approval ─────────────────
    logger.info("[Node 3] Showing judge results to user for approval...")

    # Show first 200 chars of each improved section as preview
    content_preview = _build_content_preview(improved_sections)

    user_response = interrupt({
        "type":    "judge_review",
        "message": (
            f"\n{'═'*65}\n"
            f"⚖️   JUDGE REVIEW  —  Iteration {iteration + 1}/{MAX_JUDGE_ITERATIONS}\n"
            f"{'═'*65}\n\n"
            f"{display_feedback}\n\n"
            f"{'─'*65}\n"
            f"📄  CONTENT PREVIEW (first 200 chars per section):\n"
            f"{'─'*65}\n"
            f"{content_preview}\n"
            f"{'─'*65}\n\n"
            f"Options:\n"
            f"  • Type  'approve'  → proceed to humanization\n"
            f"  • Type your correction → e.g., 'Add EWC comparison in related work'\n"
            f"  • Type  'more'     → run another judge iteration\n"
            f"{'═'*65}"
        ),
        "feedback_summary": feedback_summary,
        "weaknesses": weaknesses,
        "improvements": improvements,
        "overall_score": overall_score,
        "content_preview": content_preview,
        "iteration": iteration + 1,
    })

    # ── Step 6: Process user response ────────────────────────────────────────
    user_response = str(user_response).strip()
    response_lower = user_response.lower()

    approve_keywords = ("approve", "approved", "yes", "ok", "looks good",
                        "good", "proceed", "next", "continue", "accept")
    more_keywords    = ("more", "again", "retry", "rerun", "re-run")

    if response_lower in approve_keywords or not user_response:
        # ✅ User approved (or pressed Enter → default "approve")
        logger.info("[Node 3] User approved content. Proceeding to Node 4.")
        return {
            "search_context":   search_context,
            "judge_feedback":   feedback_summary,
            "improved_content": improved_sections,
            "judge_iteration":  iteration + 1,
            "content_approved": True,
            "user_correction":  "",
        }
    elif response_lower in more_keywords:
        # 🔄 User wants another judge iteration (no specific correction)
        logger.info("[Node 3] User requested another judge iteration.")
        return {
            "search_context":   search_context,
            "judge_feedback":   feedback_summary,
            "improved_content": improved_sections,
            "judge_iteration":  iteration + 1,
            "content_approved": False,
            "user_correction":  "",  # No correction — just re-run the judge
        }
    else:
        # 📝 User provided a specific correction
        logger.info(f"[Node 3] User requested changes: '{user_response[:80]}'")
        return {
            "search_context":   search_context,
            "judge_feedback":   feedback_summary,
            "improved_content": improved_sections,
            "judge_iteration":  iteration + 1,
            "content_approved": False,
            "user_correction":  user_response,
        }


def _format_feedback_for_display(
    summary: str,
    weaknesses: list,
    improvements: list,
    score: int,
    iteration: int
) -> str:
    """Formats judge feedback as readable text for the terminal."""
    lines = []

    # Score bar
    filled = "█" * score
    empty  = "░" * (10 - score)
    lines.append(f"📊 Quality Score: [{filled}{empty}] {score}/10\n")

    lines.append(f"📋 Summary: {summary}\n")

    if weaknesses:
        lines.append("🔍 Issues Found:")
        for w in weaknesses[:4]:
            lines.append(f"   • {w}")
        lines.append("")

    if improvements:
        lines.append("✅ Improvements Applied:")
        for imp in improvements[:4]:
            lines.append(f"   • {imp}")

    return "\n".join(lines)


def _build_content_preview(sections: dict) -> str:
    """Shows first 200 chars of each section as a preview."""
    lines = []
    for section_key, content in sections.items():
        preview = str(content)[:200]
        if len(str(content)) > 200:
            preview += "..."
        section_name = section_key.replace("_", " ").title()
        word_count = len(str(content).split())
        lines.append(f"\n[{section_name}] ({word_count} words)\n{preview}")
    return "\n".join(lines)
