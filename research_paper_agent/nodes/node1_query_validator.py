"""
nodes/node1_query_validator.py
───────────────────────────────
NODE 1 — Query Validator with Human-in-the-Loop (HIL)

PURPOSE:
  Check if the user's raw query has enough information to produce
  a high-quality research paper. If not, make a smart assumption
  and ask the user to confirm or rewrite.

READS FROM STATE:
  - raw_query

WRITES TO STATE:
  - validated_query
  - query_sufficient
  - assumed_query

HIL PATTERN:
  If query is insufficient:
    1. LLM generates an "assumed" enriched version of the query
    2. interrupt() pauses execution → shows user the assumption
    3. User types "yes" to accept OR types their corrected query
    4. Graph resumes via Command(resume=user_response)

EXAMPLES:
  raw_query = "catastrophic forgetting"
    → NOT sufficient (too vague)
    → assumed = "Catastrophic forgetting in fine-tuned BERT, RoBERTa,
                 DistilBERT evaluated on GLUE benchmarks (SST-2, MRPC, QNLI, RTE)"
    → User: "yes" → validated_query = assumed
    → User: "No, I want it on LLaMA-2 with MMLU" → validated_query = user's text

  raw_query = "Write a research paper on LoRA fine-tuning for LLMs with
               experiments on GLUE, comparing EWC and replay strategies"
    → SUFFICIENT → validated_query = raw_query (no HIL needed)
"""

import os
import json
import logging
from dotenv import load_dotenv
from langgraph.types import interrupt

from tools.llm_client import call_llm_structured, call_llm
from state import PaperState

load_dotenv()
logger = logging.getLogger(__name__)


# ── System prompt for sufficiency check ──────────────────────────────────────
SUFFICIENCY_CHECK_SYSTEM = """
You are an academic research assistant analyzing whether a user's research query
has enough information to write a complete, high-quality research paper.

A SUFFICIENT query contains at least 3 of:
1. Specific topic / problem (not just a buzzword)
2. Method, model, or approach being studied
3. Dataset, benchmark, or domain
4. Comparison baseline or evaluation criteria
5. Scope (e.g., "for NLP", "in computer vision", "for edge devices")

An INSUFFICIENT query is too vague: a single phrase, buzzword, or topic
without context about what specifically to research.

Respond ONLY with valid JSON in this exact format:
{
  "is_sufficient": true or false,
  "reason": "one sentence explaining why",
  "assumed_query": "if is_sufficient=false: write a rich, specific, complete
                    research query you assume the user wants.
                    if is_sufficient=true: copy the original query."
}
"""


def node1_query_validator(state: PaperState) -> dict:
    """
    LangGraph node function for query validation.

    Parameters
    ----------
    state : PaperState
        Current graph state. Reads: raw_query

    Returns
    -------
    dict
        State updates: validated_query, query_sufficient, assumed_query
    """
    raw_query = state.get("raw_query", "").strip()

    logger.info(f"[Node 1] Validating query: '{raw_query[:80]}...'")

    if not raw_query:
        # Edge case: user submitted empty query
        user_input = interrupt({
            "type": "empty_query",
            "message": (
                "⚠️  Your query is empty.\n"
                "Please type your research topic or question."
            )
        })
        # User provided a query after prompt
        raw_query = str(user_input).strip()
        return {
            "raw_query": raw_query,
            "validated_query": raw_query,
            "query_sufficient": True,
            "assumed_query": raw_query,
        }

    # ── Step 1: Ask LLM to evaluate the query ────────────────────────────────
    try:
        analysis = call_llm_structured(
            system_prompt=SUFFICIENCY_CHECK_SYSTEM,
            user_prompt=f'Evaluate this research query: "{raw_query}"',
            temperature=0.2
        )
    except Exception as e:
        logger.error(f"[Node 1] LLM call failed: {e}")
        # Fail-safe: assume query is sufficient and continue
        return {
            "validated_query": raw_query,
            "query_sufficient": True,
            "assumed_query": raw_query,
            "error_message": f"Query validation skipped (LLM error): {str(e)}"
        }

    is_sufficient = analysis.get("is_sufficient", True)
    assumed_query = analysis.get("assumed_query", raw_query)
    reason = analysis.get("reason", "")

    logger.info(f"[Node 1] Sufficient: {is_sufficient} | Reason: {reason}")

    # ── Step 2: If sufficient, continue without HIL ───────────────────────────
    if is_sufficient:
        logger.info("[Node 1] Query is sufficient. Proceeding to Node 2.")
        return {
            "validated_query": raw_query,
            "query_sufficient": True,
            "assumed_query": raw_query,
        }

    # ── Step 3: Query is NOT sufficient → HIL: ask user to confirm assumption ─
    logger.info(f"[Node 1] Query insufficient. Showing assumption to user.")

    # This PAUSES the graph and waits for human response
    # main.py handles the loop that resumes the graph
    user_response = interrupt({
        "type":    "query_insufficient",
        "message": (
            f"\n{'─'*60}\n"
            f"🤔  Your query seems a bit vague: \"{raw_query}\"\n\n"
            f"📝  Reason: {reason}\n\n"
            f"💡  I assumed you mean:\n"
            f"    \"{assumed_query}\"\n\n"
            f"{'─'*60}\n"
            f"Type  'yes'  to proceed with my assumption.\n"
            f"Or    type your corrected/complete query below.\n"
            f"{'─'*60}"
        ),
        "assumed_query": assumed_query,
        "original_query": raw_query,
    })

    # ── Step 4: Process user's response after HIL ────────────────────────────
    user_response = str(user_response).strip()

    if user_response.lower() in ("yes", "y", "ok", "sure", "correct", "proceed"):
        # User accepted the assumption
        logger.info("[Node 1] User confirmed assumption. Proceeding.")
        return {
            "validated_query": assumed_query,
            "query_sufficient": True,
            "assumed_query": assumed_query,
        }
    else:
        # User provided their own corrected query
        logger.info(f"[Node 1] User provided corrected query: '{user_response[:80]}'")
        return {
            "raw_query": user_response,
            "validated_query": user_response,
            "query_sufficient": True,
            "assumed_query": user_response,
        }
