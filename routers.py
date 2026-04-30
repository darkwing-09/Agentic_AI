"""
routers.py
───────────
All conditional edge router functions for the Research Paper Agent graph.

A router is a plain Python function that:
  - Receives the current state
  - Returns a STRING — the name of the next node to go to
  - Is registered with graph.add_conditional_edges()

LangGraph uses the returned string to look up which node to route to
from the mapping dict you provide in add_conditional_edges().

ROUTERS IN THIS FILE:
  1. route_after_query_validator   → node2 OR loop back to node1
  2. route_after_judge             → node4 (approved) OR node3 (loop)
  3. route_after_diagram_generator → node7 (done) OR node6 (loop)
  4. route_after_pdf_exporter      → END | node6 | node3 | node1
"""

from langgraph.graph import END
from state import PaperState


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER 1 — After Node 1 (Query Validator)
# ─────────────────────────────────────────────────────────────────────────────

def route_after_query_validator(state: PaperState) -> str:
    """
    Decision: Did we get a valid, confirmed query?

    ┌─────────────────────────────────────────────┐
    │  query_sufficient == True  → "node2"         │
    │  query_sufficient == False → "node1" (retry) │
    └─────────────────────────────────────────────┘

    When does query_sufficient == False happen here?
      Only if the interrupt() in Node 1 somehow returns
      before the user has confirmed (shouldn't happen in normal flow,
      but this is the safety net).
    """
    if state.get("query_sufficient", False):
        return "node2"
    else:
        return "node1"   # Loop back — user needs to re-enter query


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER 2 — After Node 3 (Judge Researcher)
# ─────────────────────────────────────────────────────────────────────────────

def route_after_judge(state: PaperState) -> str:
    """
    Decision: Did the user approve the improved content?

    ┌─────────────────────────────────────────────────────┐
    │  content_approved == True  → "node4" (humanize)     │
    │  content_approved == False → "node3" (judge again)  │
    └─────────────────────────────────────────────────────┘

    The judge loop runs until:
      - User types "approve"  → content_approved = True
      - MAX_JUDGE_ITERATIONS reached → auto set content_approved = True
    """
    if state.get("content_approved", False):
        return "node4"
    else:
        return "node3"   # Loop: run judge again with user's correction


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER 3 — After Node 6 (Diagram Generator)
# ─────────────────────────────────────────────────────────────────────────────

def route_after_diagram_generator(state: PaperState) -> str:
    """
    Decision: Are all diagrams processed?

    ┌────────────────────────────────────────────────────────────────┐
    │  diagram_generation_complete == True  → "node7" (export PDF)  │
    │  diagram_generation_complete == False → "node6" (next diagram) │
    └────────────────────────────────────────────────────────────────┘

    Node 6 sets diagram_generation_complete = True when:
      current_diagram_index >= len(diagram_plan)
    meaning all diagrams in the plan have been approved or skipped.
    """
    if state.get("diagram_generation_complete", False):
        return "node7"
    else:
        return "node6"   # Loop: process next diagram


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER 4 — After Node 7 (PDF Exporter) — FINAL router
# ─────────────────────────────────────────────────────────────────────────────

def route_after_pdf_exporter(state: PaperState) -> str:
    """
    Decision: Is the user satisfied with the final paper?

    ┌──────────────────────────────────────────────────────────────────┐
    │  user_satisfied == True           → END                          │
    │  restart_from == "node6"          → "node6" (redo diagrams)      │
    │  restart_from == "node3"          → "node3" (fix content)        │
    │  restart_from == "node1"          → "node1" (start over)         │
    │  (fallback)                       → END                          │
    └──────────────────────────────────────────────────────────────────┘

    User responses that trigger each route (handled in Node 7):
      "yes" / "done"       → user_satisfied=True  → END
      "redo diagrams"      → restart_from="node6" → node6
      "fix content"        → restart_from="node3" → node3
      "start over"         → restart_from="node1" → node1
    """
    if state.get("user_satisfied", False):
        return END

    restart = state.get("restart_from", "")

    if restart == "node6":
        return "node6"
    elif restart == "node3":
        return "node3"
    elif restart == "node1":
        return "node1"
    else:
        # Fallback safety — end rather than loop forever
        return END
