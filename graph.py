"""
graph.py
─────────
Wires ALL nodes and edges together into one compiled LangGraph StateGraph.

This file is the BLUEPRINT of the entire agent.
It only does ONE thing: register nodes + connect them with edges/routers.
No logic lives here — that belongs in nodes/ and routers.py.

COMPLETE FLOW:
                        ┌─────────────────────────────┐
                        │         START               │
                        └──────────────┬──────────────┘
                                       │
                               ┌───────▼───────┐
                               │    node1      │  Query Validator
                               │  (HIL: confirm│  + Human-in-Loop
                               │   or rewrite) │
                               └───────┬───────┘
                                       │ query_sufficient?
                              NO ──────┤───────── YES
                              │        │               │
                    (loop)◄───┘        └───────────────▼
                                               ┌───────────────┐
                                               │    node2      │  Prompt Engineer
                                               │               │  + Content Gen
                                               └───────┬───────┘
                                                       │
                                               ┌───────▼───────┐
                                               │    node3      │  LLM Judge
                    ◄──────────────────────────│  (HIL: approve│  + Web Search
                    │  content_approved=False   │   or correct) │  + HIL Loop
                    │  (loop with correction)   └───────┬───────┘
                    │                                   │ content_approved?
                    └───────────────────────────────────┤
                                                       YES
                                                        │
                                               ┌───────▼───────┐
                                               │    node4      │  Humanizer
                                               └───────┬───────┘
                                                       │
                                               ┌───────▼───────┐
                                               │    node5      │  LaTeX Formatter
                                               └───────┬───────┘
                                                       │
                                               ┌───────▼───────┐
                    ┌──────────────────────────│    node6      │  Diagram Generator
                    │  diagram_complete=False   │  (HIL: approve│  (Gemini + HIL)
                    │  (loop per diagram)       │   or regen)   │
                    │                           └───────┬───────┘
                    └───────────────────────────────────┤
                                              diagram_complete=True
                                                        │
                                               ┌───────▼───────┐
                                               │    node7      │  PDF Exporter
                    ┌──────────────────────────│  (HIL: final  │  (compile + HIL)
                    │  restart_from=node6/3/1   │   satisfied?) │
                    │                           └───────┬───────┘
                    └───────────────────────────────────┤
                                              user_satisfied=True
                                                        │
                                                      END
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import PaperState

# ── Node imports ──────────────────────────────────────────────────────────────
from nodes.node1_query_validator   import node1_query_validator
from nodes.node2_prompt_engineer   import node2_prompt_engineer
from nodes.node3_judge_researcher  import node3_judge_researcher
from nodes.node4_humanizer         import node4_humanizer
from nodes.node5_latex_formatter   import node5_latex_formatter
from nodes.node6_diagram_generator import node6_diagram_generator
from nodes.node7_pdf_exporter      import node7_pdf_exporter

# ── Router imports ────────────────────────────────────────────────────────────
from routers import (
    route_after_query_validator,
    route_after_judge,
    route_after_diagram_generator,
    route_after_pdf_exporter,
)


def build_graph() -> "CompiledGraph":
    """
    Builds and compiles the full Research Paper Agent graph.

    Returns
    -------
    CompiledGraph
        The compiled LangGraph app. Call app.invoke() or app.stream()
        to run the agent.

    Usage
    -----
        app = build_graph()
        result = app.invoke(initial_state)
    """

    # ── 1. Create the StateGraph with our PaperState schema ───────────────────
    graph = StateGraph(PaperState)

    # ── 2. Register ALL nodes ─────────────────────────────────────────────────
    # Each .add_node(name, function) call registers a node.
    # 'name' is the string used in routing decisions.
    # 'function' is the node function from nodes/.

    graph.add_node("node1", node1_query_validator)
    graph.add_node("node2", node2_prompt_engineer)
    graph.add_node("node3", node3_judge_researcher)
    graph.add_node("node4", node4_humanizer)
    graph.add_node("node5", node5_latex_formatter)
    graph.add_node("node6", node6_diagram_generator)
    graph.add_node("node7", node7_pdf_exporter)

    # ── 3. Set the entry point ────────────────────────────────────────────────
    # This is where graph.invoke() begins execution.
    graph.set_entry_point("node1")

    # ── 4. Wire edges ─────────────────────────────────────────────────────────

    # NODE 1 → conditional: did we get a valid query?
    # route_after_query_validator returns "node2" or "node1"
    graph.add_conditional_edges(
        "node1",
        route_after_query_validator,
        {
            "node1": "node1",   # Loop: user needs to re-enter
            "node2": "node2",   # Proceed: query confirmed
        }
    )

    # NODE 2 → always goes to NODE 3
    # (Prompt engineer always feeds the judge)
    graph.add_edge("node2", "node3")

    # NODE 3 → conditional: did user approve the content?
    # route_after_judge returns "node4" or "node3"
    graph.add_conditional_edges(
        "node3",
        route_after_judge,
        {
            "node3": "node3",   # Loop: run judge again with correction
            "node4": "node4",   # Proceed: content approved
        }
    )

    # NODE 4 → always goes to NODE 5
    # (Humanizer always feeds the LaTeX formatter)
    graph.add_edge("node4", "node5")

    # NODE 5 → always goes to NODE 6
    # (LaTeX formatter always feeds diagram generator)
    graph.add_edge("node5", "node6")

    # NODE 6 → conditional: are all diagrams done?
    # route_after_diagram_generator returns "node6" or "node7"
    graph.add_conditional_edges(
        "node6",
        route_after_diagram_generator,
        {
            "node6": "node6",   # Loop: next diagram
            "node7": "node7",   # Proceed: all done
        }
    )

    # NODE 7 → conditional: is user satisfied?
    # route_after_pdf_exporter returns END, "node6", "node3", or "node1"
    graph.add_conditional_edges(
        "node7",
        route_after_pdf_exporter,
        {
            END:      END,       # Done: user satisfied
            "node6":  "node6",   # Restart: redo diagrams
            "node3":  "node3",   # Restart: fix content (judge)
            "node1":  "node1",   # Restart: start completely over
        }
    )

    # ── 5. Compile and return ─────────────────────────────────────────────────
    # MemorySaver is REQUIRED for:
    #   - interrupt() / Human-in-the-Loop pausing
    #   - app.get_state(config) to read final state
    #   - app.stream() with Command(resume=...) to work
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled

