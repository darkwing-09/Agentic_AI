"""
utils/hil_handler.py
─────────────────────
Centralized Human-in-the-Loop (HIL) handler for the Research Paper Agent.

HOW LANGGRAPH HIL WORKS:
  When a node calls interrupt(payload), LangGraph:
  1. PAUSES the graph — saves full state to a checkpoint
  2. Emits a special "__interrupt__" event in the stream
  3. Waits for the caller to call app.stream(Command(resume=value), config)
  4. Resumes the node exactly where it paused, returning `value` from interrupt()

This module manages the outer loop:
  stream → detect interrupt → display to user → collect input → resume → repeat

INTERRUPT TYPES IN THIS AGENT (from the nodes that call interrupt()):
  ┌─────────────────────┬──────────┬─────────────────────────────────────────┐
  │ Type                │ Node     │ Expected user responses                │
  ├─────────────────────┼──────────┼─────────────────────────────────────────┤
  │ empty_query         │ Node 1   │ <research query text>                  │
  │ query_insufficient  │ Node 1   │ "yes" | corrected query text           │
  │ judge_review        │ Node 3   │ "approve" | correction text | "more"   │
  │ diagram_review      │ Node 6   │ "yes" | "no"/"regenerate" | "skip"     │
  │ final_review        │ Node 7   │ "yes"/"done" | "redo diagrams" | etc.  │
  └─────────────────────┴──────────┴─────────────────────────────────────────┘

CRITICAL LANGGRAPH CONSTRAINT:
  Command(resume=value) MUST have a truthy `value`.
  If value is "" (empty string) or None, LangGraph's map_command() yields
  zero writes → EmptyInputError. We enforce non-empty input below.

Exports:
  run_agent_with_hil(app, initial_state) -> dict (final state)
"""

import sys
import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langgraph.types import Command

logger = logging.getLogger(__name__)
console = Console()


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def run_agent_with_hil(app, initial_state: dict) -> dict:
    """
    Runs the compiled LangGraph app with full Human-in-the-Loop support.

    The loop:
      1. Stream the graph (first call with initial_state, subsequent with Command)
      2. Process events: show node progress, detect interrupts
      3. On interrupt: display payload → prompt user → create Command(resume=...)
      4. Re-stream with the Command to resume
      5. Repeat until the graph reaches END or an unrecoverable error

    Parameters
    ----------
    app : CompiledGraph
        The compiled LangGraph app from build_graph().
    initial_state : dict
        The starting state (must include raw_query at minimum).

    Returns
    -------
    dict
        The final state after the graph reaches END.
    """
    # Thread config — required for LangGraph checkpointing + HIL
    config = {"configurable": {"thread_id": "research_paper_agent_main"}}

    final_state = {}
    current_input = initial_state  # First call: dict. After interrupt: Command.

    console.print(
        Panel(
            "[bold cyan]🔬 Research Paper Agent Starting...[/bold cyan]",
            border_style="cyan"
        )
    )

    while True:
        try:
            # ── Stream the graph ─────────────────────────────────────────
            resume_command = None

            for event in app.stream(current_input, config=config, stream_mode="updates"):

                # Each event is a dict: { "node_name": {state_updates} }
                for node_name, updates in event.items():

                    if node_name == "__interrupt__":
                        # ── Graph paused — HIL needed ─────────────────────
                        interrupt_data = updates

                        # Handle both single interrupt and list/tuple of Interrupt objects
                        if isinstance(interrupt_data, (list, tuple)):
                            interrupt_data = interrupt_data[0]

                        # Extract the payload (Interrupt objects have a .value attr)
                        payload = interrupt_data
                        if hasattr(interrupt_data, 'value'):
                            payload = interrupt_data.value

                        # Display the interrupt message to the user
                        _display_interrupt(payload)

                        # Get user input (guaranteed non-empty for LangGraph)
                        user_input = _get_user_input(payload)

                        # Prepare the Command for resuming
                        resume_command = Command(resume=user_input)

                        # Break inner loop — we need to re-stream
                        break

                    else:
                        # ── Normal node completion — show progress ────────
                        _display_node_progress(node_name, updates)

                # If we got an interrupt, break the event loop to re-stream
                if resume_command is not None:
                    break

            else:
                # ── for/else: stream exhausted without interrupt → graph END ──
                console.print(
                    Panel(
                        "[bold green]✅ Research Paper Agent Complete![/bold green]",
                        border_style="green"
                    )
                )

                try:
                    final_state = app.get_state(config).values
                except Exception:
                    pass
                break

            # ── Resume with the user's response ──────────────────────────
            if resume_command is not None:
                current_input = resume_command
                # Continue the while loop → re-stream
                continue

        except KeyboardInterrupt:
            console.print(
                "\n[yellow]⚠️  Interrupted by user. Saving current state...[/yellow]"
            )
            try:
                final_state = app.get_state(config).values
            except Exception:
                pass
            break

        except Exception as e:
            logger.error(f"Graph execution error: {e}", exc_info=True)
            console.print(f"[bold red]❌ Error: {str(e)}[/bold red]")
            console.print(
                "[yellow]The graph encountered an error. "
                "Check logs for details.[/yellow]"
            )
            try:
                final_state = app.get_state(config).values
            except Exception:
                pass
            break

    return final_state


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _display_interrupt(payload: Any) -> None:
    """
    Displays the interrupt message to the user in a formatted panel.
    Handles both dict payloads (with 'message' key) and raw strings.
    """
    if isinstance(payload, dict):
        message = payload.get("message", str(payload))
        interrupt_type = payload.get("type", "input_required")
    else:
        message = str(payload)
        interrupt_type = "input_required"

    # Color the panel based on interrupt type
    color_map = {
        "query_insufficient": "yellow",
        "judge_review":       "blue",
        "diagram_review":     "magenta",
        "final_review":       "green",
        "empty_query":        "red",
        "input_required":     "cyan",
    }
    color = color_map.get(interrupt_type, "cyan")

    console.print(
        Panel(
            Text(message),
            border_style=color,
            padding=(1, 2)
        )
    )


# ═════════════════════════════════════════════════════════════════════════════
# INPUT COLLECTION
# ═════════════════════════════════════════════════════════════════════════════

# Maps each interrupt type → prompt text shown to user
_PROMPT_MAP = {
    "empty_query":        "Enter your research query: ",
    "query_insufficient": "Your response (yes / corrected query): ",
    "judge_review":       "Your decision (approve / correction / more): ",
    "diagram_review":     "Your decision (yes / no / skip): ",
    "final_review":       "Your decision (yes/done / redo diagrams / start over): ",
    "input_required":     "Your response: ",
}


def _get_user_input(payload: Any) -> str:
    """
    Prompts the user for input after an interrupt.

    CRITICAL: LangGraph requires Command(resume=value) where `value` is truthy.
    An empty string causes EmptyInputError. This function guarantees a non-empty
    return value by re-prompting until the user types something.

    No defaults — the user must always explicitly type their decision.

    Returns
    -------
    str
        Always non-empty. Safe to pass to Command(resume=...).
    """
    if isinstance(payload, dict):
        interrupt_type = payload.get("type", "input_required")
    else:
        interrupt_type = "input_required"

    prompt_text = _PROMPT_MAP.get(interrupt_type, "Your response: ")

    # ── Flush stdin buffer before asking for input ───────────────────────────
    # This prevents leftover multi-line pastes from auto-filling the prompt!
    try:
        import sys
        import select
        import termios
        
        # Flush OS buffer
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        
        # Drain Python's internal buffer by reading everything available
        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline()
    except Exception:
        pass  # Ignore on Windows or if select fails

    while True:
        console.print(f"\n[bold cyan]❯[/bold cyan] ", end="")

        try:
            user_input = input(prompt_text).strip()
        except EOFError:
            # Non-interactive environment (e.g. tests, piped input)
            console.print("[dim](auto-response: yes)[/dim]")
            return "yes"

        # If user typed something, use it
        if user_input:
            return user_input

        # User pressed Enter with no text — re-prompt
        console.print("[yellow]  ⚠ Input cannot be empty. Please type your decision.[/yellow]")


# ═════════════════════════════════════════════════════════════════════════════
# PROGRESS DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def _display_node_progress(node_name: str, updates: dict) -> None:
    """
    Shows a compact progress indicator when a node completes.
    Only shows meaningful updates (not empty dicts).
    """
    if not updates:
        return

    node_labels = {
        "node1": "🔍 Query Validated",
        "node2": "✍️  Content Generated",
        "node3": "⚖️  Judge Reviewed",
        "node4": "🌿 Content Humanized",
        "node5": "📐 LaTeX Formatted",
        "node6": "🖼️  Diagram Processed",
        "node7": "📄 PDF Exported",
    }

    label = node_labels.get(node_name, f"✅ {node_name} complete")

    # Show key state updates (not full content to keep terminal clean)
    info_parts = []
    if "paper_title" in updates and updates["paper_title"]:
        info_parts.append(f"Title: {updates['paper_title'][:60]}")
    if "judge_iteration" in updates:
        info_parts.append(f"Judge iteration: {updates['judge_iteration']}")
    if "current_diagram_index" in updates:
        info_parts.append(f"Diagram: {updates['current_diagram_index']}")
    if "final_pdf_path" in updates and updates["final_pdf_path"]:
        info_parts.append(f"Output: {updates['final_pdf_path']}")

    info_str = f" — {', '.join(info_parts)}" if info_parts else ""
    console.print(f"  [green]▶[/green] {label}{info_str}")
