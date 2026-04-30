"""
main.py
────────
The single entry point for the Research Paper Agent.

HOW TO RUN:
  python main.py
  python main.py --query "Your research topic here"
  python main.py --query "Your topic" --output ./my_output_folder

WHAT HAPPENS:
  1. Loads .env for API keys
  2. Validates env vars (warns if any are missing)
  3. Builds the LangGraph from graph.py
  4. Takes user query (from CLI arg or interactive prompt)
  5. Constructs the initial PaperState
  6. Hands off to run_agent_with_hil() which manages the full loop:
       stream → pause (HIL) → get input → resume → repeat → END
  7. Prints a summary of the final output paths
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Load .env first (before any other imports that need env vars) ─────────────
load_dotenv()

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("agent.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
# Quiet noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger  = logging.getLogger(__name__)
console = Console()


def validate_env() -> list[str]:
    """
    Checks that required environment variables are set.

    Returns
    -------
    list[str]
        List of warning messages for missing (non-critical) vars.
        Critical vars (OPENAI_API_KEY) raise immediately.
    """
    warnings = []

    # Critical — cannot run without this
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold red]❌ OPENAI_API_KEY is not set![/bold red]\n"
            "Copy .env.example to .env and add your key.\n"
            "  cp .env.example .env"
        )
        sys.exit(1)

    # Non-critical — agent degrades gracefully without these
    if not os.getenv("TAVILY_API_KEY"):
        warnings.append(
            "⚠️  TAVILY_API_KEY not set — Node 3 will judge without web search context."
        )

    return warnings


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Research Paper Agent — generates IEEE-style papers with LaTeX + diagrams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --query "Catastrophic forgetting in fine-tuned BERT on GLUE"
  python main.py --query "LoRA fine-tuning efficiency" --output ./papers
        """
    )

    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Research query (if not provided, will prompt interactively)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=os.getenv("OUTPUT_DIR", "./output"),
        help="Output directory for .tex and .pdf files (default: ./output)"
    )

    return parser.parse_args()


def print_welcome_banner() -> None:
    """Prints the welcome banner."""
    console.print(
        Panel(
            "[bold cyan]🔬 Research Paper Agent[/bold cyan]\n\n"
            "[white]Powered by:[/white] GPT-4o-mini · Matplotlib · Mermaid · Tavily Search · LangGraph\n\n"
            "[dim]Template:[/dim] IEEE Conference (IEEEtran)\n"
            "[dim]Flow:[/dim] Query → Validate → Prompt Engineer → Judge (loop) →\n"
            "[dim]       Humanize → LaTeX → Diagrams (loop) → PDF[/dim]",
            title="[bold]AI Research Paper Generator[/bold]",
            border_style="cyan",
            padding=(1, 4),
        )
    )


def get_query_interactively() -> str:
    """Prompts the user to enter their research query."""
    console.print(
        "\n[bold yellow]💡 What research paper do you want to write?[/bold yellow]\n"
        "[dim]Be as specific as possible for best results.[/dim]\n"
        "[dim]Examples:[/dim]\n"
        "[dim]  • Catastrophic forgetting in BERT fine-tuned on GLUE benchmarks[/dim]\n"
        "[dim]  • RAG-based legal AI system for Indian judiciary using FAISS[/dim]\n"
        "[dim]  • LoRA vs full fine-tuning efficiency on LLaMA-2[/dim]\n"
    )

    console.print("[bold cyan]❯[/bold cyan] ", end="")
    query = input("Your research query: ").strip()

    if not query:
        console.print("[red]Query cannot be empty. Please try again.[/red]")
        return get_query_interactively()

    return query


def build_initial_state(query: str, output_dir: str) -> dict:
    """
    Constructs the initial PaperState dict to invoke the graph with.

    All fields that nodes write to are pre-initialized to empty/default
    values to avoid KeyError in nodes that do state.get().

    Parameters
    ----------
    query : str
        The user's raw research query.
    output_dir : str
        Directory for output files.

    Returns
    -------
    dict
        Complete initial state.
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return {
        # ── Input ────────────────────────────────────────────────────────────
        "raw_query":        query,

        # ── Node 1 ───────────────────────────────────────────────────────────
        "validated_query":  "",
        "query_sufficient": False,
        "assumed_query":    "",

        # ── Node 2 ───────────────────────────────────────────────────────────
        "research_prompt":  "",
        "raw_content":      {},
        "paper_title":      "",
        "paper_keywords":   [],

        # ── Node 3 ───────────────────────────────────────────────────────────
        "judge_iteration":  0,
        "search_context":   "",
        "judge_feedback":   "",
        "improved_content": {},
        "content_approved": False,
        "user_correction":  "",

        # ── Node 4 ───────────────────────────────────────────────────────────
        "humanized_content": {},

        # ── Node 5 ───────────────────────────────────────────────────────────
        "latex_code":       "",

        # ── Node 6 ───────────────────────────────────────────────────────────
        "diagram_plan":              [],
        "current_diagram_index":     0,
        "generated_images":          [],
        "diagram_generation_complete": False,

        # ── Node 7 ───────────────────────────────────────────────────────────
        "final_latex_with_images":  "",
        "final_pdf_path":           "",
        "user_satisfied":           False,
        "restart_from":             "",

        # ── Meta ─────────────────────────────────────────────────────────────
        "output_dir":       str(Path(output_dir).resolve()),
        "error_message":    "",
    }


def print_final_summary(final_state: dict) -> None:
    """Prints a summary table of all output files after the agent ends."""
    console.print("\n")

    table = Table(title="📁 Output Summary", border_style="green")
    table.add_column("File",       style="cyan",  no_wrap=True)
    table.add_column("Location",   style="white")
    table.add_column("Status",     style="green")

    output_dir  = final_state.get("output_dir", "./output")
    paper_title = final_state.get("paper_title", "research_paper")

    # Draft LaTeX (pre-images)
    draft_tex = Path(output_dir) / "paper_draft.tex"
    if draft_tex.exists():
        table.add_row("Draft LaTeX",   str(draft_tex),   "✅ Saved")

    # Final LaTeX (with images)
    final_pdf  = final_state.get("final_pdf_path", "")
    if final_pdf:
        if final_pdf.endswith(".pdf"):
            table.add_row("Final PDF",  final_pdf, "✅ Compiled")
        else:
            table.add_row("Final LaTeX", final_pdf, "✅ Saved (compile manually)")

    # Images directory
    images_dir = Path(output_dir) / "images"
    if images_dir.exists():
        image_count = len(list(images_dir.glob("*.png")))
        if image_count > 0:
            table.add_row(
                f"Diagrams ({image_count})",
                str(images_dir),
                "✅ Generated"
            )

    # Agent log
    if Path("agent.log").exists():
        table.add_row("Agent Log", "agent.log", "✅ Written")

    console.print(table)

    if not final_pdf:
        # Check WHY there's no PDF
        error_msg = final_state.get("error_message", "")
        has_latex = (Path(output_dir) / "paper_draft.tex").exists()

        if has_latex and shutil.which("pdflatex"):
            console.print(
                "\n[yellow]⚠️  LaTeX was generated but PDF compilation was not reached. "
                "The agent may have stopped early.[/yellow]"
            )
        elif has_latex:
            console.print(
                "\n[yellow]Tip: Upload the .tex file to https://overleaf.com "
                "if pdflatex is not installed locally.[/yellow]"
            )
        else:
            console.print(
                "\n[yellow]⚠️  The agent did not complete. "
                "Re-run to generate the paper.[/yellow]"
            )


def main() -> None:
    """
    Main entry point — orchestrates the entire agent run.

    Flow:
      1. Print banner
      2. Validate env
      3. Parse args / get query
      4. Build initial state
      5. Build graph
      6. Run with HIL handler
      7. Print summary
    """
    print_welcome_banner()

    # ── Validate environment ──────────────────────────────────────────────────
    warnings = validate_env()
    for w in warnings:
        console.print(f"[yellow]{w}[/yellow]")

    # ── Parse arguments ───────────────────────────────────────────────────────
    args = parse_args()

    # ── Get research query ────────────────────────────────────────────────────
    query = args.query.strip() if args.query else get_query_interactively()

    console.print(
        f"\n[bold green]✅ Query received:[/bold green] "
        f"[italic]{query}[/italic]\n"
    )

    # ── Build initial state ───────────────────────────────────────────────────
    initial_state = build_initial_state(
        query=query,
        output_dir=args.output
    )

    # ── Build the LangGraph ───────────────────────────────────────────────────
    console.print("[dim]Building agent graph...[/dim]")
    try:
        from graph import build_graph
        app = build_graph()
        console.print("[dim]Graph compiled successfully.[/dim]\n")
    except Exception as e:
        console.print(f"[bold red]❌ Failed to build graph: {e}[/bold red]")
        logger.error(f"Graph build failed: {e}", exc_info=True)
        sys.exit(1)

    # ── Run the agent ─────────────────────────────────────────────────────────
    from utils.hil_handler import run_agent_with_hil

    logger.info(f"Starting agent with query: '{query}'")

    final_state = run_agent_with_hil(app, initial_state)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_final_summary(final_state)

    logger.info("Agent completed.")


if __name__ == "__main__":
    main()
