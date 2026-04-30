"""
tools/diagram_engine.py
────────────────────────
Matplotlib + Mermaid diagram generation engine for Node 6.

Replaces the former Gemini-based image generation with fully local,
deterministic rendering using:
  • matplotlib + seaborn  → bar charts, line graphs, heatmaps,
                            pie charts, scatter plots, confusion matrices
  • mermaid-cli (mmdc)    → flowcharts, block diagrams, sequence diagrams
  • matplotlib TikZ-style → architecture / pipeline block diagrams (fallback)

Every function returns the absolute path to the saved PNG file.

Exports:
  generate_diagram(diagram_type, spec, output_path, topic) -> str
  build_diagram_spec(diagram_type, section_content, topic) -> dict
  SUPPORTED_DIAGRAM_TYPES: list[str]
"""

import os
import json
import logging
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display needed

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ── Supported diagram types ──────────────────────────────────────────────────
SUPPORTED_DIAGRAM_TYPES = [
    "flowchart",
    "bar_chart",
    "line_graph",
    "heatmap",
    "scatter_plot",
    "pie_chart",
    "block_diagram",
    "confusion_matrix",
    "topk_curve",
]

# ── IEEE-quality figure defaults ─────────────────────────────────────────────
IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def generate_diagram(
    diagram_type: str,
    spec: dict,
    output_path: str,
    topic: str = "",
) -> str:
    """
    Main entry point — dispatches to the correct renderer.

    Parameters
    ----------
    diagram_type : str
        One of SUPPORTED_DIAGRAM_TYPES.
    spec : dict
        Structured data for the diagram (labels, values, etc.).
        Keys vary by type — see each renderer's docstring.
    output_path : str
        Absolute path to save the PNG (e.g. ./output/images/fig_01_arch.png).
    topic : str
        Paper topic for titling (optional).

    Returns
    -------
    str
        Absolute path to the saved PNG file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    renderers = {
        "flowchart":        _render_flowchart,
        "bar_chart":        _render_bar_chart,
        "line_graph":       _render_line_graph,
        "heatmap":          _render_heatmap,
        "scatter_plot":     _render_scatter_plot,
        "pie_chart":        _render_pie_chart,
        "block_diagram":    _render_block_diagram,
        "confusion_matrix": _render_confusion_matrix,
        "topk_curve":       _render_topk_curve,
    }

    renderer = renderers.get(diagram_type)
    if renderer is None:
        logger.warning(
            f"Unknown diagram type '{diagram_type}'. Falling back to bar_chart."
        )
        renderer = _render_bar_chart

    logger.info(f"Rendering {diagram_type} → {output_path}")

    try:
        with plt.rc_context(IEEE_RC):
            renderer(spec, output_path, topic)
        logger.info(f"✅ Diagram saved: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Diagram rendering failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to render {diagram_type}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_bar_chart(spec: dict, output_path: str, topic: str) -> None:
    """
    Grouped bar chart.

    spec keys:
        title: str
        xlabel: str
        ylabel: str
        categories: list[str]       — x-axis labels
        groups: dict[str, list[float]]  — group_name -> values (one per category)
    """
    title      = spec.get("title", topic or "Performance Comparison")
    xlabel     = spec.get("xlabel", "Method")
    ylabel     = spec.get("ylabel", "Score (%)")
    categories = spec.get("categories", ["Method A", "Method B", "Method C"])
    groups     = spec.get("groups", {"Accuracy": [85.2, 78.4, 91.0],
                                      "F1 Score": [82.1, 75.9, 89.3]})

    x = np.arange(len(categories))
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_groups, 2)))

    for i, (group_name, values) in enumerate(groups.items()):
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=group_name,
                       color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def _render_line_graph(spec: dict, output_path: str, topic: str) -> None:
    """
    Multi-line graph (training curves style).

    spec keys:
        title: str
        xlabel: str
        ylabel: str
        x_values: list[float]
        lines: dict[str, list[float]]  — line_name -> y_values
    """
    title    = spec.get("title", topic or "Training Curves")
    xlabel   = spec.get("xlabel", "Epoch")
    ylabel   = spec.get("ylabel", "Accuracy (%)")
    x_values = spec.get("x_values", list(range(1, 11)))
    lines    = spec.get("lines", {
        "Proposed": [60, 68, 74, 79, 83, 86, 88, 89.5, 90.2, 91.0],
        "Baseline": [55, 62, 67, 71, 74, 76, 77.5, 78, 78.3, 78.5],
    })

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(lines), 2)))

    for i, (name, y_values) in enumerate(lines.items()):
        ax.plot(x_values[:len(y_values)], y_values,
                marker=markers[i % len(markers)], markersize=4,
                color=colors[i], label=name, linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def _render_heatmap(spec: dict, output_path: str, topic: str) -> None:
    """
    Heatmap (correlation matrix style).

    spec keys:
        title: str
        row_labels: list[str]
        col_labels: list[str]
        values: list[list[float]]  — 2D matrix
        cmap: str (default "Blues")
    """
    title      = spec.get("title", topic or "Heatmap")
    row_labels = spec.get("row_labels", ["A", "B", "C", "D"])
    col_labels = spec.get("col_labels", ["A", "B", "C", "D"])
    values     = spec.get("values", [
        [1.0, 0.8, 0.3, 0.1],
        [0.8, 1.0, 0.5, 0.2],
        [0.3, 0.5, 1.0, 0.7],
        [0.1, 0.2, 0.7, 1.0],
    ])
    cmap = spec.get("cmap", "Blues")

    data = np.array(values)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            color = "white" if val > data.max() * 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def _render_confusion_matrix(spec: dict, output_path: str, topic: str) -> None:
    """
    Confusion matrix with annotated cells.

    spec keys:
        title: str
        labels: list[str]
        values: list[list[int]]
    """
    title  = spec.get("title", "Confusion Matrix")
    labels = spec.get("labels", ["Class A", "Class B", "Class C"])
    values = spec.get("values", [
        [45, 3, 2],
        [5, 40, 5],
        [1, 4, 45],
    ])

    data = np.array(values)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = data[i, j]
            color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def _render_scatter_plot(spec: dict, output_path: str, topic: str) -> None:
    """
    Scatter plot with optional clusters.

    spec keys:
        title: str
        xlabel: str
        ylabel: str
        clusters: dict[str, {"x": list, "y": list}]
    """
    title    = spec.get("title", topic or "Data Distribution")
    xlabel   = spec.get("xlabel", "Dimension 1")
    ylabel   = spec.get("ylabel", "Dimension 2")
    clusters = spec.get("clusters", {
        "Cluster A": {"x": list(np.random.randn(30) + 2),
                      "y": list(np.random.randn(30) + 2)},
        "Cluster B": {"x": list(np.random.randn(30) - 2),
                      "y": list(np.random.randn(30) - 2)},
    })

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(clusters), 2)))

    for i, (name, points) in enumerate(clusters.items()):
        ax.scatter(points["x"], points["y"], label=name,
                   color=colors[i], alpha=0.7, s=30, edgecolors="white",
                   linewidth=0.4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def _render_pie_chart(spec: dict, output_path: str, topic: str) -> None:
    """
    Pie chart with percentage labels.

    spec keys:
        title: str
        labels: list[str]
        values: list[float]
    """
    title  = spec.get("title", topic or "Distribution")
    labels = spec.get("labels", ["Category A", "Category B", "Category C"])
    values = spec.get("values", [45, 30, 25])

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")

    ax.set_title(title, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def _render_topk_curve(spec: dict, output_path: str, topic: str) -> None:
    """
    Top-k cumulative accuracy curve.

    spec keys:
        title: str
        k_values: list[int]
        methods: dict[str, list[float]]  — method_name -> accuracy at each k
    """
    title    = spec.get("title", "Top-k Cumulative Accuracy")
    k_values = spec.get("k_values", [1, 3, 5, 10, 15, 20])
    methods  = spec.get("methods", {
        "Hybrid": [72.0, 84.5, 90.2, 95.1, 97.0, 98.5],
        "Semantic Only": [65.0, 77.0, 83.5, 88.0, 91.2, 93.0],
        "Structural Only": [58.0, 70.2, 76.8, 82.5, 86.0, 89.0],
    })

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    markers = ["o", "s", "^", "D", "v"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 2)))

    for i, (name, acc) in enumerate(methods.items()):
        ax.plot(k_values[:len(acc)], acc,
                marker=markers[i % len(markers)], markersize=5,
                color=colors[i], label=name, linewidth=1.8)

    ax.set_xlabel("k (Top-k)")
    ax.set_ylabel("Cumulative Accuracy (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.set_ylim(50, 100)
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MERMAID RENDERER (flowcharts, block diagrams)
# ══════════════════════════════════════════════════════════════════════════════

def _render_flowchart(spec: dict, output_path: str, topic: str) -> None:
    """
    Renders a flowchart or pipeline using Mermaid CLI (mmdc).
    Falls back to matplotlib block-rendering if mmdc is not installed.

    spec keys:
        title: str
        mermaid_code: str   — raw Mermaid diagram syntax
    """
    title = spec.get("title", "System Architecture")
    mermaid_code = spec.get("mermaid_code", _default_flowchart_mermaid(topic))

    if _try_mermaid_render(mermaid_code, output_path):
        return

    # Fallback: render as matplotlib block diagram
    logger.info("mmdc not available — rendering flowchart via matplotlib fallback.")
    nodes = spec.get("nodes", ["Input", "Processing", "Retrieval", "Generation", "Output"])
    _render_pipeline_matplotlib(nodes, title, output_path)


def _render_block_diagram(spec: dict, output_path: str, topic: str) -> None:
    """
    Renders a block / architecture diagram using Mermaid or matplotlib.

    spec keys:
        title: str
        mermaid_code: str  — raw Mermaid syntax
        nodes: list[str]   — fallback: linear pipeline node names
    """
    title = spec.get("title", "Architecture Diagram")
    mermaid_code = spec.get("mermaid_code", _default_block_diagram_mermaid(topic))

    if _try_mermaid_render(mermaid_code, output_path):
        return

    nodes = spec.get("nodes", ["Frontend", "API Layer", "Retrieval Engine", "LLM", "Storage"])
    _render_pipeline_matplotlib(nodes, title, output_path)


def _try_mermaid_render(mermaid_code: str, output_path: str) -> bool:
    """
    Attempts to render Mermaid syntax using mmdc CLI.
    Returns True on success, False if mmdc is not installed.
    """
    mmdc = shutil.which("mmdc")
    if not mmdc:
        return False

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mmd", delete=False
        ) as f:
            f.write(mermaid_code)
            mmd_path = f.name

        result = subprocess.run(
            [mmdc, "-i", mmd_path, "-o", output_path,
             "-b", "white", "-w", "1200", "-H", "800"],
            capture_output=True, text=True, timeout=30,
        )

        os.unlink(mmd_path)

        if result.returncode == 0 and Path(output_path).exists():
            logger.info(f"✅ Mermaid diagram rendered: {output_path}")
            return True
        else:
            logger.warning(f"mmdc failed: {result.stderr[:200]}")
            return False

    except Exception as e:
        logger.warning(f"Mermaid rendering error: {e}")
        return False


def _render_pipeline_matplotlib(
    nodes: list, title: str, output_path: str
) -> None:
    """Matplotlib fallback for flowchart / block diagram — renders a pipeline."""
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    ax.set_xlim(-0.5, len(nodes) - 0.5)
    ax.set_ylim(-0.5, 1.0)
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=11, pad=10)

    colors = plt.cm.Set3(np.linspace(0.1, 0.9, len(nodes)))

    for i, node_label in enumerate(nodes):
        rect = mpatches.FancyBboxPatch(
            (i - 0.4, 0.05), 0.8, 0.6,
            boxstyle="round,pad=0.08",
            facecolor=colors[i], edgecolor="#333333", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(i, 0.35, node_label, ha="center", va="center",
                fontsize=8, fontweight="bold", wrap=True)

        if i < len(nodes) - 1:
            ax.annotate(
                "", xy=(i + 0.5, 0.35), xytext=(i + 0.45, 0.35),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5),
            )

    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


# ── Default Mermaid code generators ──────────────────────────────────────────

def _default_flowchart_mermaid(topic: str) -> str:
    return f"""graph LR
    A["User Query"] --> B["Query Validation"]
    B --> C["Retrieval Engine"]
    C --> D["Hybrid Fusion"]
    D --> E["LLM Generation"]
    E --> F["Response"]

    style A fill:#E8F5E9,stroke:#4CAF50
    style B fill:#E3F2FD,stroke:#2196F3
    style C fill:#FFF3E0,stroke:#FF9800
    style D fill:#F3E5F5,stroke:#9C27B0
    style E fill:#FFEBEE,stroke:#F44336
    style F fill:#E8F5E9,stroke:#4CAF50
"""


def _default_block_diagram_mermaid(topic: str) -> str:
    return f"""graph TD
    subgraph Frontend
        UI["Web Interface"]
    end

    subgraph Backend
        API["API Gateway"]
        AUTH["Auth Layer"]
    end

    subgraph Retrieval
        VS["Vector Store"]
        GI["Graph Index"]
        HF["Hybrid Fusion"]
    end

    subgraph Generation
        LLM["LLM Engine"]
        PP["Post-processing"]
    end

    UI --> API
    API --> AUTH
    AUTH --> VS
    AUTH --> GI
    VS --> HF
    GI --> HF
    HF --> LLM
    LLM --> PP
    PP --> UI

    style UI fill:#E3F2FD,stroke:#1976D2
    style API fill:#E8F5E9,stroke:#388E3C
    style VS fill:#FFF3E0,stroke:#F57C00
    style GI fill:#FFF3E0,stroke:#F57C00
    style HF fill:#F3E5F5,stroke:#7B1FA2
    style LLM fill:#FFEBEE,stroke:#D32F2F
"""


def build_diagram_spec(
    diagram_type: str,
    section_content: str,
    topic: str,
) -> dict:
    """
    Builds a default diagram spec for a given type.
    Node 6 uses this as fallback when the LLM spec is unavailable.

    Returns a dict with keys matching the renderer's spec format.
    """
    content_snippet = section_content[:300] if section_content else topic

    defaults = {
        "flowchart": {
            "title": f"System Architecture — {topic[:40]}",
            "mermaid_code": _default_flowchart_mermaid(topic),
            "nodes": ["Input", "Preprocessing", "Retrieval", "Generation", "Output"],
        },
        "bar_chart": {
            "title": f"Performance Comparison",
            "xlabel": "Method",
            "ylabel": "Score (%)",
            "categories": ["Baseline", "Proposed", "SOTA"],
            "groups": {
                "Accuracy": [78.5, 91.2, 89.0],
                "F1-Score": [75.1, 89.3, 87.6],
                "Precision": [80.2, 92.0, 88.5],
            },
        },
        "line_graph": {
            "title": "Training Convergence",
            "xlabel": "Epoch",
            "ylabel": "Accuracy (%)",
            "x_values": list(range(1, 11)),
            "lines": {
                "Proposed": [60.0, 68.5, 75.2, 80.1, 84.3, 87.0, 89.2, 90.5, 91.0, 91.2],
                "Baseline": [55.0, 61.0, 66.5, 70.8, 74.0, 76.2, 77.8, 78.3, 78.5, 78.6],
            },
        },
        "heatmap": {
            "title": "Feature Correlation Matrix",
            "row_labels": ["Feature A", "Feature B", "Feature C", "Feature D"],
            "col_labels": ["Feature A", "Feature B", "Feature C", "Feature D"],
            "values": [
                [1.0, 0.82, 0.35, 0.12],
                [0.82, 1.0, 0.48, 0.21],
                [0.35, 0.48, 1.0, 0.73],
                [0.12, 0.21, 0.73, 1.0],
            ],
        },
        "scatter_plot": {
            "title": "Embedding Space Visualization",
            "xlabel": "Dimension 1",
            "ylabel": "Dimension 2",
            "clusters": {
                "Class A": {"x": list(np.random.randn(25) + 3),
                            "y": list(np.random.randn(25) + 3)},
                "Class B": {"x": list(np.random.randn(25) - 2),
                            "y": list(np.random.randn(25) + 1)},
                "Class C": {"x": list(np.random.randn(25)),
                            "y": list(np.random.randn(25) - 3)},
            },
        },
        "pie_chart": {
            "title": "Dataset Distribution",
            "labels": ["Training", "Validation", "Test"],
            "values": [70, 15, 15],
        },
        "block_diagram": {
            "title": f"Architecture — {topic[:40]}",
            "mermaid_code": _default_block_diagram_mermaid(topic),
            "nodes": ["Frontend", "API", "Retrieval", "LLM", "Storage"],
        },
        "confusion_matrix": {
            "title": "Classification Confusion Matrix",
            "labels": ["Class A", "Class B", "Class C", "Class D"],
            "values": [
                [42, 3, 2, 1],
                [4, 38, 5, 3],
                [1, 4, 43, 2],
                [2, 2, 3, 41],
            ],
        },
        "topk_curve": {
            "title": "Top-k Retrieval Accuracy",
            "k_values": [1, 3, 5, 10, 15, 20],
            "methods": {
                "Hybrid": [72.0, 84.5, 90.2, 95.1, 97.0, 98.5],
                "Semantic Only": [65.0, 77.0, 83.5, 88.0, 91.2, 93.0],
                "Structural Only": [58.0, 70.2, 76.8, 82.5, 86.0, 89.0],
            },
        },
    }

    return defaults.get(diagram_type, defaults["bar_chart"])
