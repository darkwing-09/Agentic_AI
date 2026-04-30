"""
templates/latex_template.py
────────────────────────────
The IEEE Conference (IEEEtran) LaTeX template for the research paper.

IMPORTANT: We use <<<PLACEHOLDER>>> style markers — NOT Python's {} format strings —
because LaTeX itself uses { } everywhere and mixing the two causes chaos.

Node 5 (latex_formatter) calls fill_latex_template() to produce the final .tex file.
Node 7 (pdf_exporter) later replaces FIGURE_PLACEHOLDER comments with real
\\includegraphics{} commands once images are confirmed.
"""

# ── The IEEE Conference Template ─────────────────────────────────────────────
# All <<<CAPS>>> markers will be replaced by fill_latex_template()
# FIGURE_PLACEHOLDER comments will be replaced by node7_pdf_exporter

DEFAULT_LATEX = r"""
\documentclass[conference]{IEEEtran}
\usepackage{tikz}
\usetikzlibrary{shapes, arrows.meta, positioning}
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath, amssymb}
\usepackage{cite}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{caption}

%% ── Hyperlink Setup ─────────────────────────────────────────────────────────
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan,
    citecolor=blue,
    pdftitle={<<<TITLE>>>},
    pdfauthor={Research Agent},
    pdfkeywords={<<<KEYWORDS>>>}
}

% ----------------------------------------------------------------
%  TITLE
% ----------------------------------------------------------------
\title{<<<TITLE>>>}

% ----------------------------------------------------------------
%  AUTHORS
% ----------------------------------------------------------------
\author{
<<<AUTHORS>>>
}

\begin{document}

\maketitle

% ================================================================
%  ABSTRACT  (200–250 words)
% ================================================================
\begin{abstract}
<<<ABSTRACT>>>
\end{abstract}

% ----------------------------------------------------------------
%  KEYWORDS
% ----------------------------------------------------------------
\begin{IEEEkeywords}
<<<KEYWORDS>>>
\end{IEEEkeywords}


% ================================================================
\section{Introduction}
% ================================================================

% ----------------------------------------------------------------
%  I-A BACKGROUND  (120–160 words)
% ----------------------------------------------------------------
\subsection{Background}
<<<INTRO_BACKGROUND>>>

% ----------------------------------------------------------------
%  I-B PROBLEM STATEMENT  (150–200 words)
% ----------------------------------------------------------------
\subsection{Problem Statement}
<<<INTRO_PROBLEM_STATEMENT>>>

% ----------------------------------------------------------------
%  I-C CONTRIBUTIONS  (180–220 words)
% ----------------------------------------------------------------
\subsection{Contributions}
<<<INTRO_CONTRIBUTIONS>>>


% ================================================================
\section{Related Work}
% ================================================================

% ----------------------------------------------------------------
%  II-A EXISTING RESEARCH  (200–250 words)
% ----------------------------------------------------------------
\subsection{Existing Research}
<<<RELATED_EXISTING_RESEARCH>>>

% ----------------------------------------------------------------
%  II-B PRELIMINARIES  (200–250 words)
% ----------------------------------------------------------------
\subsection{Preliminaries}
<<<RELATED_PRELIMINARIES>>>

% ----------------------------------------------------------------
%  II-C DESIGN CONSIDERATIONS  (180–230 words)
% ----------------------------------------------------------------
\subsection{Design Considerations}
<<<RELATED_DESIGN_CONSIDERATIONS>>>


% ================================================================
\section{Proposed Framework}
% ================================================================

% ----------------------------------------------------------------
%  III-A SYSTEM ARCHITECTURE  (200–240 words)
% ----------------------------------------------------------------
\subsection{System Architecture}

% FIGURE_PLACEHOLDER: fig:architecture

<<<FRAMEWORK_ARCHITECTURE>>>

% ----------------------------------------------------------------
%  III-B METHODOLOGY  (240–280 words)
% ----------------------------------------------------------------
\subsection{Methodology}

% FIGURE_PLACEHOLDER: fig:methodology

<<<FRAMEWORK_METHODOLOGY>>>

% ----------------------------------------------------------------
%  III-C MITIGATION STRATEGIES  (200–240 words)
% ----------------------------------------------------------------
\subsection{Mitigation Strategies}
<<<FRAMEWORK_MITIGATION>>>


% ================================================================
\section{Evaluation}
% ================================================================

% ----------------------------------------------------------------
%  IV-A QUALITATIVE ANALYSIS  (180–230 words)
% ----------------------------------------------------------------
\subsection{Qualitative Analysis}

% FIGURE_PLACEHOLDER: fig:scope_detection

<<<EVAL_QUALITATIVE>>>

% ----------------------------------------------------------------
%  IV-B QUANTITATIVE ANALYSIS  (240–280 words)
% ----------------------------------------------------------------
\subsection{Quantitative Analysis}

% FIGURE_PLACEHOLDER: fig:confusion_matrix
% FIGURE_PLACEHOLDER: fig:topk_curve

<<<EVAL_QUANTITATIVE>>>

% ----------------------------------------------------------------
%  IV-C FUTURE WORK  (150–200 words)
% ----------------------------------------------------------------
\subsection{Future Work}
<<<EVAL_FUTURE_WORK>>>


% ================================================================
\section{Conclusion}  (180–220 words)
% ================================================================

<<<CONCLUSION>>>


% ================================================================
%  REFERENCES  (22–25 entries)
% ================================================================
\begin{thebibliography}{00}

<<<REFERENCES>>>

\end{thebibliography}

\end{document}
"""

# ── Section key mapping: state dict key → template placeholder ───────────────
SECTION_PLACEHOLDER_MAP = {
    "abstract":                     "<<<ABSTRACT>>>",
    "intro_background":             "<<<INTRO_BACKGROUND>>>",
    "intro_problem_statement":      "<<<INTRO_PROBLEM_STATEMENT>>>",
    "intro_contributions":          "<<<INTRO_CONTRIBUTIONS>>>",
    "related_existing_research":    "<<<RELATED_EXISTING_RESEARCH>>>",
    "related_preliminaries":        "<<<RELATED_PRELIMINARIES>>>",
    "related_design_considerations":"<<<RELATED_DESIGN_CONSIDERATIONS>>>",
    "framework_architecture":       "<<<FRAMEWORK_ARCHITECTURE>>>",
    "framework_methodology":        "<<<FRAMEWORK_METHODOLOGY>>>",
    "framework_mitigation":         "<<<FRAMEWORK_MITIGATION>>>",
    "eval_qualitative":             "<<<EVAL_QUALITATIVE>>>",
    "eval_quantitative":            "<<<EVAL_QUANTITATIVE>>>",
    "eval_future_work":             "<<<EVAL_FUTURE_WORK>>>",
    "conclusion":                   "<<<CONCLUSION>>>",
    "references":                   "<<<REFERENCES>>>",
}

# ── Required section keys (for validation) ───────────────────────────────────
REQUIRED_SECTIONS = list(SECTION_PLACEHOLDER_MAP.keys())


def fill_latex_template(
    sections: dict,
    title: str,
    keywords: list[str],
    authors: str = "",
) -> str:
    """
    Fills the DEFAULT_LATEX template with actual content.

    Parameters
    ----------
    sections : dict
        Keys match SECTION_PLACEHOLDER_MAP. Values are the text content.
    title : str
        Paper title string.
    keywords : list[str]
        List of keyword strings.
    authors : str, optional
        Formatted author block. Defaults to "Research Agent".

    Returns
    -------
    str
        Complete .tex file content, ready to compile.
    """
    latex = DEFAULT_LATEX

    # Replace title, keywords, and authors
    latex = latex.replace("<<<TITLE>>>", _escape_latex(title))
    latex = latex.replace("<<<KEYWORDS>>>", ", ".join(keywords))

    authors_block = authors or (
        "Research Agent\\\\\n"
        "\\textit{AI Research Laboratory}\\\\\n"
        "\\href{mailto:research@agent.ai}{research@agent.ai}"
    )
    latex = latex.replace("<<<AUTHORS>>>", authors_block)

    # Replace each section
    for section_key, placeholder in SECTION_PLACEHOLDER_MAP.items():
        content = sections.get(section_key, f"% Section '{section_key}' not provided.")
        latex = latex.replace(placeholder, content)

    return latex


def _escape_latex(text: str) -> str:
    """
    Escapes special LaTeX characters in plain text strings
    (used for title/keywords, NOT for body content which is already formatted).
    """
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\~{}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def insert_figure_into_latex(latex: str, label: str, file_path: str, caption: str) -> str:
    """
    Replaces a FIGURE_PLACEHOLDER comment with a real LaTeX figure block.

    Called by node7_pdf_exporter for each approved image.
    This is the LEGACY method — kept for backward compatibility.
    Prefer insert_all_figures() for section-aware placement.

    Parameters
    ----------
    latex : str
        Current LaTeX content with FIGURE_PLACEHOLDER comments.
    label : str
        e.g. "fig:accuracy_comparison"
    file_path : str
        Path to the image file (absolute or relative).
    caption : str
        Figure caption text.

    Returns
    -------
    str
        Updated LaTeX with the figure block inserted.
    """
    placeholder = f"% FIGURE_PLACEHOLDER: {label}"

    figure_block = (
        f"\n\\begin{{figure}}[!t]\n"
        f"    \\centering\n"
        f"    \\includegraphics[width=0.48\\textwidth]{{{file_path}}}\n"
        f"    \\caption{{{_escape_latex(caption)}}}\n"
        f"    \\label{{{label}}}\n"
        f"\\end{{figure}}\n"
    )
    return latex.replace(placeholder, figure_block)


# ── Section key → LaTeX subsection heading mapping ──────────────────────────
# Used by insert_all_figures() to locate where each image should go.
_SECTION_TO_HEADING = {
    "framework_architecture":       r"\subsection{System Architecture}",
    "framework_methodology":        r"\subsection{Methodology}",
    "framework_mitigation":         r"\subsection{Mitigation Strategies}",
    "eval_qualitative":             r"\subsection{Qualitative Analysis}",
    "eval_quantitative":            r"\subsection{Quantitative Analysis}",
    "eval_future_work":             r"\subsection{Future Work}",
    "intro_background":             r"\subsection{Background}",
    "intro_problem_statement":      r"\subsection{Problem Statement}",
    "intro_contributions":          r"\subsection{Contributions}",
    "related_existing_research":    r"\subsection{Existing Research}",
    "related_preliminaries":        r"\subsection{Preliminaries}",
    "related_design_considerations":r"\subsection{Design Considerations}",
}


def insert_all_figures(latex: str, approved_images: list, output_dir: str) -> str:
    """
    Inserts ALL approved figures into the LaTeX at their correct section positions.

    Strategy (in priority order):
      1. Try to find the exact subsection heading for the image's section
      2. Try to find any FIGURE_PLACEHOLDER comment and replace it
      3. Fall back: append before \\end{document}

    Uses RELATIVE image paths (relative to output_dir) so pdflatex can
    resolve them regardless of where it's invoked from.

    Parameters
    ----------
    latex : str
        Complete LaTeX document content.
    approved_images : list[GeneratedImage]
        List of approved images with section, file_path, title, latex_label.
    output_dir : str
        The output directory where the .tex file lives.

    Returns
    -------
    str
        LaTeX with all figures inserted.
    """
    import os
    import re

    for img in approved_images:
        file_path  = img.get("file_path", "")
        title      = img.get("title", "Figure")
        label      = img.get("latex_label", f"fig:{img.get('index', 0)}")
        section    = img.get("section", "")

        # Make path relative to output_dir so pdflatex finds it
        try:
            rel_path = os.path.relpath(file_path, output_dir)
        except ValueError:
            rel_path = file_path  # Fallback: use absolute path

        figure_block = (
            f"\n\\begin{{figure}}[!t]\n"
            f"    \\centering\n"
            f"    \\includegraphics[width=0.45\\textwidth]{{{rel_path}}}\n"
            f"    \\caption{{{_escape_latex(title)}}}\n"
            f"    \\label{{{label}}}\n"
            f"\\end{{figure}}\n"
        )

        inserted = False

        # Strategy 1: Find the subsection heading for this image's section
        if section and section in _SECTION_TO_HEADING:
            heading = _SECTION_TO_HEADING[section]
            pos = latex.find(heading)
            if pos != -1:
                # Insert AFTER the heading line
                end_of_line = latex.find("\n", pos)
                if end_of_line != -1:
                    latex = latex[:end_of_line + 1] + figure_block + latex[end_of_line + 1:]
                    inserted = True

        # Strategy 2: Try matching any remaining FIGURE_PLACEHOLDER
        if not inserted:
            placeholder_pattern = r'% FIGURE_PLACEHOLDER: \S+\n?'
            match = re.search(placeholder_pattern, latex)
            if match:
                latex = latex[:match.start()] + figure_block + latex[match.end():]
                inserted = True

        # Strategy 3: Append before \end{document}
        if not inserted:
            end_doc = latex.rfind(r"\end{document}")
            if end_doc != -1:
                latex = latex[:end_doc] + figure_block + "\n" + latex[end_doc:]

    # Clean up any remaining unused placeholders
    latex = re.sub(r'% FIGURE_PLACEHOLDER: \S+\n?', '', latex)

    return latex
