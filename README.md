# 🔬 Research Paper Agent

An end-to-end AI agent that takes a user's research query and produces a
complete, IEEE-style research paper in LaTeX + PDF with auto-generated diagrams.

Built with **LangGraph** · **GPT-4o-mini** · **Gemini Imagen 3** · **Tavily Search**

---

## 📐 Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Node 1: Query Validator                                 │
│   • LLM checks if query has enough information          │
│   • If vague → assumes enriched query → shows user      │
│   • HIL: user confirms OR rewrites                      │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Node 2: Prompt Engineer                                 │
│   • Stage A: Crafts industrial-grade master prompt      │
│   • Stage B: GPT-4o-mini generates all 8 sections       │
│   • Extracts: title, keywords, raw_content dict         │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Node 3: Judge Researcher  ◄──────────────────┐          │
│   • Tavily searches recent papers/benchmarks │          │
│   • LLM-as-Judge reviews + improves content  │  Loop    │
│   • HIL: user approves OR provides correction│          │
│   • If correction → loops back with it       ──────────►│
└───────────────────┬─────────────────────────────────────┘
                    │ (approved)
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Node 4: Humanizer                                       │
│   • Removes all AI writing patterns per section         │
│   • Applies section-specific academic tone              │
│   • References section: cleaned separately             │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Node 5: LaTeX Formatter                                 │
│   • Converts markdown → LaTeX syntax                   │
│   • Fills DEFAULT_LATEX template                        │
│   • Leaves FIGURE_PLACEHOLDER comments for Node 7      │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Node 6: Diagram Generator  ◄─────────────────┐          │
│   • LLM plans 4-7 diagrams (all diff types)  │  Loop    │
│   • Calls Gemini Imagen 3 per diagram         │  per     │
│   • HIL: user approves OR regenerates         │  diagram │
│   • Approved → stored, next diagram ─────────►│          │
└───────────────────┬─────────────────────────────────────┘
                    │ (all done)
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Node 7: PDF Exporter                                    │
│   • Inserts \includegraphics{} for each approved image  │
│   • Runs pdflatex (twice for cross-refs)               │
│   • HIL: user satisfied?                                │
│   • "redo diagrams" → Node 6                            │
│   • "fix content"   → Node 3                            │
│   • "start over"    → Node 1                            │
│   • "yes/done"      → END                               │
└───────────────────┬─────────────────────────────────────┘
                    │
                   END
```

---

## 📁 File Structure

```
research_paper_agent/
│
├── main.py                         # Entry point — run this
├── graph.py                        # All nodes + edges wired together
├── state.py                        # PaperState TypedDict (shared notepad)
├── routers.py                      # Conditional edge decision functions
│
├── nodes/
│   ├── __init__.py
│   ├── node1_query_validator.py    # Validate query + HIL
│   ├── node2_prompt_engineer.py    # Craft prompt + generate content
│   ├── node3_judge_researcher.py   # LLM judge + web search + HIL loop
│   ├── node4_humanizer.py          # Remove AI patterns
│   ├── node5_latex_formatter.py    # Convert to LaTeX
│   ├── node6_diagram_generator.py  # Generate images (Gemini) + HIL loop
│   └── node7_pdf_exporter.py       # Insert images + compile PDF + final HIL
│
├── tools/
│   ├── __init__.py
│   ├── llm_client.py               # OpenAI GPT-4o-mini wrapper
│   ├── web_search.py               # Tavily search wrapper
│   ├── gemini_image.py             # Google Gemini Imagen 3 wrapper
│   └── pdf_converter.py            # pdflatex compiler wrapper
│
├── templates/
│   ├── __init__.py
│   └── latex_template.py           # DEFAULT_LATEX + fill_latex_template()
│
├── utils/
│   ├── __init__.py
│   └── hil_handler.py              # HIL stream/interrupt/resume loop
│
├── output/                         # Generated files saved here
│   └── images/                     # Generated diagram PNGs
│
├── requirements.txt
├── .env.example
└── agent.log                       # Created at runtime
```

---

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone <your-repo>
cd research_paper_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-...          # Required — https://platform.openai.com
TAVILY_API_KEY=tvly-...        # Recommended — https://tavily.com
GEMINI_API_KEY=AIzaSy...       # Required for diagrams — https://aistudio.google.com
OUTPUT_DIR=./output
MAX_JUDGE_ITERATIONS=5
MAX_DIAGRAMS=7
```

### 3. (Optional) Install pdflatex for PDF compilation

```bash
# Ubuntu/Debian
sudo apt install texlive-full

# macOS
brew install --cask mactex

# Windows: Install MiKTeX from https://miktex.org
```

If pdflatex is not installed, the agent saves a `.tex` file you can upload to [Overleaf](https://overleaf.com).

---

## 🚀 Running

```bash
# Interactive (will prompt for query)
python main.py

# With query as argument
python main.py --query "Catastrophic forgetting in BERT fine-tuned on GLUE benchmarks"

# Custom output directory
python main.py --query "RAG for legal AI" --output ./my_papers
```

---

## 🔄 Human-in-the-Loop Interactions

The agent pauses 3 times for your input:

| Node | When it pauses | Your options |
|------|----------------|--------------|
| Node 1 | Query is vague | `yes` (accept assumption) or type corrected query |
| Node 3 | After each judge review | `approve` or type specific correction |
| Node 6 | After each diagram | `yes` / `no` / `skip` / type custom prompt |
| Node 7 | Final review | `yes` / `redo diagrams` / `fix content` / `start over` |

---

## 📄 Output Files

```
output/
├── paper_draft.tex              # LaTeX before images
├── <paper_title>_final.tex      # Final LaTeX with images
├── <paper_title>.pdf            # Compiled PDF (if pdflatex installed)
└── images/
    ├── fig_00_system_architecture.png
    ├── fig_01_performance_comparison.png
    ├── fig_02_training_curves.png
    └── ...
```

---

## 🛠️ Customization

**Change the LLM model:**
Edit `tools/llm_client.py` → `LLM_MODEL = "gpt-4o"` (or any OpenAI model)

**Change max diagrams:**
Edit `.env` → `MAX_DIAGRAMS=5`

**Change max judge iterations:**
Edit `.env` → `MAX_JUDGE_ITERATIONS=3`

**Add a new diagram type:**
Edit `tools/gemini_image.py` → add to `ALL_DIAGRAM_TYPES` and `prompts` dict

**Customize the LaTeX template:**
Edit `templates/latex_template.py` → modify `DEFAULT_LATEX`
