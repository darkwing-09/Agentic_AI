# 🔬 Research Paper Agent


## ⚙️ Setup

### 1. Clone & Install

```bash
git clone <https://github.com/darkwing-09/Agentic_AI>
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

**Customize the LaTeX template:**
Edit `templates/latex_template.py` → modify `DEFAULT_LATEX`
