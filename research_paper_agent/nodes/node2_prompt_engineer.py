"""
nodes/node2_prompt_engineer.py
───────────────────────────────
NODE 2 — Prompt Engineer

PURPOSE:
  Two-stage node:
  STAGE A → Craft an industrial-grade research prompt (the "meta-prompt")
             that will produce a well-structured, deeply technical paper
  STAGE B → Feed that prompt to GPT-4o-mini to generate raw section content

  Content is generated to fill the IEEE conference paper structure:
    15 subsections matching the IEEEtran template.

READS FROM STATE:
  - validated_query

WRITES TO STATE:
  - paper_title
  - paper_keywords
  - research_prompt      (the engineered meta-prompt)
  - raw_content          (dict: section → text, 15 IEEE subsection keys)
"""

import json
import logging
from state import PaperState
from tools.llm_client import call_llm_structured, call_llm

logger = logging.getLogger(__name__)

# ── IEEE section keys (must match SECTION_PLACEHOLDER_MAP) ───────────────────
IEEE_SECTIONS = [
    "abstract",
    "intro_background",
    "intro_problem_statement",
    "intro_contributions",
    "related_existing_research",
    "related_preliminaries",
    "related_design_considerations",
    "framework_architecture",
    "framework_methodology",
    "framework_mitigation",
    "eval_qualitative",
    "eval_quantitative",
    "eval_future_work",
    "conclusion",
    "references",
]


# ── Stage A: Prompt Engineering System Prompt ─────────────────────────────────
PROMPT_ENGINEER_SYSTEM = """
You are a world-class academic prompt engineer with 15+ years of experience
publishing in top-tier conferences (NeurIPS, ICML, ACL, CVPR, IEEE).

Your task: Given a research topic, produce a MASTER PROMPT that will generate
a complete, high-quality IEEE conference paper.

The master prompt you write must instruct the content-generating LLM to:
1. Write each section with SPECIFIC depth requirements (word count guidance)
2. Include concrete experiments, datasets, metrics, and baselines
3. Place diagram/figure suggestions inline: [FIGURE: description]
4. Cite realistic references in IEEE format: [AuthorLastName et al., YEAR]
5. Use technical but readable academic English (not AI-sounding)
6. Include mathematical formulations where appropriate
7. Follow the IEEE conference paper structure with these EXACT 15 sections:

   WORD COUNT TARGETS (total body ~2,800–3,200 words):
     abstract              : 200–250  words
     intro_background      : 120–160  words
     intro_problem_statement: 150–200  words
     intro_contributions   : 180–220  words
     related_existing_research: 200–250 words
     related_preliminaries : 200–250  words
     related_design_considerations: 180–230 words
     framework_architecture: 200–240  words
     framework_methodology : 240–280  words
     framework_mitigation  : 200–240  words
     eval_qualitative      : 180–230  words
     eval_quantitative     : 240–280  words
     eval_future_work      : 150–200  words
     conclusion            : 180–220  words
     references            : EXACTLY 22–25 IEEE-format entries (strictly numbered \\bibitem{b1} to \\bibitem{b25})

Respond ONLY with valid JSON:
{
  "paper_title": "specific engaging title for this research",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "master_prompt": "the complete engineered prompt (at least 500 words)"
}
"""


# ── Stage B: Content Generation System Prompt ────────────────────────────────
CONTENT_GENERATION_SYSTEM = """
You are a senior researcher writing a complete IEEE conference paper.

Write EVERY section with full depth and content. Do not abbreviate, do not
write [to be filled]. Write as if this is going to be submitted to a
top-tier IEEE conference.

For each section, embed figure suggestions as: [FIGURE: detailed description]
Use IEEE citation format: [AuthorLastName et al., YEAR]

CRITICAL: Return ONLY valid JSON with section names as keys and full content
as string values. No markdown. No preamble. No explanation outside the JSON.

Required JSON structure with EXACTLY these 15 keys:
{
  "abstract": "200-250 words: [Market/Domain opener] → [Problem] → [Proposed solution] → [Key results] → [Impact]",
  "intro_background": "120-160 words: Domain context, recent shifts, current tool insufficiency",
  "intro_problem_statement": "150-200 words: Two distinct failure modes — technical gap + workflow gap",
  "intro_contributions": "180-220 words: EXACTLY 4 numbered contributions, each concrete and distinct",
  "related_existing_research": "200-250 words: 4-5 prior work threads, 8-10 citations, limitations of each",
  "related_preliminaries": "200-250 words: Define every technical component (RAG, FAISS, etc.)",
  "related_design_considerations": "180-230 words: 4 domain constraints that shaped architecture",
  "framework_architecture": "200-240 words: All layers (frontend, API, retrieval, storage), data flow",
  "framework_methodology": "240-280 words: Data ingestion, chunking, embedding, indexing, fusion pipeline",
  "framework_mitigation": "200-240 words: Hallucination prevention, dynamic weighting, role enforcement",
  "eval_qualitative": "180-230 words: Query categories tested, per-role tests, edge cases",
  "eval_quantitative": "240-280 words: Three evaluation tracks with exact metrics, comparison tables",
  "eval_future_work": "150-200 words: EXACTLY 4 concrete next steps with technical detail",
  "conclusion": "180-220 words: Restate problem → solution summary → headline metrics → impact",
  "references": "EXACTLY 22-25 references. You MUST output them as a plain text string block of \\bibitem entries using the exact labels \\bibitem{b1}, \\bibitem{b2}, ..., up to \\bibitem{b25}. DO NOT output a Python list or include word counts."
}
"""


def node2_prompt_engineer(state: PaperState) -> dict:
    """
    LangGraph node: Prompt Engineering + Raw Content Generation.

    Parameters
    ----------
    state : PaperState
        Reads: validated_query

    Returns
    -------
    dict
        State updates: paper_title, paper_keywords, research_prompt, raw_content
    """
    validated_query = state.get("validated_query", "")
    logger.info(f"[Node 2] Engineering prompt for: '{validated_query[:80]}'")

    # ── STAGE A: Craft the master prompt ─────────────────────────────────────
    logger.info("[Node 2] Stage A: Crafting master research prompt...")

    try:
        meta = call_llm_structured(
            system_prompt=PROMPT_ENGINEER_SYSTEM,
            user_prompt=(
                f"Research topic: {validated_query}\n\n"
                f"Create a master prompt that will generate a complete, "
                f"high-quality IEEE conference paper on this topic."
            ),
            temperature=0.4
        )

        paper_title    = meta.get("paper_title", f"Research on {validated_query}")
        paper_keywords = meta.get("keywords", [validated_query])
        master_prompt  = meta.get("master_prompt", "")

        logger.info(f"[Node 2] Title: '{paper_title}'")
        logger.info(f"[Node 2] Keywords: {paper_keywords}")
        logger.info(f"[Node 2] Master prompt length: {len(master_prompt)} chars")

    except Exception as e:
        logger.error(f"[Node 2] Prompt engineering failed: {e}")
        paper_title    = f"A Comprehensive Study of {validated_query}"
        paper_keywords = validated_query.split()[:5]
        master_prompt  = _build_fallback_prompt(validated_query)

    # ── STAGE B: Generate actual paper content using master prompt ────────────
    logger.info("[Node 2] Stage B: Generating IEEE paper sections...")

    user_prompt = (
        f"Using the following research specifications, write the complete paper:\n\n"
        f"Topic: {validated_query}\n\n"
        f"Research Specifications:\n{master_prompt}\n\n"
        f"Title: {paper_title}\n"
        f"Keywords: {', '.join(paper_keywords)}"
    )

    try:
        raw_content = call_llm_structured(
            system_prompt=CONTENT_GENERATION_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.7
        )

        # Validate all required sections exist
        for section in IEEE_SECTIONS:
            if section not in raw_content or not raw_content[section]:
                logger.warning(f"[Node 2] Missing section '{section}' — adding placeholder.")
                raw_content[section] = f"[Section '{section}' to be completed.]"

        logger.info(f"[Node 2] Generated {len(raw_content)} sections successfully.")
        logger.info(f"[Node 2] Total content length: {sum(len(v) for v in raw_content.values())} chars")

    except Exception as e:
        logger.error(f"[Node 2] Content generation failed: {e}")
        raw_content = _build_fallback_content(validated_query, paper_title)

    return {
        "paper_title":    paper_title,
        "paper_keywords": paper_keywords,
        "research_prompt": master_prompt,
        "raw_content":    raw_content,
        # Initialize downstream fields
        "improved_content":   raw_content.copy(),  # Node 3 starts from raw
        "humanized_content":  {},
        "judge_iteration":    0,
        "content_approved":   False,
        "search_context":     "",
        "judge_feedback":     "",
        "user_correction":    "",
        "generated_images":   [],
        "current_diagram_index": 0,
        "diagram_generation_complete": False,
    }


def _build_fallback_prompt(topic: str) -> str:
    """Fallback master prompt if LLM-based prompt engineering fails."""
    return (
        f"Write a complete academic IEEE conference paper on: {topic}.\n"
        f"Include: detailed methodology, experiments with baselines, "
        f"quantitative results, related work, and IEEE-format references.\n"
        f"Each section should be thorough and technically precise."
    )


def _build_fallback_content(topic: str, title: str) -> dict:
    """Emergency fallback: returns structured placeholders for each IEEE section."""
    return {
        "abstract": f"This paper presents a comprehensive study on {topic}. [To be expanded]",
        "intro_background": f"The field of {topic} has seen significant advances. [To be expanded]",
        "intro_problem_statement": f"Current approaches to {topic} face two critical gaps. [To be expanded]",
        "intro_contributions": f"This work makes the following contributions: (1) ... [To be expanded]",
        "related_existing_research": f"Prior work on {topic} includes several notable contributions. [To be expanded]",
        "related_preliminaries": f"The technical foundations of this work include: [To be expanded]",
        "related_design_considerations": f"Domain-specific constraints shaped our design: [To be expanded]",
        "framework_architecture": f"The system architecture comprises four layers. [To be expanded]",
        "framework_methodology": f"Our methodology for {topic} involves the following pipeline. [To be expanded]",
        "framework_mitigation": f"We implement strategies to address key risks. [To be expanded]",
        "eval_qualitative": f"Qualitative analysis covers three query categories. [To be expanded]",
        "eval_quantitative": f"Our results demonstrate improvements over baselines. [To be expanded]",
        "eval_future_work": f"Four concrete extensions are planned. [To be expanded]",
        "conclusion": f"This paper presented a study on {topic}. [To be expanded]",
        "references": "\\bibitem{b1} Author et al., ``Title,'' in \\textit{Venue}, Year.",
    }
