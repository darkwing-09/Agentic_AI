"""
nodes/node4_humanizer.py
─────────────────────────
NODE 4 — Content Humanizer + Citation Injector

PURPOSE:
  Takes the judge-approved content and:
  1. Deeply humanizes ALL sections (removes AI patterns)
  2. Injects IEEE-style \\cite{bN} citations into every body section
     - Each section gets 2-5 citations
     - All 22-25 references are cited at least once across the paper
     - Citations are placed naturally alongside claims, methods, and comparisons

  This is a TWO-PASS process:
    Pass 1: Humanize each section (remove AI patterns, improve tone)
    Pass 2: Inject citations across ALL sections in one LLM call
            (ensures full reference coverage and no duplicates)

READS FROM STATE:
  - improved_content
  - paper_title
  - validated_query

WRITES TO STATE:
  - humanized_content
"""

import re
import json
import logging
from state import PaperState
from tools.llm_client import call_llm, call_llm_structured

logger = logging.getLogger(__name__)


# ── Humanization System Prompt ────────────────────────────────────────────────
HUMANIZER_SYSTEM = """
You are a senior academic editor who specializes in making AI-generated research
text sound genuinely human-written.

Your task: Rewrite the given research paper section to sound like it was written
by an expert human researcher — NOT a language model.

STRICT RULES — Remove ALL of these AI patterns:

❌ FORBIDDEN OPENINGS:
   Never start paragraphs with: "In recent years", "It is worth noting",
   "Notably", "Importantly", "Furthermore", "Moreover", "Additionally",
   "It should be noted", "One can observe"

❌ FORBIDDEN STRUCTURES:
   - Rule of three bullet lists as prose ("fast, efficient, and scalable")
   - Em-dash overuse (max 1 per page)
   - Hollow intensifiers ("truly", "certainly", "undoubtedly")
   - Passive voice chains (more than 2 in a row)
   - Meta-commentary ("This section discusses...", "As mentioned above...")

❌ FORBIDDEN HEDGING:
   - "may potentially", "could possibly", "it seems that"
   - "a number of", "various" (say exactly how many)

✅ INSTEAD USE:
   - Direct claims with evidence: "Our experiments show X = Y."
   - Varied sentence length: short punchy sentences mixed with longer ones
   - Specific numbers: "68.3% of samples" not "a majority"
   - First-person plural: "We observe", "Our results indicate", "We find"
   - Natural transitions: "This matters because...", "The key insight is..."
   - Technical precision: exact method names, paper titles, years

PRESERVE:
   - All technical content, equations, citations, and data
   - IEEE LaTeX formatting hints (keep [FIGURE: ...] markers)
   - All cited references and \\cite{} commands if present
   - Section structure

OUTPUT: Return ONLY the rewritten section text. No JSON. No labels. Just text.
"""


# ── Citation Injection System Prompt ──────────────────────────────────────────
CITATION_INJECTOR_SYSTEM = r"""
You are an IEEE citation placement specialist. Your ONLY job is to insert
\cite{bN} commands into existing academic text.

RULES:
1. Every body section (NOT abstract) must have 2-5 \cite{} commands.
2. ALL references from b1 to b{MAX_REF} must be cited at least once
   across the entire paper. No reference should go uncited.
3. Cite in ORDER of first appearance (b1 first, then b2, etc.).
4. Place citations NATURALLY — after claims, methods, comparisons, or
   facts that need attribution. Examples:
   - "RAG reduces hallucination \cite{b1}."
   - "FAISS enables sub-millisecond retrieval \cite{b4}."
   - "Prior work on legal NLP \cite{b10, b11} focused on..."
   - "The hybrid approach outperforms single-modality systems \cite{b5}."
5. Group related citations: \cite{b3, b4} when two papers support the same claim.
6. Do NOT add citations to the abstract section.
7. Do NOT change ANY text content. ONLY insert \cite{bN} commands.
8. Do NOT remove or modify existing \cite{} commands.

CITATION DISTRIBUTION GUIDE:
  intro_background:             2-3 citations (foundational papers)
  intro_problem_statement:      2-3 citations (papers showing the gap)
  intro_contributions:          1-2 citations (comparison references)
  related_existing_research:    5-8 citations (heavy — this is lit review)
  related_preliminaries:        3-5 citations (component papers: RAG, FAISS, etc.)
  related_design_considerations: 2-3 citations (domain constraints papers)
  framework_architecture:       2-3 citations (architectural inspiration)
  framework_methodology:        3-4 citations (methods papers)
  framework_mitigation:         2-3 citations (hallucination, safety papers)
  eval_qualitative:             2-3 citations (evaluation methodology)
  eval_quantitative:            2-3 citations (baselines, benchmarks)
  eval_future_work:             2-3 citations (extension opportunities)
  conclusion:                   1-2 citations (key impact references)

Respond ONLY with valid JSON:
{
  "cited_sections": {
    "section_key": "full section text with \\cite{bN} inserted...",
    ...
  },
  "citation_map": {
    "b1": ["intro_background", "related_existing_research"],
    "b2": ["related_existing_research"],
    ...
  },
  "total_unique_refs_cited": 24
}
"""


# Section-specific tone adjustments (IEEE structure)
SECTION_TONE_HINTS = {
    "abstract": (
        "Dense and informative. Every sentence carries weight. "
        "No sentence is wasted. Clear problem → method → result → impact. "
        "200-250 words. NO citations in abstract — this is IEEE standard."
    ),
    "intro_background": (
        "Set the domain context. What recent shift or change has "
        "created a new challenge? 120-160 words. "
        "MUST cite 2-3 foundational/domain papers."
    ),
    "intro_problem_statement": (
        "Two clearly distinct failure modes: one technical, one workflow. "
        "State real-world consequences of each. 150-200 words. "
        "MUST cite 2-3 papers that demonstrate the gap."
    ),
    "intro_contributions": (
        "Exactly 4 contributions as a numbered list. Each concrete and "
        "distinct. Include at least one metric per contribution. 180-220 words. "
        "Cite 1-2 comparison references."
    ),
    "related_existing_research": (
        "Critical analysis, not just listing papers. "
        "Group 4-5 prior work threads thematically. "
        "Point out limitations. 200-250 words. "
        "MUST cite 5-8 papers — this is the lit review section."
    ),
    "related_preliminaries": (
        "Define every technical component used. "
        "Each sub-paragraph = one component + how it is used. 200-250 words. "
        "MUST cite 3-5 source papers for each component (RAG, FAISS, etc.)."
    ),
    "related_design_considerations": (
        "At least 4 domain-specific constraints that shaped architecture. "
        "Be concrete about what each constraint prevents. 180-230 words. "
        "Cite 2-3 domain/regulatory papers."
    ),
    "framework_architecture": (
        "Describe all layers (frontend, API, retrieval, storage). "
        "Reference Fig. 1. Data flow between layers. 200-240 words. "
        "Cite 2-3 architectural inspiration papers."
    ),
    "framework_methodology": (
        "Step-by-step precision. Data ingestion, chunking, embedding, "
        "indexing, fusion pipeline. Reference Fig. 2. 240-280 words. "
        "Cite 3-4 methods papers."
    ),
    "framework_mitigation": (
        "Hallucination prevention, dual-corpus management, dynamic "
        "retrieval weighting, role enforcement. 200-240 words. "
        "Cite 2-3 safety/hallucination papers."
    ),
    "eval_qualitative": (
        "Query categories tested, per-role tests, edge-case failures. "
        "Acknowledge limitations honestly. 180-230 words. "
        "Cite 2-3 evaluation methodology papers."
    ),
    "eval_quantitative": (
        "Three evaluation tracks with exact metrics. "
        "Compare hybrid vs. single-modality. 240-280 words. "
        "Cite 2-3 baseline/benchmark papers."
    ),
    "eval_future_work": (
        "Exactly 4 concrete next steps with technical detail. "
        "State what problem each solves. 150-200 words. "
        "Cite 2-3 papers that motivate extensions."
    ),
    "conclusion": (
        "Concise synthesis. No new claims. Restate problem → solution → "
        "four headline metrics → impact. 180-220 words. "
        "Cite 1-2 key impact references only."
    ),
    "references": (
        "Keep references exactly as they are — do not reformat or add commentary."
    ),
}


def node4_humanizer(state: PaperState) -> dict:
    """
    LangGraph node: Humanizes all sections + injects IEEE citations.

    TWO-PASS PIPELINE:
      Pass 1: Humanize each section individually (tone, style, AI pattern removal)
      Pass 2: Inject \\cite{bN} citations across all sections in one call
              (ensures full 22-25 reference coverage)

    Parameters
    ----------
    state : PaperState
        Reads: improved_content, paper_title, validated_query

    Returns
    -------
    dict
        State updates: humanized_content
    """
    improved_content = state.get("improved_content", {})
    paper_title      = state.get("paper_title", "")
    validated_query  = state.get("validated_query", "")

    if not improved_content:
        logger.error("[Node 4] No content to humanize!")
        return {"humanized_content": {}, "error_message": "Node 4: no content received."}

    logger.info(f"[Node 4] Humanizing {len(improved_content)} sections...")

    # ══════════════════════════════════════════════════════════════════════════
    #  PASS 1: Humanize each section individually
    # ══════════════════════════════════════════════════════════════════════════
    humanized_content = {}

    for section_key, section_text in improved_content.items():
        if not section_text or len(str(section_text)) < 50:
            logger.warning(f"[Node 4] Skipping empty section: {section_key}")
            humanized_content[section_key] = section_text
            continue

        # References section: don't humanize — just clean up
        if section_key == "references":
            humanized_content["references"] = _clean_references(section_text)
            continue

        # Build section-specific prompt
        tone_hint = SECTION_TONE_HINTS.get(section_key, "Natural academic tone.")
        section_name = section_key.replace("_", " ").title()

        user_prompt = (
            f"Paper: {paper_title}\n"
            f"Topic: {validated_query}\n"
            f"Section: {section_name}\n"
            f"Tone requirement: {tone_hint}\n\n"
            f"--- ORIGINAL TEXT TO HUMANIZE ---\n"
            f"{section_text}\n"
            f"--- END ORIGINAL TEXT ---\n\n"
            f"Rewrite this section following all humanization rules. "
            f"Output ONLY the rewritten text."
        )

        logger.info(f"[Node 4] Pass 1 — Humanizing: {section_name}...")

        try:
            humanized_text = call_llm(
                system_prompt=HUMANIZER_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.8,
                max_tokens=4096
            )

            humanized_content[section_key] = humanized_text
            logger.info(
                f"[Node 4] ✅ {section_name}: "
                f"{len(section_text)} → {len(humanized_text)} chars"
            )

        except Exception as e:
            logger.error(f"[Node 4] Failed to humanize '{section_key}': {e}")
            humanized_content[section_key] = section_text

    logger.info("[Node 4] Pass 1 complete. Starting Pass 2 — citation injection...")

    # ══════════════════════════════════════════════════════════════════════════
    #  PASS 2: Inject \\cite{bN} citations across ALL sections
    # ══════════════════════════════════════════════════════════════════════════
    humanized_content = _inject_citations(
        humanized_content, paper_title, validated_query
    )

    logger.info(f"[Node 4] Both passes complete. {len(humanized_content)} sections ready.")

    return {"humanized_content": humanized_content}


def _inject_citations(
    sections: dict,
    paper_title: str,
    topic: str,
) -> dict:
    """
    Pass 2: Sends ALL sections to the LLM in one call to inject \\cite{bN}
    citations. This ensures:
      - Every section gets 2-5 citations (except abstract)
      - All 22-25 references are cited at least once
      - Citations appear in order of first appearance (b1, b2, ...)
      - No reference goes uncited

    Parameters
    ----------
    sections : dict
        Section key → humanized text (from Pass 1).
    paper_title : str
    topic : str

    Returns
    -------
    dict
        Same sections dict but with \\cite{bN} inserted throughout.
    """
    # Count references to know the range b1..bN
    ref_text = sections.get("references", "")
    max_ref = _count_references(ref_text)
    if max_ref == 0:
        logger.warning("[Node 4] Could not find any references. Defaulting to 10 for citations.")
        max_ref = 10
    
    logger.info(f"[Node 4] Citation injection — targeting b1..b{max_ref} based on actual references found")

    # Build compact section summaries for the citation injector
    # Send full text for body sections (abstract excluded from citation)
    body_sections = {
        k: v for k, v in sections.items()
        if k not in ("abstract", "references") and v and len(str(v)) > 30
    }

    # Build the prompt
    system_prompt = CITATION_INJECTOR_SYSTEM.replace("{MAX_REF}", str(max_ref))

    user_prompt = (
        f"Paper Title: {paper_title}\n"
        f"Topic: {topic}\n"
        f"Total references: {max_ref} (b1 through b{max_ref})\n\n"
        f"=== REFERENCE LIST ===\n"
        f"{ref_text[:2000]}\n\n"
        f"=== SECTIONS TO ADD CITATIONS TO ===\n"
        f"(Insert \\cite{{bN}} commands into these texts. "
        f"Do NOT modify any wording — ONLY add \\cite{{}} commands.)\n\n"
    )

    for section_key, section_text in body_sections.items():
        section_name = section_key.replace("_", " ").title()
        user_prompt += (
            f"--- {section_key} ---\n"
            f"{section_text}\n\n"
        )

    user_prompt += (
        f"\nInsert citations into ALL sections above. "
        f"Every reference from b1 to b{max_ref} must be cited at least once. "
        f"Return the JSON as specified."
    )

    try:
        result = call_llm_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=8000,
        )

        cited_sections = result.get("cited_sections", {})
        citation_map   = result.get("citation_map", {})
        total_cited    = result.get("total_unique_refs_cited", 0)

        logger.info(f"[Node 4] Citations injected — {total_cited} unique refs cited")

        # Merge: use cited version where available, keep original otherwise
        for section_key in body_sections:
            if section_key in cited_sections and cited_sections[section_key]:
                # Validate that citations were actually added
                cited_text = cited_sections[section_key]
                cite_count = len(re.findall(r'\\cite\{[^}]+\}', cited_text))
                if cite_count >= 1:
                    sections[section_key] = cited_text
                    logger.info(
                        f"[Node 4]   {section_key}: {cite_count} citations inserted"
                    )
                else:
                    logger.warning(
                        f"[Node 4]   {section_key}: LLM returned text without citations, "
                        f"keeping humanized version"
                    )

        # Log citation coverage
        all_cited_refs = set()
        for refs_list in citation_map.values():
            if isinstance(refs_list, list):
                # refs_list contains section names, key is ref id
                pass
        # Count from actual text
        for text in sections.values():
            for m in re.finditer(r'\\cite\{([^}]+)\}', str(text)):
                for ref in m.group(1).split(","):
                    all_cited_refs.add(ref.strip())

        logger.info(
            f"[Node 4] Citation coverage: {len(all_cited_refs)}/{max_ref} refs cited"
        )

        # Check for uncited references and log them
        expected_refs = {f"b{i}" for i in range(1, max_ref + 1)}
        uncited = expected_refs - all_cited_refs
        if uncited:
            logger.warning(
                f"[Node 4] ⚠️ Uncited references: {sorted(uncited)}. "
                f"Consider manual insertion or re-running."
            )

    except Exception as e:
        logger.error(f"[Node 4] Citation injection failed: {e}")
        logger.info("[Node 4] Proceeding with humanized text without injected citations.")

    return sections


def _count_references(ref_text: str) -> int:
    """Counts the number of \\bibitem entries or [N] references in the text."""
    if not ref_text:
        return 0

    # Count \bibitem{bN} entries
    bibitem_count = len(re.findall(r'\\bibitem\{', ref_text))
    if bibitem_count > 0:
        return bibitem_count

    # Count [N] entries
    bracket_count = len(re.findall(r'^\s*\[\d+\]', ref_text, re.MULTILINE))
    if bracket_count > 0:
        return bracket_count

    # Estimate from line count (each reference ≈ 1-2 lines)
    lines = [l.strip() for l in ref_text.split("\n") if l.strip()]
    return max(len(lines) // 2, 10)


def _clean_references(references_text: str) -> str:
    """
    Lightly cleans the references section:
    - Ensures each reference starts on its own line
    - Removes any AI-generated commentary or markdown code blocks
    """
    if not references_text:
        return references_text

    # Strip markdown formatting if the LLM included it
    references_text = references_text.replace("```latex", "").replace("```text", "").replace("```", "")
    
    # Strip any brackets with word counts like "[404 words]"
    import re
    references_text = re.sub(r'\[\d+\s*words?\]', '', references_text, flags=re.IGNORECASE)

    # Clean up extra blank lines
    lines = references_text.strip().split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Also ignore python list wrappers if it hallucinates them
        if stripped.startswith("['") or stripped.startswith('["') or stripped == "]" or stripped == "['" or stripped == '["':
            continue
            
        if stripped:
            cleaned.append(stripped)

    return "\n".join(cleaned)
