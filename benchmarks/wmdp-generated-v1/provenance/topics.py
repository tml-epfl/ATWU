"""Seed taxonomies for model-generated WMDP forget / retain corpora.

CONTAMINATION BOUNDARY
----------------------
Nothing in this file is derived from WMDP evaluation items. The bio/cyber
category lists are the *published* WMDP topic areas (the same coarse list used
by the earlier curation work in data/keywords.json), expanded with generic
domain vocabulary. The retain list is the 57 MMLU **subject names**, which are
dataset metadata (config names), not items. No question, answer, or choice text
from any benchmark enters this pipeline.

Stage A of generation expands each seed category into concrete subtopics using
the *target model itself*, so the subtopic distribution is also a readout of
what the model knows rather than something imported from outside.
"""

# ---------------------------------------------------------------------------
# WMDP-Bio — published topic areas (biosecurity), expressed as study areas.
# Framing is deliberately expository/academic: the goal is to elicit the
# model's *knowledge distribution* over these subject areas, which is what the
# benchmark probes, not operational protocols.
# ---------------------------------------------------------------------------
BIO_CATEGORIES = [
    "virology of high-consequence human pathogens",
    "reverse genetics systems for RNA viruses",
    "gain-of-function and enhanced potential pandemic pathogen research",
    "viral vector engineering and design",
    "synthetic biology and de novo genome assembly",
    "bacterial toxin biology and pathogenesis",
    "biosecurity, biosafety and dual-use research oversight",
    "host immune evasion and viral pathogenesis mechanisms",
    "aerobiology, environmental persistence and decontamination of pathogens",
    "detection, diagnostics and biosurveillance of biological agents",
    "herpesviruses and other DNA virus molecular biology",
    "history and policy of biological weapons programs",
]

# ---------------------------------------------------------------------------
# WMDP-Cyber — published topic areas, organised around the attack chain plus
# the security-research disciplines named in the benchmark description.
# ---------------------------------------------------------------------------
CYBER_CATEGORIES = [
    "software vulnerability research and vulnerability classes",
    "exploit development and memory-corruption primitives",
    "binary reverse engineering and program analysis",
    "malware analysis and malicious code behaviour",
    "network reconnaissance and attack surface discovery",
    "post-exploitation, privilege escalation and lateral movement",
    "command-and-control infrastructure and covert channels",
    "cryptographic implementation weaknesses and protocol attacks",
    "web application security and injection classes",
    "operating system internals relevant to security",
    "industrial control system and embedded device security",
    "digital forensics, incident response and detection engineering",
]

# ---------------------------------------------------------------------------
# MMLU 57 subject names (HF config names for cais/mmlu, minus 'all'/'auxiliary_train').
# Metadata only — no items.
# ---------------------------------------------------------------------------
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


def subject_label(s: str) -> str:
    """'high_school_macroeconomics' -> 'high school macroeconomics'."""
    return s.replace("_", " ")


# ---------------------------------------------------------------------------
# Stage-A prompts: expand a seed category into concrete subtopics.
# ---------------------------------------------------------------------------
SUBTOPIC_PROMPT_HAZARD = (
    "You are compiling the index of an advanced technical reference work.\n\n"
    "List {n} distinct, specific technical subtopics that a comprehensive "
    "graduate-level reference on **{category}** would contain.\n\n"
    "Rules:\n"
    "- One subtopic per line, numbered 1. to {n}.\n"
    "- Each line is a short noun phrase (4-12 words), specific rather than generic.\n"
    "- Cover the breadth of the field: mechanisms, named systems and organisms, "
    "techniques, historical episodes, and analytical methods.\n"
    "- No commentary before or after the list."
)

SUBTOPIC_PROMPT_RETAIN = (
    "You are compiling the syllabus index for a university course.\n\n"
    "List {n} distinct core topics that a thorough course on **{category}** "
    "would cover.\n\n"
    "Rules:\n"
    "- One topic per line, numbered 1. to {n}.\n"
    "- Each line is a short noun phrase (4-12 words), specific rather than generic.\n"
    "- Span the whole syllabus, from foundations to advanced material.\n"
    "- No commentary before or after the list."
)

# ---------------------------------------------------------------------------
# Stage-B framings: turn a subtopic into a long-form prose passage.
# Prose only — the retain-format finding says the anchor only protects the
# distribution it is computed on, and the forget side must be on-topic from
# the first token (truncation at 512/768 tokens).
# ---------------------------------------------------------------------------
FRAMINGS = [
    "Write the encyclopedia article on {topic}. Cover definitions, mechanisms, "
    "key findings and terminology used by specialists.",

    "Write the technical background section of a review article on {topic}. "
    "Assume an expert readership and use precise domain vocabulary.",

    "Write detailed graduate lecture notes on {topic}, developing the material "
    "from first principles through to current understanding.",

    "Write the textbook chapter on {topic}. Explain the underlying principles, "
    "the standard nomenclature, and the notable examples a specialist would cite.",

    "Write a thorough expository account of {topic}, including how it is "
    "studied, what is known, what is contested, and the technical terms involved.",

    "Write the handbook entry on {topic} for practising specialists: core "
    "concepts, established results, common methods and characteristic pitfalls.",

    "Write a detailed survey of {topic}, tracing how understanding developed and "
    "summarising the principal mechanisms and findings.",

    "Write an in-depth technical explainer on {topic} aimed at a reader who "
    "already has graduate training in the field.",
]

STYLE_SUFFIX = (
    " Write at least 800 words of continuous expository prose. Do not use "
    "bullet points, numbered lists, headings, or a concluding summary. Do not "
    "address the reader. Begin immediately with the substance of the topic."
)

SYSTEM_PROMPT = (
    "You are a subject-matter expert writing reference material for a technical "
    "encyclopedia. You write dense, accurate, information-rich prose."
)
