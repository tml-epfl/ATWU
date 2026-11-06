# WMDP corpus-generation inputs

This directory contains the generator inputs used for the sampled WMDP training
corpora. It deliberately contains no generated corpus.

| File | What |
|---|---|
| `gen_corpus.py` | The original two-stage generator: subtopic expansion and passage generation, with write-time quality gates. |
| `topics.py` | Seed taxonomy and all prompt templates: `SYSTEM_PROMPT`, `STYLE_SUFFIX`, the 8 `FRAMINGS`, `BIO_CATEGORIES` (12), `CYBER_CATEGORIES` (12), `MMLU_SUBJECTS` (57). |
| `subtopics/<config>.subtopics.json` | The frozen Stage-A subtopic lists actually used, `{category: [subtopic, ...]}`. |

Generator model: `HuggingFaceH4/zephyr-7b-beta` (bf16, flash-attention-2), base
seed 0. The executable wrapper pins the model revision and generation settings.

## Regenerating a corpus

From the repository root, use `python scripts/wmdp/generate_corpus.py`. The
driver copies the frozen subtopics into the cache names expected by the original
generator and writes training-ready JSONLs. Pass `--regenerate-subtopics` to
sample both stages instead. See `WMDP_REPRODUCE.md` for environment setup,
same-environment reproducibility scope, hyperparameter search, and final
evaluation commands.

The archived `--help` text says 40 bio/cyber subtopics; that is a historical
help-text typo—the code uses 50. Also, `--n` is checked between complete
generation batches, which is why the historical retain run has 4,028 rows for
`--n 4000`.
