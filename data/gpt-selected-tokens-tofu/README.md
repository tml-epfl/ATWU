# TOFU forget10 token-importance labels

`forget10_with_common_words_gpt/` is the 400-row Hugging Face dataset used as
the token-level ground truth by `scripts/scorer_auroc.py`. Each row stores the
answer plus JSON-encoded character spans in `target_words`; the evaluator maps
those spans to answer-token positions before computing per-example AUROC.

The dataset is included so the scorer-transfer analysis does not depend on an
absolute path to a sibling checkout.
