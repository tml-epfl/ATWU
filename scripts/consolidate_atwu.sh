#!/bin/bash
# Collect ATWU results into results_archive/final_runs_atwu_short.json
# (see scripts/consolidate_final_runs_short.py for the schema)
python scripts/consolidate_final_runs_short.py \
    --runs-dir saves/unlearn/ATWU_TOFU \
    --output "${1:-results_archive/final_runs_atwu_short.json}"
