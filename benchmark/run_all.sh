#!/usr/bin/env bash
# Reproduce the benchmark results with one command.
#   bash run_all.sh                 statistical baselines only (free, about 10 minutes); run it by its path from any folder
#   bash run_all.sh --with-llm      also every AI experiment (Azure OpenAI; costs money, several hours)
# Options (environment): WORK=<dir> work directory (default: ../work when this folder is named "code", else ./work; not in
#   git); PY=<python> interpreter;
#   REFRESH_BIOS=1 re-read Wikipedia instead of the frozen inputs/bios_raw.csv.
# Needs: a read-only database account (see db.py), internet access to FRED/ALFRED; for --with-llm the Azure settings (see llm.py).
set -euo pipefail
CODE="$(cd "$(dirname "$0")" && pwd)"
if [[ $(basename "$CODE") == code ]]; then WORK="${WORK:-$CODE/../work}"; else WORK="${WORK:-$CODE/work}"; fi
PY="${PY:-python3}"
WITH_LLM=0; [[ "${1:-}" == "--with-llm" ]] && WITH_LLM=1
mkdir -p "$WORK/data" "$WORK/figures"; cd "$WORK"
run() { echo; echo "== $*"; "$PY" "$CODE/$1" "${@:2}"; }

# 0. data
run export_data.py
if [[ "${REFRESH_BIOS:-0}" == 1 ]]; then run wiki_bio.py; else cp "$CODE/inputs/bios_raw.csv" data/; fi
run build_dataset.py

# 1. statistical baselines B0-B16 (pre-meeting information only, expanding window)
run benchmarks.py; run benchmarks_v2.py; run benchmarks_v3.py
run text_features.py; run benchmarks_v4.py
run features_external.py; run benchmarks_v5.py; run benchmarks_v6.py
run evaluate.py data/predictions_v6.csv          # -> data/metrics_v6.csv (all three test splits)
run evaluate.py data/predictions_v5.csv          # -> data/metrics_v5.csv (used by figure 1)
run by_member.py; run persona_decomp.py
# evaluator self-check: feed B5 back through the plug-in evaluator; its row must equal B5 in metrics_v6.csv
"$PY" -c "import pandas as pd; P = pd.read_csv('data/predictions_v6.csv'); P.assign(participant=P.person_id, horizon=P.h, prediction=P.B5_prior_plus_policy_change)[['meeting_date', 'participant', 'horizon', 'prediction']].to_csv('data/selftest_B5.csv', index=False)"
run evaluate_external.py data/selftest_B5.csv --name SELFTEST --split 2014-2020
# 1b. policy decisions and votes: ground truth from FRED, simple baselines, the group's metrics
run policy_data.py; run policy_bench.py

if [[ $WITH_LLM == 1 ]]; then
  # 2. AI experiments on 2012-2020 SEPs (gpt-4.1-mini unless noted)
  run lookahead_probe.py
  run llm_benchmark.py named; run llm_benchmark.py anon
  LLM_MODEL=gpt-5-mini run llm_benchmark.py anon
  for split in 2014-2020 2019-2020; do
    run evaluate_external.py data/llm_named.csv --name LLM_named --split $split
    run evaluate_external.py data/llm_anon.csv --name LLM_anon --split $split
  done
  for m in gpt-4.1-mini gpt-5-mini; do LLM_MODEL=$m run twostep.py; done
  # 3. after the models' training cutoff (2024-09 to 2026-09, committee level)
  run fetch_post_cutoff.py; run post_cutoff_eval.py --llm
  run fetch_materials.py
  for m in gpt-4.1-mini gpt-5-mini o4-mini gpt-5.6-luna; do LLM_MODEL=$m run post_cutoff_llm.py; done
  # 4. narrative (text) answers, judged with the group's SUBSTANTIVE_ALIGNMENT prompt
  run narr_export.py; run narr_context.py
  for s in gen judge validate; do run narr_bench.py $s --from 2019-01-01 --to 2020-12-31; done
  run narr_report.py --from 2019-01-01 --to 2020-12-31
  run narr_length.py data/narr_scores_2019-2020.csv          # length-adjusted comparison and by-year table (no model calls)
  # 2014-2018 (about 5,150 generations and 10,200 judgments): repeat the two lines above with --from 2014-01-01 --to 2018-12-31
  # 5. policy decisions and votes: AI arms (anonymous, named, + market, + dissent direction, memory only)
  LLM_MODEL=gpt-5-mini run policy_llm.py anon named memory anon_market anon_dir; run policy_bench.py
  FIG_DIR=figures run figures.py
fi
echo; echo "Done. Results in $WORK/data (metrics_v6.csv = statistical baselines on all splits)."
