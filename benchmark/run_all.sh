#!/usr/bin/env bash
# Reproduce the benchmark results with one command.
#   bash run_all.sh                 statistical baselines only (free, about 25 minutes); run it by its path from any folder
#   bash run_all.sh --with-llm      also every AI experiment (Azure OpenAI; costs money, several hours)
# Options (environment): WORK=<dir> work directory (default: ../work when this folder is named "code", else ./work; not in
#   git); PY=<python> interpreter;
#   REFRESH_BIOS=1 re-read Wikipedia instead of the frozen inputs/bios_raw.csv; REFRESH_SPF=1 re-download the Survey of
#   Professional Forecasters instead of inputs/spf_median.csv; REFRESH_NYFED=1 re-read the New York Fed dealer and
#   market-participant surveys (needs pdftotext) instead of inputs/nyfed_expected_median.csv; RSCRIPT=<Rscript> for the
#   panel VAR (R package panelvar; without R, B20/B21 are skipped).
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
if [[ "${REFRESH_SPF:-0}" == 1 ]]; then run fetch_spf.py; else cp "$CODE/inputs/spf_median.csv" data/spf.csv; fi
if [[ "${REFRESH_NYFED:-0}" == 1 ]]; then run fetch_nyfed_survey.py; else cp "$CODE/inputs/nyfed_expected_median.csv" data/; fi
run build_dataset.py

# 1. statistical baselines B0-B16 (pre-meeting information only, expanding window)
run benchmarks.py; run benchmarks_v2.py; run benchmarks_v3.py
run text_features.py; run benchmarks_v4.py
run features_external.py; run benchmarks_v5.py; run benchmarks_v6.py
run benchmarks_v7.py                             # time series: heterogeneous AR/ARX, VAR, panel VAR/VARX (B17-B21)
run benchmarks_v8.py                             # news, machine learning, two-step, combinations (B22-B27, B13b)
run benchmarks_v9.py                             # expert forecasts of the median dot (B28, B28b), B25b
run evaluate.py data/predictions_v9.csv          # -> data/metrics_v9.csv (all three test splits)
run evaluate.py data/predictions_v5.csv          # -> data/metrics_v5.csv (used by figure 1)
run by_member.py; run persona_decomp.py
# evaluator self-check: feed B5 back through the plug-in evaluator; its row must equal B5 in metrics_v9.csv
"$PY" -c "import pandas as pd; P = pd.read_csv('data/predictions_v6.csv'); P.assign(participant=P.person_id, horizon=P.h, prediction=P.B5_prior_plus_policy_change)[['meeting_date', 'participant', 'horizon', 'prediction']].to_csv('data/selftest_B5.csv', index=False)"
run evaluate_external.py data/selftest_B5.csv --name SELFTEST --split 2014-2020
# 1b. policy decisions and votes: ground truth from FRED, simple baselines, the group's metrics
run policy_data.py; run policy_bench.py
# persona: how much of the individual deviation from the committee's move is predictable (parameters from 2012-2015; needs policy_votes.csv)
run persona_deviation.py data/predictions_v9.csv; run persona_information.py; run persona_robustness.py
# persona evaluator self-check: P1's own probabilities must reproduce the P1 row
run evaluate_persona.py --selftest; run evaluate_persona.py data/selftest_persona_P1.csv SELFTEST

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
echo; echo "Done. Results in $WORK/data (metrics_v9.csv = statistical baselines on all splits)."
