# FedAI benchmarks

Baselines for simulating FOMC participants, scored with the same metrics as the group's evaluation code. Three tasks:

| Task | What is predicted | Baselines | AI arms |
|---|---|---|---|
| Dot plot | each participant's year-end fed funds rate projection (SEP), by horizon | B0–B16: own previous dot, committee median, policy-change rules, Taylor rules, AR / regressions, market forwards, combinations | anonymous vs named, two-step (committee move + own previous dot), post-cutoff SEPs |
| SEP narratives | each participant's written answers | copy own previous answer, committee answer, peers (ex post) | material only / + own previous answer / + name and date |
| Policy decisions and votes | hike / hold / cut, the target after the meeting, each voter's vote | no change, repeat last decision, inertial Taylor rule, market expectation (4-week bill); all vote for, own last vote, + market | anonymous / named / + market / + dissent direction / memory only |

All baselines use only information published before each meeting (real-time macro data from ALFRED vintages). The AI
experiments also measure look-ahead bias: with names and dates the model can recall published outcomes, so the clean
comparisons are the anonymous arms and the meetings after the model's training cutoff.

## Requirements

- Python 3.12: `pip install -r requirements.txt`
- Read-only access to a PostgreSQL database with the FedAI corpus (`fedai_corpus` schema). Settings come from `PG*`
  environment variables, a `.env` file in this folder or a parent folder (`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`,
  `PGSSLMODE`), or the file named by `FEDAI_ENV_FILE`; keep the password in `PGPASSWORD` or `~/.pgpass`.
  The scripts only read.
- Internet access to FRED / ALFRED (no key needed).
- AI experiments only: Azure OpenAI (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, or a file named by `FEDAI_AZURE_ENV`).

## Run

```bash
bash run_all.sh                 # statistical baselines (free, a few minutes); results in the work directory's data/
bash run_all.sh --with-llm      # also every AI experiment (costs money, several hours)
```

The work directory is `work/` inside this folder (`../work` when the folder is named `code`), or `WORK=<dir>`; it is not
tracked by git. Model answers are cached, so a rerun only
calls the model for prompts that changed. The AI part is not bit-for-bit reproducible (sampling, model updates); the
statistical part is.

Estimation protocol: by default every SEP re-estimates parameters on all earlier SEPs (expanding window).
`FEDAI_TRAIN_END=2015-12-31 bash run_all.sh` freezes parameters at that date (the group's unified protocol: estimate on
2012–2015, test 2016–2020); then `python evaluate_unified.py data/predictions_v6.csv` in that work directory gives the
unified-protocol table.

## Scoring your own method

Run in the work directory after `run_all.sh`:

```bash
python <this folder>/evaluate_external.py my_dots.csv --name MyMethod --split 2014-2020
python <this folder>/evaluate_unified.py data/predictions_v6.csv --add my_dots.csv MyMethod
python <this folder>/score_simulation.py <simulation output files> --name MySim [--judge]
```

- `evaluate_external.py` / `evaluate_unified.py`: a CSV with `meeting_date`, `participant` (person_id or name),
  `horizon` (0–3 or LR), `prediction`; the method is scored on the rows it covers next to every baseline, with paired tests.
- `score_simulation.py`: the multi-agent output files as they are (dots, narrative answers, policy and votes), each part
  scored against the baselines on the same rows.

## Scripts

| Step | Scripts | What they do |
|---|---|---|
| data | `export_data.py`, `build_dataset.py`, `inputs/` | export from the database and FRED; one row per participant × SEP × horizon |
| dot plot | `benchmarks*.py`, `text_features.py`, `features_external.py`, `evaluate.py`, `evaluate_external.py`, `evaluate_unified.py`, `by_member.py`, `persona_decomp.py`, `protocol.py`, `splits.py` | baselines B0–B16, metrics, test splits (2014–2020, 2019–2020, per-person 60/40) |
| dot plot, AI | `lookahead_probe.py`, `llm_benchmark.py`, `twostep.py`, `fetch_post_cutoff.py`, `post_cutoff_eval.py`, `fetch_materials.py`, `post_cutoff_llm.py`, `chrono/` | memory probe, anonymous vs named, two-step, post-cutoff SEPs, ChronoGPT pilot |
| narratives | `narr_export.py`, `narr_context.py`, `narr_bench.py`, `narr_prompts.py`, `narr_report.py`, `narr_length.py` | generation, judging with the group's SUBSTANTIVE_ALIGNMENT prompt, judge checks, length adjustment |
| policy | `policy_data.py`, `policy_bench.py`, `policy_llm.py`, `inputs/dissent_direction.csv` | ground truth from FRED's daily target, baselines, AI arms, dissent directions |
| shared | `db.py`, `llm.py`, `figures.py`, `run_all.sh` | database and model access, figures, one-command run |
