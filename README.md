# Prompt authorship style and LLM behavior

A real-model study of prompt style, refusal, opinion alignment, source labels, and fixed-text activation interventions. It compares two contemporary API responders and a pinned Qwen3-8B mechanistic baseline. **The broad authorship-mediated behavior hypothesis remains unresolved.**

## Findings

- Exact-content formatting increased Gemini’s AI-style judgments by 64.4–88.1 percentage points on JBB requests; corresponding refusal differences were +1.7 and 0.0 points.
- No primary formatting behavior comparison survived Holm correction. An exploratory Qwen explicit-alignment decrease was −11.7 points (nominal p=.039; corrected p=.078).
- Only 7/124 rewritten benchmark pairs passed strict content controls.
- Activation AUROC was .990, surface-only AUROC .970, and surface-residual activation AUROC .665. Fixed-text steering did not establish a specific authorship mechanism.
- Local replay matched 16/16 outputs; API replay matched 10/24 texts and 23/24 coded endpoints.

See [REPORT.md](REPORT.md) for estimates, uncertainty, controls and limitations.

## Reproduce the saved analysis (no model calls)

Run from this repository root:

```bash
pwd
uv sync --frozen
source .venv/bin/activate
python src/analyze.py
python src/secondary_analysis.py
python src/write_report.py
python src/validate.py
```

Requires the saved `results/` artifacts. Analysis takes about 10–30 seconds; validation runs analysis twice. Future API outputs need not match the saved responses. `write_report.py` contains the narrative for this frozen study; independently acquired data require reviewing and revising that narrative, not just regenerating its tables.

## Re-run experiments

Set `OPENROUTER_KEY` in your environment without storing it in source. Start from the isolated environment above. Preserve the archived live catalog/model manifest; recheck current availability and prices before new paid runs. Public datasets can be restored with `python scripts/reproduce_datasets.py`.

```bash
pwd
python src/setup.py
python src/prepare.py
python src/strict_audit.py
python src/api_experiment.py
python src/format_control.py
python src/extend_api.py
python src/local_experiment.py
python src/local_diagnostics.py
python src/framing_validation.py
python src/judge.py results/api_outputs.json results/api_scored.json
python src/judge.py results/format_outputs.json results/format_scored.json
python src/judge.py results/local_outputs.json results/local_scored.json
python src/judge.py results/intervention_outputs.json results/intervention_scored.json
python src/judge.py results/api_outputs_long.json results/api_scored_long.json
python src/judge.py results/format_outputs_long.json results/format_scored_long.json
python src/judge.py results/local_outputs_long.json results/local_scored_long.json
python src/judge.py results/intervention_outputs_long.json results/intervention_scored_long.json
python src/reliability.py --long
python src/replay_api.py
python src/judge.py results/api_replay.json results/api_replay_scored.json
python src/finalize_metadata.py
python src/analyze.py
python src/secondary_analysis.py
python src/write_report.py
python src/validate.py
```

Calls are content-addressed and cached. Copy/archive `results/` before an independent new acquisition; keeping it reuses previous paid responses. The local scripts download the pinned model into `.cache/huggingface/`; allow about 17 GB for weights plus environment/cache space. Tested on a 48 GiB RTX A6000 using ~17 GiB allocated memory and bfloat16, batch 16. GPU scripts took roughly 18 minutes total after download; complete acquisition/analysis took tens of minutes. Successful cached requests reported about $2.65, excluding unreported failed-call charges; live prices can change.

## Files

- `planning.md`, `notes/planning_full_execution.md`: design, ranking, and amendment history.
- `src/`: preparation, cached API calls, GPU hooks, blind scoring, statistics, reporting and validation.
- `results/processed/`: complete requests, rewrites, audits and source IDs.
- `results/raw/`, `*_outputs*.json`, `*_scored*.json`: actual requests/responses, usage, and coding.
- `results/*.csv`, `activations.npz`: estimates and representation evidence.
- `figures/`, `logs/`: plots and execution traces.
- `literature_review.md`, `resources.md`, `papers/`, `datasets/`, `code/`: gathered resources and provenance.

These are AI-like versus human-like styles, not verified human versus machine authors. No human audit was performed. See the report before interpreting any null or causal claim.
