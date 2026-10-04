# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-04T02:16:11.814288Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/do-llms-behave-differently-when-a323-codex-gpt-6-astra`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
Resource-finder Phases 1–5 complete: 24 PDFs (all ten specified papers fully chunk-read), three dataset collections, six pinned code repositories. Handoff: resources.md, literature_review.md, planning.md, per-directory READMEs. Exactly three directions retained: controlled style pairs, crossed authorship labels, generalizing probes with causal-specificity interventions; rejected directions recorded. No experiments run.
Integrity evidence: notes/final_validation.json, pdf_validation.json (677 readable pages), dataset_validation.json, code_validation.json (151 syntax parses), manifests and reproduction scripts. Data/code excluded from Git; samples total 3,689 bytes. Source PDF 2407.12856 has a nonfatal blending-color-space warning; readable text/rendered chunks inspected, original preserved.
USE datasets/anthropic_verified/ (30,051 rows): requested HF philosophy duplicated NLP; repaired from pinned original source. JBB: 100 harmful/100 benign +300 judge rows. WildChat: only 1,000 unpinned prefix rows, 995 unique hashes/523 English; not representative or verified human authorship.
Next: experiment_runner reads these artifacts, constructs and audits semantic pairs, predeclares metrics/grouped splits and costs, runs behavioral pilot, then controlled representation work if feasible. Known limits: no factual-accuracy benchmark/model weights/runtime tests; JailbreakBench excludes Python 3.12; format repo lacks advertised paired-training script; awareness repo lacks steering. RTX A6000 49,140 MiB observed, 7–8B feasibility untested. No unresolved download failure or ranking change. Completion marker: .resource_finder_complete.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
Phases 0–6 complete. Actual results and limitations: REPORT.md; reproduction: README.md; full original protocol/amendments: notes/planning_full_execution.md. 180 benchmark +100 calibration requests; strict audit only 7/124 rewritten benchmark pairs. Exact-content formatting shifts Gemini AI-style judgments strongly; six primary behavioral tests all Holm p=1. Exploratory Qwen explicit-alignment effect −11.7 pp (p=.039, Holm=.078), not confirmed. Activation AUROC .990 versus surface .970; residual .665. Fixed-text, norm-matched interventions did not establish authorship mediation; weak framing transfer (.616 AUC). Validation passed all 13 software/artifact checks, including identical two-run analysis hashes and no running experiment jobs (results/validation.json). Local replay 16/16 exact; API 10/24 texts and 23/24 endpoints match. 9,612 cached request records, reported successful usage $2.6493; costs/retries caveats documented. All generated sections/resource-finder notes preserved. Research uncertainty remains as a scientific finding, not unfinished execution. Final evidence index: results/execution_summary.json.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
