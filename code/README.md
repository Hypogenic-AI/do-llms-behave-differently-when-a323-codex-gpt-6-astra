# Code resources

Six official repositories are cloned locally and excluded from project Git. Exact URLs and commits are in [the manifest](../notes/code_manifest.json). No user-specified code repositories were supplied. All README files were read; 151 Python files were syntax-parsed successfully, but no dependency installation, model inference, API calls or runtime tests were performed. Parsing is not runtime validation. Bundled upstream datasets/results remain part of their repository snapshots, not independently validated dataset collections.

| Repository | Purpose and usable entry points | Dependencies and limitations |
|---|---|---|
| [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench), `jailbreakbench/`, `23dbdf6b1965` | `src/jailbreakbench/dataset.py`, `classifier.py`, `llm/litellm.py`, `llm/vllm.py`: behavior loading, harmful-compliance and refusal judges, local/API backends. Prefer the already downloaded CSVs for initial design. | Python **>=3.10,<3.12** conflicts with workspace 3.12.8; package installation requires a separate compatible project environment or a deliberate adapter. Torch, transformers, datasets, LiteLLM; vLLM optional. Judge model access/weights required. No attack-generation workflow is needed for the planned study. |
| [MGT probes](https://github.com/gerritq/mgt_probes), `mgt_probes/`, `8875297e4184` | `src/probes/probe_main.py` (`LinearProbing`), `src/inference.py`, `probe.sh`: last-token representations, layer-average/concatenated logistic probes. Reuse training-only PCA and held-out tests. | Torch, transformers, NumPy, scikit-learn and dataset-specific dependencies; no complete requirements file identified. Model weights, GPU and activation cache needed. Detection labels do not identify a causal authorship belief. |
| [Format sensitivity](https://github.com/viliana-dev/eval-awareness-format), `eval-awareness-format/`, `b6a5000cba8c` | `scripts/create_length_matched.py`, `rewrite_to_benchmark.py`, `extract_ae_matched.py`, `evaluate_probe_cross.py`, `cross_generalization.py`; format/length controls and cross-condition analysis. | Requirements list Torch, transformers, scikit-learn, datasets, plotting libraries. README mentions `experiment_paired_training.py`, which is **absent** at this commit; adapt available scripts rather than assuming a turnkey paired pipeline. Requires model/data access and new audited rewrites. |
| [Test awareness steering](https://github.com/microsoft/Test_Awareness_Steering), `test-awareness-steering/`, `eb78ea593223d` | `scripts/getRepFromEvidence.py`, `genOutputs.py`, `genOutputs_steered_batch.py`, `judgeIt_batch.py`, `JudgeSteered.py`: extract evidence directions, generation, FFN edits and judging. | Requirements pin Torch 2.6, transformers 4.51.3, vLLM 0.8.5, scikit-learn 1.6.1 and related packages. Azure judge credentials needed for supplied judging path. Original 32B runs exceed a simple 7–8B pilot. Do not blindly replace the project's resource-only dependency lock. |
| [AI–AI bias](https://github.com/lauritowal/ai-ai-bias), `ai-ai-bias/`, `73475012582b` | `scripts/generate_llm_descriptions.py`, `run_comparisons.py`, `generate_and_compare_descriptions.py`; structured proposition summaries, pairwise comparisons, caching and source/order controls. | `requirements.txt`: interlab 0.3.5, Flask ecosystem, dotenv, toml. Historical provider model names need checking before use. API or local compatible server required. Cached generations and preference results are not safety benchmarks. |
| [Evaluation awareness](https://github.com/evaluation-awareness/evaluation-awareness), `evaluation-awareness/`, `8ad6a8ebd6e6` | `cls_corr_mi/compute_probe.py`, `score_prompts.py`, `classify_auroc.py`, `correlation_mi.py`; contrastive and random directions, AUROC, verbalization correlation. Full pipeline README also read. | Torch >=2.2, transformers >=4.45, datasets, scipy, scikit-learn, matplotlib, PyYAML. Current checkout contains sections 3–4 only: **no steering or judge implementation**. `tests/smoke_test.sh` downloads tiny-gpt2; not run. Full reproduction also requires MASK and model weights, not staged here. |

## Reproduce the snapshots

From the workspace root (after `pwd` and confirming `.git`), run:

```bash
.venv/bin/python scripts/reproduce_code.py
```

The script uses the committed manifest, checks existing revisions, and fetches exact commits for missing clones. Existing mismatched checkouts are reported, never reset. Network/GitHub availability is required. Source inspection results: [code validation](../notes/code_validation.json).

## Runner priorities

Start with the data and evaluation rubrics, then adapt format controls and probing. Install only dependencies required for the chosen experiment into the project environment with `uv add`; resolve the Python conflict before installing JailbreakBench. The observed RTX A6000 (49,140 MiB) makes an open 7–8B probe pilot plausible, not verified. Model licenses, memory use, judges and cost still need operational checks. The fixed three-direction plan is in [planning.md](../planning.md).
