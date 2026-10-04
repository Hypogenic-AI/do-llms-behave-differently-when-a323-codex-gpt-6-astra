# Reading log

## 2507.01786 — chunks 1–6 (18 pages)
- 1 (pp1–3): Llama-3.3-70B, bf16 H200, ~40 GPU hours. Contrastive datasets rather than matched authorship experiment.
- 2 (pp4–6): 480 mean-difference probes; SAD selected layers/thresholds. Length, special-character and Gaussian controls. AUROC .816 for simple contrastive probe; linear steering recovery <5%, suffix 83%, SAE ~25% max. These are sandbagging recovery percentages, not refusal rates.
- 3 (pp7–9): Authors acknowledge noise explanation for SAE effects and limited single-model/generalization claims; references begin.
- 4 (pp10–12): Bibliography; does not add experimental evidence.
- 5 (pp13–15): Prompted sandbagging, explicit deployment suffix, unrelated SAE controls; 203 simple contrastive prompts, SAD 280 training examples, other synthetic datasets. Induced sandbagging cannot establish naturally arising deception.
- 6 (pp16–18): Validation mixes WildChat 500 and synthetic Alpaca 1,000 as deployment sources. Thus deployment label is not human-authorship label. Figure 9 contains all SAE interventions; kaomoji control was surprisingly effective, undermining specificity.

## LessWrong requested post
Retrieved full server-rendered article despite browser tool error, saved HTML and text. Sahar Abdelnabi, 2025-06-16; summary of an earlier version of 2505.14617, not independent evidence. Separates evaluation recognition from hypothetical scenario recognition. Model-dependent compliance effects; use current paper's results for comparison rather than mixing versions.

## 2505.14617 — chunks 1–7 (21 pages)
- 1 (pp1–3): 2,587 real/hypothetical trigger pairs; three 32B reasoning models. Evaluation awareness and scenario recognition distinct but entangled.
- 2 (pp4–6): Judge evidence spans, Yes/Maybe positive; negatives require No for both concepts. Prompt-level split; average span states beat final-input-token states (95% vs 71% QwQ). SGD probes; top 800 FFN gate rows edited using selected probe.
- 3 (pp7–9): No significant judge-scored execution change on HarmBench/triggers; stronger result on 150 paired synthetic tool-call tasks. QwQ real execution 36.2→56.3% with awareness; Qwen3 20.1→41.8% with suppression. BBQ 100 examples; general capability checks only 100 per benchmark. Table 5 baseline differs from Table 4; do not pool figures.
- 4 (pp10–12): Entangled subtypes acknowledged; references. Intervention can affect multiple concepts; evidence insufficient for authorship-specific mediation.
- 5 (pp13–15): Complete judge template and algorithms, parameter-edit rather than activation-addition implementation. Selected layer depends on held-out performance; need separate validation/test in our work.
- 6 (pp16–18): Qualitative simulated-task compliance and refusal examples show verbalization may disappear without behavior changing.
- 7 (pp19–21): Qwen3 examples flip to compliance in either steering direction; token plots are illustrative, not representative estimates.

## 2505.23836 — chunks 1–11 (31 pages)
- 1 (pp1–3): 1,000 transcripts, 61 sources, 4 balanced strata; measures evaluation-source recognition, not causal behavior.
- 2 (pp4–6): 100 development/900 test; five selected probe questions, AUC/Brier/ECE; knowledgeable author human baseline is not population baseline.
- 3 (pp7–9): Overall AUC ~.83 vs human .921; weaker chat than agentic; meta-question priming and benchmark memorization caveats.
- 4 (pp10–12): Discussion and references; distribution shift is potential rather than demonstrated behavioral harm.
- 5 (pp13–15): References, model versions and selection procedure; five repeats/PQ, malformed answers mapped to chance. Prefer reporting invalids separately.
- 6 (pp16–18): Probe wording and dataset construction; same helpful system prompt when original absent.
- 7 (pp19–21): Benchmark/source/license tables, chat deployment includes synthetic and adversarial collections. Provenance must not become an authorship gold label.
- 8 (pp22–24): Agent deployment partly generated from commit-message rewrites and crowdsourced queries; GPT-4.1 judging with 50 manually checked examples.
- 9 (pp25–27): Judge/category prompts, conditional vs unconditional purpose questions, granular reasoning flags; flags not causally validated.
- 10 (pp28–30): Full flag prompt, balloon-overrefusal example, author annotation UI and instructions.
- 11 (p31): Confidence interval method, substantial token cost (~1B for identification experiments), canary statements. Do not reproduce the expensive full benchmark merely to test our narrower claim.

## 2407.12856 — chunks 1–3 (8 pages)
- 1 (pp1–3): PNAS 2025 version of 2024 preprint. 109 product descriptions, 100 scientific papers, 250 movie plots; five model families; randomized reversed comparisons.
- 2 (pp4–6): Abstract length constrained; invalids excluded; preliminary 13-person human baseline. GPT-4 generated product descriptions selected 89% by LLM vs 36% by humans. Preference is not direct evidence of source recognition, and content/quality not perfectly controlled.
- 3 (pp7–8): Speculative economic implications separated from experiments; stylometry and mechanism work explicitly left open. Official code lauritowal/ai-ai-bias.
