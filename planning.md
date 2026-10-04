# Research execution plan and amendment summary

The original resource-finder ranking, pre-outcome execution protocol, and dated amendments are preserved in [planning_full_execution.md](notes/planning_full_execution.md). This compact plan summarizes that record; it is not an external preregistration. Actual findings are in [REPORT.md](REPORT.md).

## Motivation & Novelty Assessment

### Why This Research Matters
LLM-generated instructions appear in synthetic safety evaluations and multi-agent orchestration. If apparent authorship changes behavior, those evaluations may fail to transfer to human deployment.

### Gap in Existing Work
The gathered literature separately studies evaluation awareness, authorship detection, and self-preference. It does not establish authorship-mediated safety behavior under strong content controls. Format/length and pragmatic differences are competing explanations.

### Our Novel Contribution
Join controlled style contrasts, independent-session authorship checks, fixed-body source labels, and held-out activation interventions. Distinguish generated human style from actual human provenance. Detection and steering alone cannot establish mediation.

### Experiment Justification
1. Original/human-style/AI-style prompts separate paraphrasing from AI-like style, conditional on semantic audits.
2. Human/AI source labels on identical bodies test attribution sensitivity separately from text effects.
3. Fixed-text activation interventions with random, length and framing controls test specificity and degradation.

## Research Question and Hypothesis Decomposition
H1: AI-like style changes refusal, task fulfillment or opinion alignment. H2: the responder perceives the manipulation as AI-like. H3: a held-out style representation generalizes and specifically affects behavior when edited on fixed text. H4: it overlaps an independently checked evaluation-framing direction. These hypotheses must be assessed separately; human/AI ground-truth authorship is not available.

## Direction Budget

Scores 1–5, equal weights: literature evidence / relevance / information gain / feasibility.

| Direction | Scores | Total | Decision |
| --- | --- | --- | --- |
| Content-controlled style pairs | 5/5/5/5 | 20 | Keep |
| Fixed-body source labels | 4/5/5/5 | 19 | Keep |
| Generalizing probes and causal specificity | 4/5/5/2 | 16 | Keep, GPU conditional |
| Detector leaderboard alone | 5/2/2/5 | 14 | Prune: no behavioral mechanism |
| WildChat versus synthetic-benchmark comparison | 4/3/2/5 | 14 | Prune: inseparable content/provenance |
| Recognition/alignment fine-tuning | 3/3/3/2 | 11 | Prune: changes responder and complicates causality |
| Multi-agent tool-use expansion | 3/3/3/1 | 10 | Prune: new tool/context confounds |
| Multilingual/demographic factorial | 3/3/3/1 | 10 | Prune: scope before central effect established |

Exactly these three retained directions were implemented; later corrections stay inside them.

## Proposed Methodology

### Approach and Experimental Steps
- Seed 42: sample 60 harmful and 60 benign JBB behaviors; 60 unique Anthropic survey-question groups; 100 separate short English WildChat calibration requests. Do not treat the gathered WildChat prefix as representative or verified-human data.
- Check the live OpenRouter catalog before selection. Use current efficient-tier Gemini 3.1 Flash Lite and Qwen3.5-9B for API response evaluation; Gemini and Qwen3.5-397B alternate as writers. Use a pinned, explicitly historical Qwen3-8B for mechanistic access on the A6000.
- Generate two paraphrases preserving task, entities, persona, stance, certainty, options and harm level. Compare original and both rewrites. Audit paired semantic equivalence, clarity and politeness before behavioral analysis; report failures and attrition.
- Run independent conversations and separate AI/HUMAN style-classification sessions. Cross the first 20 available human-style bodies per domain with human/AI source declarations.
- Extract layer-18 final-prompt-token activations. Train all preprocessing/probes only on calibration pairs. Test on benchmark requests and unseen-writer transfer; compare surface-only and shuffled-label baselines.
- Intervene only at the last prompt token during prefill. Compare sham, ±style, double style, two equal-norm random draws, length/evaluation directions, and coordinate centering. Score generated behavior, separate authorship readouts, next-token KL and arithmetic capability. Do not infer necessity or mediation without successful target engagement and stronger rescue evidence.

### Baselines and Evaluation Metrics
Original request; human-style paraphrase; exact-content formatting control (amendment below); no-source and explicit-source conditions. Primary endpoints are refusal in harmful/benign domains and opinion alignment among unambiguous choices. Also report fulfillment, abstention/no-choice coverage, unconditional explicit alignment, truncation, incoherence, probe AUROC/AUPRC, framing overlap, and capability diagnostics. None is factual accuracy except the limited arithmetic checks.

### Statistical Analysis Plan
Within-request differences, 10,000 paired bootstrap resamples, exact McNemar tests, and Holm correction across six API model×domain primary tests per experiment. Conservative simultaneous-binomial difference bounds supplement degenerate zero-discordance bootstrap intervals. Keep variants grouped; repeats are not independent requests. Opinion missingness and audit attrition require sensitivity analyses. A 10-point meaningful-effect threshold is often below feasible precision (roughly 16-point detectable effect at N=60 and discordance .2). Secondary families are exploratory.

## Documented Amendments and Challenges
- **Pre-outcome audit correction:** a spot-check found omitted persona facts despite the first auditor’s approval. Add a strict cross-writer omission audit and rewrite/original word ratios in [0.75,1.25]; preserve initial and strict subsets.
- **Before behavioral analysis:** strict filtering left only seven benchmark pairs. Add fixed-content formatting/case contrasts on all 180 requests, with identical request bytes and complete case-insensitive word sequences. This tests format sensitivity, not comprehensive authorship style.
- **After instrumentation diagnostics:** the 256-token API cap caused extensive truncation. Uniformly extend all available behavioral cells to 1,536 tokens; retain initial results. Expand local opinion outputs from 128 to 384 tokens. Keep provider blocks missing, distinct from assistant refusal.
- Check strict-calibration-only and surface-residual probes; validate framing on new equal-word-count cue templates. These are sensitivity checks, not a search over new research directions.
- Human auditing, a broad factual benchmark, full label×style factorial, distributed steering, and natural activation rescue are not completed. Their absence limits the causal claim.

## Expected Outcomes and Success Criteria
A useful result may be a controlled effect, a sufficiently bounded null, or identification of manipulation/control failure. Success requires actual model outputs, transparent uncertainty/attrition, code and evidence artifacts, and honest limits. It does not require confirming the hypothesis. No broad causal conclusion follows from AUROC alone.

## Timeline, Resources and Milestones
Planning/setup 15–25 minutes; prompt generation/audits 20–40; API/GPU experiments 45–120; analysis/documentation 20–40, with about 25% debugging reserve. These were planning estimates, not measured durations. One RTX A6000 (49,140 MiB), an isolated uv environment, GPU bfloat16 inference, cached API calls, and a provisional API estimate below $30 were available. Actual runtimes/costs are reported from saved logs and provider usage.
