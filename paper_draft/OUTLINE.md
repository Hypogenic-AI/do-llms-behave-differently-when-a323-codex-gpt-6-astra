# Paper outline and evidence map

Working title: Prompt Formatting Changes Reported Authorship Judgments Without Establishing Behavioral Effects

1. Abstract (150–250 words): motivation; three complementary comparisons; Gemini format classification changes; six adjusted null results; failed semantic gates; probe/surface/residual contrast; unresolved mechanism.
2. Introduction: synthetic prompts and transfer; distinguish provenance, style, labels, judgments, activations, behavior; study diagram; quantitative preview; three contributions; roadmap.
3. Related work: evaluation awareness (Needham, Abdelnabi, Nguyen, Heidari); authorship/preference (Panickssery, Laurito, Xu); detection and surface confounds (Chen, Quaremba, Devbunova, Chalkidis); opinion measurement (Perez, Sharma).
4. Methodology: paired estimands; 180 benchmark/100 calibration requests; rewrite gates and amendment; exact wrappers and fixed-body labels; API/local separation; blind scoring and no-choice handling; paired uncertainty/multiplicity; train-only probes and norm-matched interventions.
5. Results: exact-content primary table and classification/behavior figure; unconditional alignment sensitivity; attrition table; exploratory paraphrase/label results; activation baselines table; fixed-text intervention table and computational effects.
6. Discussion: measurement dissociation and control failure; power/ceiling/readout limitations; scoring/budget/provider variability; bounded external validity and broader impact.
7. Conclusion: controlled source-judgment sensitivity, no established authorship-mediated behavior; replication priorities.
8. Appendix: full paraphrase/label/perception tables; exact protocol and scoring details; runtime/model versions/cost; reproducibility and artifact paths; amendment record.

Evidence: REPORT.md is the narrative authority. Numerical tables/plots use results/paired_effects.csv, attrition.csv, label_effects.csv, intervention_effects.csv, probe_metrics.json and saved report values. Metadata use notes/papers_catalog.json and datasets/README.md. Existing examples directory has only README.md; formatting follows templates/paper_writing/lab_style_guide.md and supplied command files. No new experiments or statistical tests are planned.

Presentation: required NeurIPS preamble, exact author block, modular sections, original native LaTeX method diagram, vector PDF plot, booktabs tables. Bold highest classification AUROC only; behavioral effects have no universally desirable direction and are not ranked. Final checks: cited keys, factual consistency, 150–250-word abstract, compilation log, rendered PDF, no overflow or unresolved references.
