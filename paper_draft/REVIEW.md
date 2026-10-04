# Final manuscript review

## Resources and style

- Read `.codex/skills/paper-writer/SKILL.md` and `templates/paper_writing/lab_style_guide.md` before drafting.
- Inspected `paper_examples/`: only a README was supplied; no example sections, tables, or macros were available to browse.
- Verified and imported all three supplied command templates. Customized project terms; retained shared templates.
- Used the citation-manager skill and saved catalog metadata; no example content was copied.

## Content and language

- Abstract: 196 whitespace-delimited words (within 150–250).
- Exact author block: `\author{Dang Nguyen and NeuriCo}`.
- Introduction follows motivation, gap, approach with figure, quantitative preview, three specific contributions, and roadmap.
- Related work is organized by awareness, recognition/preference, detection/confounds, and opinion measurement.
- Main claims remain bounded: generated human style is not verified human provenance; no primary significance does not establish general equivalence; activation discrimination does not establish mediation.
- Preserved semantic attrition, conditional-choice selection, unconditional Qwen signal with adjusted p=.0781, scoring uncertainty, local truncation, and API replay variability.
- Distinguished six-test primary correction, two-model opinion sensitivity correction, and exploratory source labels/interventions.
- Checked intervention scaling and centering formula against implementation. Clarified 2,000 bootstrap draws for probe AUROC versus 10,000 for paired behavior.
- Numerical behavioral, perception, attrition, label, and intervention tables generated directly from saved CSVs; probe numbers checked against report and saved metrics.

## Formatting and compilation

- Required NeurIPS `[final]` style, hyperref, booktabs, graphicx, amsmath/amssymb, command imports, and plainnat are present.
- Native LaTeX method diagram and vector statistical figure have self-contained captions; figure labels visually inspected at rendered size.
- Tables use booktabs without vertical lines. Highest probe AUROC is bold; behavioral extrema are not falsely framed as performance wins.
- Citation validator: 18 entries, zero errors, zero warnings.
- Final PDF: 14 pages. Inspected rendered contact sheet and individual title, plot, and appendix pages; fixed an overlapping plot title and awkward appendix spacing.
- Final LaTeX log: no errors, undefined citations/references, overfull boxes, or rerun warnings. Some harmless underfull-box notices remain.
- PDF contains clickable citation and cross-reference links.
- All sections complete; no prose placeholders. Abstract follows the standard unnumbered conference environment.

## Build correction

The supplied style left `\@trackname` undefined under the required `[final]` option. Added a neutral fallback notice in `main.tex`, preserving the required package line and avoiding an unsupported conference-acceptance implication.
