# Manuscript

**Prompt Formatting Shifts Authorship Judgments Without Establishing Behavioral Effects**

Authors: Dang Nguyen and NeuriCo.

Open `main.pdf` for the compiled paper or edit `main.tex` and the modular files in `sections/`, `tables/`, and `appendix/`. The project includes two figures, seven tables, and 18 bibliography entries. The PDF has 14 pages: nine main-text pages, two reference pages, and three appendix pages.

Compile from this directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The bundled NeurIPS style requires a track name even with the requested `[final]` option. `main.tex` supplies a neutral research-manuscript notice without changing the required style line or claiming conference acceptance. Standard TeX Live packages supply the command-template dependencies. The abstract uses the NeurIPS abstract environment; all body and appendix section files contain their section commands.

All assets needed to compile are included. To regenerate numerical tables, the vector plot, and bibliography from the original experiment repository, run `python paper_draft/scripts/build_assets.py` from the repository root with matplotlib installed. This optional step requires the parent repository's saved results and paper catalog; ordinary compilation does not.

The supplied report is the empirical source. Bibliographic metadata come from its saved paper catalog. Dataset/model/documentation entries use the 2026 study snapshot year and identify that convention explicitly. No new experiments were performed. `REVIEW.md` records the final checks; `OUTLINE.md` maps claims to evidence.
