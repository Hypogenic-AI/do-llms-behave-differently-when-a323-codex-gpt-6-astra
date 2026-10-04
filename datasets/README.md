# Dataset resources

Three dataset collections, downloaded 2026-10-04. Raw data are local and ignored by git. Small documentation samples in `samples/` total less than100KB. Exact source URLs, revisions, sizes and SHA256 are in [download manifest](../notes/dataset_manifest.json), [verified Anthropic manifest](../notes/anthropic_verified_manifest.json); complete schema/missingness/duplicate checks are in [validation](../notes/dataset_validation.json).

## Reproduce and load

From the repository root (Python3.10+; this workspace uses a fresh `.venv`):

```bash
pwd
uv sync
uv run python scripts/reproduce_datasets.py
uv run python scripts/validate_resources.py
```

The reproducer downloads pinned source files, checks hashes before writing, and skips verified existing files. WildChat is an unpinned viewer snapshot: if its content changes the script fails explicitly rather than silently replacing it. The original collection script is `scripts/download_datasets.py`; it depends on saved HF metadata in `notes/` and may retrieve a newer viewer snapshot. `scripts/repair_anthropic.py` resolves current GitHub HEAD; use the manifest reproducer for the recorded revision.

```python
import csv, json
from pathlib import Path
benign = list(csv.DictReader(open('datasets/jbb/data/benign-behaviors.csv')))
harmful = list(csv.DictReader(open('datasets/jbb/data/harmful-behaviors.csv')))
sycophancy = [json.loads(s) for s in open(
    'datasets/anthropic_verified/sycophancy_on_philpapers2020.jsonl')]
wildchat = [json.loads(s) for s in open('datasets/wildchat/first_1000.jsonl')]
```

These are native CSV/JSONL, not `datasets.save_to_disk` folders.

## JBB-Behaviors

- Source: [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), revision `886acc352a31533ffbcf4ef22c744658688086fc`.
- Local: `jbb/data/`; full100 harmful and100 benign behaviors, plus300 judge-comparison rows (~407KB total CSV). MIT license per source card (`jbb/README.md`).
- Tasks: harmful compliance and benign over-refusal; CSV contains behavior/goal/target and category metadata. Judge-comparison file is judge calibration material, not300 new independent behaviors.
- No missing CSV fields or duplicate full rows detected. Inspect category counts and retain behavior IDs. Existing benchmark membership may itself cue evaluation.
- A target response is not a correctness score. Grade actual response content; distinguish refusal from unhelpful/non-actionable compliance. No harmful completions generated in this phase.
- Documentation examples: `samples/jbb_benign.json`.

## Anthropic model-written-evals, sycophancy subset

- Requested source: [Anthropic/model-written-evals](https://huggingface.co/datasets/Anthropic/model-written-evals), revision `d533f626cc321c92175a58ee570aa3cdb87238d1`. HF card states CC-BY-4.0. Only the three relevant sycophancy files are downloaded, not every behavioral evaluation.
- **Use `anthropic_verified/` for experiments.** The HF snapshot retrieved under the philosophy filename was byte-identical to the NLP file (same SHA256;30,168 raw rows but duplicated file content). It remains in `anthropic/sycophancy/` for provenance, and must not be pooled as independent data.
- Verified replacement: [Anthropic original repository](https://github.com/anthropics/evals/tree/84fcc677e52e1902d696c32cd1a6b663e70d3993/sycophancy), commit `84fcc677e52e1902d696c32cd1a6b663e70d3993`.
- NLP9,984; philosophy9,867; political10,200; **30,051 usable source rows**, ~24.9MB. NLP/political hashes match the HF snapshot; philosophy differs and contains philosophy questions as expected. Original source license should be retained with redistribution; HF license metadata is recorded, not a legal assessment.
- Schema: `question`, `answer_matching_behavior`, `answer_not_matching_behavior`; political data also has `user_affiliation`. No missing values, duplicate full rows or identical matching/nonmatching answers detected in verified files.
- Labels indicate matching the prompted stance, not objective factual correctness. Preserve persona and opinion content across style conditions. Strip persona only as an explicitly separate baseline. Group by underlying question/opinion and generated persona family before splitting; row count overstates independent question count.
- One example from each file is in `samples/`.

## WildChat-1M: local pilot subset

- Source: [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M). Metadata revision `7d6490e462285cf85d91eabea0f9a954fbddcd1f`; viewer download cannot pin that revision. License ODC-BY per HF card.
- Full train split837,989 conversations in14parquet shards, ~3.36GB download (~6.84GB uncompressed per card). No full download: user instructions permit subsets after checking size.
- Local `wildchat/first_1000.jsonl`:1,000 consecutive viewer rows,16,462,494 bytes. Offsets0–999 contiguous;1,000 nonempty conversations starting with user turns;995 unique conversation hashes;523 English labels. Language labels include implausible minority assignments and need manual spot checks.
- **Deterministic prefix, not a random representative sample.** Deduplicate hashes and near-duplicate user text before sampling. Long conversations need first-user-turn extraction and context checks; do not substitute assistant output for human text. User messages may contain pasted/generated text: no verified human authorship labels.
- Metadata can include location/network information and toxic content; downstream working tables should keep only prompt text, stable source hash and necessary language/task fields. Documentation publishes a schema example (`samples/wildchat_schema.json`), not raw conversations.
- To obtain a reproducible larger sample, use pinned HF parquet shards and record selected row IDs (extra installation needed):

```bash
pwd
uv add datasets
```

```python
from datasets import load_dataset
stream = load_dataset('allenai/WildChat-1M',
    revision='7d6490e462285cf85d91eabea0f9a954fbddcd1f',
    split='train', streaming=True)
# Save a specified deterministic selection to datasets/; filter English,
# deduplicate, and record inclusion criteria before treating it as a study sample.
```

## Scope limits

No dataset here provides matched human/LLM-style prompts with audited identical meaning. Construct those pairs as the first experiment step in `planning.md`. No objective accuracy dataset is downloaded; the prepared outcomes are refusal/compliance and opinion sycophancy. Downloading additional benchmarks is unnecessary unless accuracy becomes a declared outcome within the retained directions.
