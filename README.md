# Replication package

When Tone Is Not Outlook: Dictionary Sentiment Drift in Chinese Economists' Commentary.

## Contents

| Folder | File | Description |
| --- | --- | --- |
| `code/` | `step1_measures.py` | Cleans the corpus, codes affiliations and builds the dictionary text measures (requires the original corpus file, see below). |
| `code/` | `step5_llm.py` | Draws the LLM sample, scores articles through the API (resumable), computes inter-model agreement, human-validation statistics, DSL-corrected estimates and the LLM difference-in-differences. |
| `code/` | `step6_revision.py` | Composition checks, relative-magnitude bounds, randomization inference (1,000 permutations), placebo distribution, leave-one-out estimates and term-level dictionary checks. |
| `data/` | `measures.csv` | Article-level dictionary measures for all 20,058 cleaned articles (no article text). |
| `data/` | `llm_scores.jsonl` | Codes from qwen-plus-2025-12-01 for 3,690 articles, with the verbatim model output, model name and call date. |
| `data/` | `llm_scores_qwen3-235b.jsonl` | Codes from qwen3-235b-a22b-instruct-2507 for the 500-article agreement subsample. |
| `human_validation/` | `human_coding_coder1.xlsx`, `human_coding_coder2.xlsx` | Independent codes of the two human coders for the 200 validation articles. |
| `human_validation/` | `human_key.csv` | Maps validation IDs to article IDs and treatment cells. |
| `results/` | `*.json` | Estimates reported in the paper (LLM DiD, agreement, validation, DSL, robustness, revision analyses). |

## Data not included

The full texts of the commentaries are copyrighted by their authors and publishers and are not redistributed. They were collected from the public website of the China Chief Economist Forum (CCEF). `step1_measures.py` rebuilds every text measure from the original corpus file; all later steps run from the files in this package.

## Environment

Python 3.12 with pandas 2.3.3, jieba 0.42.1, pyfixest 0.60.0, statsmodels and scipy. The sentiment dictionaries are read from the `cntext` package. API keys are read from environment variables and are not included.
