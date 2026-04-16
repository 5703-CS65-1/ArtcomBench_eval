# ArtcomBench Faithfulness Evaluation

Evaluate how faithful a multimodal model's aesthetic commentary is, given a
calibrated gold reference of observations, claims, and final outputs.

## Metrics (7)

| Metric | Type | Description |
|--------|------|-------------|
| Faithful-Precision (soft) | Precision | Average support score across candidate claims |
| Evidence Grounding Score | Precision | Average (support × evidence) across candidate claims |
| LF-Perception | Precision | Support score for perception-level claims |
| LF-Cognition | Precision | Support score for cognition-level claims |
| LF-Emotion | Precision | Support score for emotion-level claims |
| Claim-Recall | Recall | Coverage of gold claims by the candidate output |

## Pipeline stages

```
Stage A  →  Candidate model generates commentary for each image
Stage B  →  LLM judge extracts atomic claims (no gold given)
Stage C  →  LLM judge scores precision + recall (gold + image given)
Stage D  →  Python aggregates sample & corpus metrics
```

## Quick start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."

# Full run (500 samples)
python eval_pipeline.py

# Partial run (first 10 samples)
python eval_pipeline.py --limit 10

# Custom paths
python eval_pipeline.py --data 1.jsonl --output results/
```

## Configuration

Edit `config.py` to change:

- `CANDIDATE_MODEL` / `JUDGE_MODEL` — which models to use
- `MAX_CONCURRENT_REQUESTS` — API concurrency
- `IMAGE_DIR` — path to images

## Output files

All outputs go to `outputs/` (configurable):

| File | Content |
|------|---------|
| `candidate_texts.jsonl` | Raw model outputs |
| `candidate_claims.jsonl` | Extracted atomic claims |
| `judged_results.jsonl` | Per-claim scores + per-gold-claim recall |
| `sample_metrics.jsonl` | Per-sample 7-metric summary |
| `corpus_metrics.json` | Dataset-level aggregated metrics |
