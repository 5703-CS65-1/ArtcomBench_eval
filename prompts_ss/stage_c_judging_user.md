You are given a painting image together with calibrated gold reference data and a set of candidate claims extracted from a model's commentary.

## Gold reference (all equally authoritative)

observations, claims, and final_outputs below have been human-calibrated and are ALL fully trustworthy. They jointly serve as the unified gold reference for EVERY scoring dimension (support, evidence grounding, and recall).

### Gold observations
{observations_json}

### Gold claims
{claims_json}

### Gold final_outputs
{final_outputs_json}

### Hard negatives (examples of unsupported / unverifiable content)
{hard_negatives_json}

## Candidate data

### Candidate claims (extracted atomic propositions)
{candidate_claims_json}

### Candidate raw text (for recall evaluation)
{candidate_text}

───────────────────────────────────────────────────
## Task A — Precision scoring

For EACH candidate claim, evaluate:

1. **assigned_level**: perception / cognition / emotion
2. **matched_gold_claim_ids**: which gold claim(s) align with this candidate claim (empty list if none)
3. **matched_obs_ids**: which observation(s) relate (empty list if none)
4. **support_score** (1.0 / 0.5 / 0.0):
   - 1.0 = clearly supported by the gold reference
   - 0.5 = partially or weakly supported
   - 0.0 = unsupported, speculative, or hallucinated
   - IMPORTANT — Cap rule: if the closest matching gold claim has support_label = "weakly_supported", the candidate's support_score MUST NOT exceed 0.5.
5. **evidence_score** (1.0 / 0.75 / 0.5 / 0.25 / 0.0):
   - Measures how strongly the claim is grounded in the image and gold reference.
   - 1.0 = directly, clearly anchored
   - 0.75 = well grounded, requires synthesizing multiple references
   - 0.5 = moderate grounding with some inferential leap
   - 0.25 = weak grounding
   - 0.0 = no grounding
6. **reason**: one-sentence justification.

Observations CAN rescue a faithful paraphrase that does not match any gold claim verbatim. A candidate claim grounded in the gold reference should still receive credit even without exact wording overlap.

───────────────────────────────────────────────────
## Task B — Recall scoring

For EACH gold claim, judge whether the candidate's RAW TEXT (not just the extracted claims) covers it:

1. **gold_claim_id**: the id of the gold claim
2. **recall_score** (1.0 / 0.5 / 0.0):
   - 1.0 = the candidate text clearly covers this gold claim's core judgment
   - 0.5 = partially or implicitly touched
   - 0.0 = not covered at all
3. **matched_cand_claim_ids**: which candidate claim(s) contribute to the coverage (empty list if none)
4. **reason**: one-sentence justification.

───────────────────────────────────────────────────
## Output format

Return valid JSON only, with no markdown fencing. Use this exact schema:

{{
  "precision_judgments": [
    {{
      "cand_claim_id": "cand_01",
      "text": "...",
      "assigned_level": "perception",
      "matched_gold_claim_ids": ["claim_02"],
      "matched_obs_ids": ["obs_03"],
      "support_score": 1.0,
      "evidence_score": 1.0,
      "reason": "..."
    }}
  ],
  "recall_judgments": [
    {{
      "gold_claim_id": "claim_01",
      "recall_score": 1.0,
      "matched_cand_claim_ids": ["cand_02"],
      "reason": "..."
    }}
  ]
}}