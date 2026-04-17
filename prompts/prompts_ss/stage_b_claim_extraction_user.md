Split the following candidate commentary into minimal atomic evaluative claims.

Rules:
1. Do NOT invent content not explicitly present in the candidate commentary.
2. If one sentence contains multiple independent judgments, split them into separate claims.
3. If a single statement spans two cognitive levels (e.g. a perceptual fact AND an emotional inference), split it into two claims.
4. Keep each claim semantically minimal but still a meaningful proposition.
5. Ignore filler, hedging, meta-discourse (e.g. "I think", "overall speaking"), and pure restatements of the prompt.
6. Do NOT score or judge the claims — extraction only.
7. Return valid JSON only, with no markdown fencing.

Return schema:
{{
  "candidate_claims": [
    {{
      "cand_claim_id": "cand_01",
      "text": "..."
    }}
  ]
}}

Candidate commentary:
{candidate_text}