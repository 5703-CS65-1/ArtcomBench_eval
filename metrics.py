"""
Metric aggregation for ArtcomBench Faithfulness Evaluation.

Precision-side (per candidate claim):
    Faithful-Precision (soft)  = mean(support_score)
    Evidence Grounding Score   = mean(support_score * evidence_score)
    LF-Perception              = mean(support_score) over perception claims
    LF-Cognition               = mean(support_score) over cognition claims
    LF-Emotion                 = mean(support_score) over emotion claims

Recall-side (per gold claim):
    Claim-Recall               = mean(recall_score)

All dataset-level metrics use micro-average (pool all claims, then average).
"""

from __future__ import annotations

from typing import Any

from schemas import (
    CorpusMetrics,
    JudgedCandidateClaim,
    JudgingResult,
    RecallJudgment,
    SampleMetrics,
)


def aggregate_sample_metrics(
    sample_id: str,
    judging: JudgingResult,
    num_gold_claims: int,
) -> SampleMetrics:
    pj = judging.precision_judgments
    rj = judging.recall_judgments

    m = len(pj)
    if m == 0:
        return SampleMetrics(
            sample_id=sample_id,
            num_candidate_claims=0,
            num_gold_claims=num_gold_claims,
            faithful_precision_soft=0.0,
            evidence_grounding_score=0.0,
            lf_perception=None,
            lf_cognition=None,
            lf_emotion=None,
            claim_recall=_mean_recall(rj),
        )

    fp_soft = sum(c.support_score for c in pj) / m
    egs = sum(c.support_score * c.evidence_score for c in pj) / m

    lf_p = _level_fp(pj, "perception")
    lf_c = _level_fp(pj, "cognition")
    lf_e = _level_fp(pj, "emotion")

    cr = _mean_recall(rj)

    return SampleMetrics(
        sample_id=sample_id,
        num_candidate_claims=m,
        num_gold_claims=num_gold_claims,
        faithful_precision_soft=fp_soft,
        evidence_grounding_score=egs,
        lf_perception=lf_p,
        lf_cognition=lf_c,
        lf_emotion=lf_e,
        claim_recall=cr,
    )


def _level_fp(
    claims: list[JudgedCandidateClaim], level: str
) -> float | None:
    subset = [c for c in claims if c.assigned_level == level]
    if not subset:
        return None
    return sum(c.support_score for c in subset) / len(subset)


def _mean_recall(rj: list[RecallJudgment]) -> float:
    if not rj:
        return 0.0
    return sum(r.recall_score for r in rj) / len(rj)


# ═══════════════════════════════════════════════════════════════════════════
# Corpus-level aggregation (micro-average)
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_corpus_metrics(
    sample_metrics_list: list[SampleMetrics],
    raw_results: list[dict[str, Any]],
) -> CorpusMetrics:
    """Micro-average across ALL claims from all samples."""

    all_pj: list[JudgedCandidateClaim] = []
    all_rj: list[RecallJudgment] = []

    for r in raw_results:
        jr = r.get("judging_result")
        if jr is None:
            continue
        result = JudgingResult.model_validate(jr)
        all_pj.extend(result.precision_judgments)
        all_rj.extend(result.recall_judgments)

    n_cand = len(all_pj)
    n_gold = len(all_rj)
    n_samples = len(sample_metrics_list)

    if n_cand == 0:
        fp_soft = egs = lf_p = lf_c = lf_e = 0.0
    else:
        fp_soft = sum(c.support_score for c in all_pj) / n_cand
        egs = sum(c.support_score * c.evidence_score for c in all_pj) / n_cand

        p_claims = [c for c in all_pj if c.assigned_level == "perception"]
        c_claims = [c for c in all_pj if c.assigned_level == "cognition"]
        e_claims = [c for c in all_pj if c.assigned_level == "emotion"]

        lf_p = (sum(c.support_score for c in p_claims) / len(p_claims)) if p_claims else 0.0
        lf_c = (sum(c.support_score for c in c_claims) / len(c_claims)) if c_claims else 0.0
        lf_e = (sum(c.support_score for c in e_claims) / len(e_claims)) if e_claims else 0.0

    cr = (sum(r.recall_score for r in all_rj) / n_gold) if n_gold else 0.0

    return CorpusMetrics(
        num_samples=n_samples,
        num_total_candidate_claims=n_cand,
        num_total_gold_claims=n_gold,
        faithful_precision_soft=fp_soft,
        evidence_grounding_score=egs,
        lf_perception=lf_p,
        lf_cognition=lf_c,
        lf_emotion=lf_e,
        claim_recall=cr,
    )
