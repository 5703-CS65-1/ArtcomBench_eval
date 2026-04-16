"""
ArtcomBench Faithfulness Evaluation Pipeline
=============================================
Stage A: Run candidate model on image + fixed prompt  →  candidate_text
Stage B: Extract atomic claims from candidate_text    →  candidate_claims
Stage C: Bidirectional claim-level judging             →  precision + recall
Stage D: Aggregate metrics                             →  sample & corpus metrics
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

import config
from metrics import aggregate_corpus_metrics, aggregate_sample_metrics
from prompts import (
    JUDGING_SYSTEM,
    JUDGING_USER,
)
from schemas import (
    CandidateClaim,
    CandidateOutput,
    CorpusMetrics,
    GoldSample,
    JudgingResult,
    SampleMetrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

client: AsyncOpenAI | None = None
semaphore: asyncio.Semaphore | None = None


def _ensure_client() -> AsyncOpenAI:
    global client
    if client is None:
        client = AsyncOpenAI()
    return client


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_samples(path: Path) -> list[GoldSample]:
    samples: list[GoldSample] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(GoldSample.model_validate_json(line))
            except Exception as exc:
                log.warning("Skipping line %d: %s", lineno, exc)
    log.info("Loaded %d samples from %s", len(samples), path)
    return samples


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode()


def image_media_type(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower()
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        ext, "image/png"
    )


def _strip_json_fences(text: str) -> str:
    """Remove optional ```json ... ``` fences the model sometimes emits."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        text = text[first_nl + 1 :]
    if text.endswith("```"):
        text = text[: -3]
    return text.strip()


async def _call_llm(
    messages: list[dict],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call the OpenAI chat endpoint with semaphore throttling."""
    assert semaphore is not None
    c = _ensure_client()
    extra: dict = {}
    if config.ENABLE_THINKING:
        extra["extra_body"] = {"enable_thinking": True}
    async with semaphore:
        resp = await c.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
    return resp.choices[0].message.content or ""


async def _call_llm_json(
    messages: list[dict],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int = config.MAX_JSON_RETRIES,
) -> dict[str, Any]:
    """Call LLM and parse response as JSON, with automatic retries."""
    for attempt in range(1 + retries):
        raw = await _call_llm(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        try:
            return json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError as exc:
            if attempt < retries:
                log.warning("JSON parse failed (attempt %d), retrying: %s", attempt + 1, exc)
            else:
                log.error("JSON parse failed after %d attempts. Raw output:\n%s", retries + 1, raw[:500])
                raise


# ═══════════════════════════════════════════════════════════════════════════
# Stage A: Candidate generation
# ═══════════════════════════════════════════════════════════════════════════

async def run_candidate_model(sample: GoldSample) -> CandidateOutput:
    image_path = config.IMAGE_DIR / sample.image
    b64 = encode_image_base64(image_path)
    mt = image_media_type(sample.image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mt};base64,{b64}"},
                },
                {"type": "text", "text": config.CANDIDATE_PROMPT},
            ],
        }
    ]

    text = await _call_llm(
        messages,
        model=config.CANDIDATE_MODEL,
        temperature=0,
        max_tokens=config.JUDGE_MAX_TOKENS,
    )
    return CandidateOutput(sample_id=sample.id, candidate_text=text)


# ═══════════════════════════════════════════════════════════════════════════
# Stage B: Rule-based claim extraction by 10 aesthetic dimensions
# ═══════════════════════════════════════════════════════════════════════════

DIMENSIONS = [
    "Layout and Composition",
    "Space and Perspective",
    "Light and Shadow",
    "Color",
    "Details and Texture",
    "Theme and Logic",
    "Mood",
    "The Overall",
    "Creativity",
    "Sense of Order",
]

_DIMENSION_PATTERN = "|".join(re.escape(d) for d in DIMENSIONS)
_HEADER_RE = re.compile(
    r"^[\s*#]*\d{0,2}\.?\s*(" + _DIMENSION_PATTERN + r")[\s*:]*$",
    re.MULTILINE | re.IGNORECASE,
)


def extract_claims(candidate_text: str) -> list[CandidateClaim]:
    """Split candidate text into claims by the 10 aesthetic dimension headings."""
    matches = list(_HEADER_RE.finditer(candidate_text))

    if not matches:
        log.warning("No dimension headings found — returning entire text as one claim")
        text = candidate_text.strip()
        if text:
            return [CandidateClaim(cand_claim_id="cand_01", text=text)]
        return []

    claims: list[CandidateClaim] = []
    for i, m in enumerate(matches):
        dim_name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(candidate_text)
        content = candidate_text[start:end].strip()
        if content:
            claims.append(
                CandidateClaim(
                    cand_claim_id=f"cand_{i + 1:02d}",
                    text=f"[{dim_name}] {content}",
                )
            )

    if not claims:
        log.warning("Dimension headings found but all sections empty — fallback")
        text = candidate_text.strip()
        if text:
            return [CandidateClaim(cand_claim_id="cand_01", text=text)]

    return claims


# ═══════════════════════════════════════════════════════════════════════════
# Stage C: Bidirectional claim-level judging
# ═══════════════════════════════════════════════════════════════════════════

async def judge_claims(
    sample: GoldSample,
    candidate_claims: list[CandidateClaim],
    candidate_text: str,
) -> JudgingResult:
    obs_json = json.dumps(
        [o.model_dump() for o in sample.observations], ensure_ascii=False
    )
    claims_json = json.dumps(
        [c.model_dump() for c in sample.claims], ensure_ascii=False
    )
    final_json = json.dumps(
        sample.final_outputs.model_dump(), ensure_ascii=False
    )
    hn_json = json.dumps(
        [h.model_dump() for h in sample.hard_negatives], ensure_ascii=False
    )
    cand_json = json.dumps(
        [c.model_dump() for c in candidate_claims], ensure_ascii=False
    )

    user_text = JUDGING_USER.format(
        observations_json=obs_json,
        claims_json=claims_json,
        final_outputs_json=final_json,
        hard_negatives_json=hn_json,
        candidate_claims_json=cand_json,
        candidate_text=candidate_text,
    )

    if config.JUDGE_WITH_IMAGE:
        image_path = config.IMAGE_DIR / sample.image
        b64 = encode_image_base64(image_path)
        mt = image_media_type(sample.image)
        user_content: str | list = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mt};base64,{b64}"},
            },
            {"type": "text", "text": user_text},
        ]
    else:
        user_text = user_text.replace(
            "You are given a painting image together with calibrated gold reference data",
            "You are given calibrated gold reference data",
            1,
        )
        user_content = user_text

    messages = [
        {"role": "system", "content": JUDGING_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    data = await _call_llm_json(
        messages,
        model=config.JUDGE_MODEL,
        temperature=config.JUDGE_TEMPERATURE,
        max_tokens=config.JUDGE_MAX_TOKENS,
    )
    return JudgingResult.model_validate(data)


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample orchestration
# ═══════════════════════════════════════════════════════════════════════════

async def process_sample(
    sample: GoldSample,
    *,
    skip_candidate: bool = False,
    existing_candidate: CandidateOutput | None = None,
) -> dict[str, Any]:
    """Run the full pipeline for one sample. Returns all intermediate artifacts."""
    sid = sample.id

    # Stage A
    if existing_candidate is not None:
        cand = existing_candidate
    elif skip_candidate:
        raise ValueError(f"No existing candidate for {sid} and skip_candidate=True")
    else:
        log.info("[A] Generating candidate for %s", sid)
        cand = await run_candidate_model(sample)

    # Stage B
    log.info("[B] Extracting claims for %s", sid)
    candidate_claims = extract_claims(cand.candidate_text)
    if not candidate_claims:
        log.warning("No claims extracted for %s — skipping judging", sid)
        return {
            "sample_id": sid,
            "candidate_output": cand.model_dump(),
            "candidate_claims": [],
            "judging_result": None,
            "sample_metrics": None,
            "error": "no_claims_extracted",
        }

    # Stage C
    log.info("[C] Judging %d claims for %s", len(candidate_claims), sid)
    judging = await judge_claims(sample, candidate_claims, cand.candidate_text)

    # Stage D (per-sample)
    metrics = aggregate_sample_metrics(sid, judging, len(sample.claims))

    return {
        "sample_id": sid,
        "candidate_output": cand.model_dump(),
        "candidate_claims": [c.model_dump() for c in candidate_claims],
        "judging_result": judging.model_dump(),
        "sample_metrics": metrics.model_dump(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Full run
# ═══════════════════════════════════════════════════════════════════════════

async def run_all(
    data_path: Path = config.DATA_JSONL,
    output_dir: Path = config.OUTPUT_DIR,
    limit: int | None = None,
) -> CorpusMetrics:
    global semaphore
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(data_path)
    if limit:
        samples = samples[:limit]

    t0 = time.time()
    log.info("Starting evaluation on %d samples …", len(samples))

    results: list[dict[str, Any]] = []
    all_sample_metrics: list[SampleMetrics] = []

    # Stream results to disk so partial progress is preserved on crash
    cand_f = open(output_dir / "candidate_texts.jsonl", "w", encoding="utf-8")
    claims_f = open(output_dir / "candidate_claims.jsonl", "w", encoding="utf-8")
    judged_f = open(output_dir / "judged_results.jsonl", "w", encoding="utf-8")
    sample_m_f = open(output_dir / "sample_metrics.jsonl", "w", encoding="utf-8")

    async def _wrapped(s: GoldSample) -> dict[str, Any] | None:
        sid = s.id
        try:
            # Stage A — 完成即落盘，防止后续阶段失败时丢失
            log.info("[A] Generating candidate for %s", sid)
            cand = await run_candidate_model(s)
            cand_f.write(json.dumps(cand.model_dump(), ensure_ascii=False) + "\n")
            cand_f.flush()

            # Stage B — 完成即落盘
            log.info("[B] Extracting claims for %s", sid)
            candidate_claims = extract_claims(cand.candidate_text)
            claims_f.write(
                json.dumps(
                    {"sample_id": sid, "candidate_claims": [c.model_dump() for c in candidate_claims]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            claims_f.flush()

            if not candidate_claims:
                log.warning("No claims extracted for %s — skipping judging", sid)
                return None

            # Stage C
            log.info("[C] Judging %d claims for %s", len(candidate_claims), sid)
            judging = await judge_claims(s, candidate_claims, cand.candidate_text)

            judged_f.write(
                json.dumps({"sample_id": sid, **judging.model_dump()}, ensure_ascii=False) + "\n"
            )
            judged_f.flush()

            # Stage D (per-sample)
            metrics = aggregate_sample_metrics(sid, judging, len(s.claims))
            sample_m_f.write(json.dumps(metrics.model_dump(), ensure_ascii=False) + "\n")
            sample_m_f.flush()

            return {
                "sample_id": sid,
                "candidate_output": cand.model_dump(),
                "candidate_claims": [c.model_dump() for c in candidate_claims],
                "judging_result": judging.model_dump(),
                "sample_metrics": metrics.model_dump(),
            }
        except Exception:
            log.exception("Failed processing sample %s", sid)
            return None

    tasks = [_wrapped(s) for s in samples]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is None:
            continue
        results.append(result)
        sm = SampleMetrics.model_validate(result["sample_metrics"])
        all_sample_metrics.append(sm)

    for fh in (cand_f, claims_f, judged_f, sample_m_f):
        fh.close()

    # Corpus-level aggregation
    corpus = aggregate_corpus_metrics(all_sample_metrics, results)
    (output_dir / "corpus_metrics.json").write_text(
        json.dumps(corpus.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    elapsed = time.time() - t0
    log.info(
        "Evaluation complete: %d samples in %.1fs\n"
        "  FP_soft=%.4f  EGS=%.4f  LF-P=%.4f  LF-C=%.4f  LF-E=%.4f  CR=%.4f",
        len(all_sample_metrics),
        elapsed,
        corpus.faithful_precision_soft,
        corpus.evidence_grounding_score,
        corpus.lf_perception,
        corpus.lf_cognition,
        corpus.lf_emotion,
        corpus.claim_recall,
    )
    return corpus


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ArtcomBench Faithfulness Eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=config.DATA_JSONL,
                        help="Path to gold JSONL file")
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Path to image directory (overrides config)")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N samples")
    parser.add_argument("--candidate-model", type=str, default=None,
                        help="Model for candidate generation (overrides config)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model for judge LLM (overrides config)")
    parser.add_argument("--judge-max-tokens", type=int, default=None,
                        help="Max tokens for judge responses")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Max concurrent API requests")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                        help="Enable thinking mode (e.g. for qwen3.6-plus)")
    parser.add_argument("--no-judge-image", action="store_true", default=False,
                        help="Do not send image to judge model (text-only judging based on gold reference)")
    args = parser.parse_args()

    if args.candidate_model:
        config.CANDIDATE_MODEL = args.candidate_model
    if args.judge_model:
        config.JUDGE_MODEL = args.judge_model
    if args.judge_max_tokens:
        config.JUDGE_MAX_TOKENS = args.judge_max_tokens
    if args.concurrency:
        config.MAX_CONCURRENT_REQUESTS = args.concurrency
    if args.image_dir:
        config.IMAGE_DIR = args.image_dir
    if args.enable_thinking:
        config.ENABLE_THINKING = True
    if args.no_judge_image:
        config.JUDGE_WITH_IMAGE = False

    global client
    client_kwargs: dict[str, str] = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = AsyncOpenAI(**client_kwargs) if client_kwargs else AsyncOpenAI()

    asyncio.run(run_all(data_path=args.data, output_dir=args.output, limit=args.limit))


if __name__ == "__main__":
    main()
