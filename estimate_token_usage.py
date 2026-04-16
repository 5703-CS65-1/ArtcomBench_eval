from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean

import tiktoken
from PIL import Image

from prompts import (
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_EXTRACTION_USER,
    JUDGING_SYSTEM,
    JUDGING_USER,
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


class ApproxEncoding:
    name = "approx_en_v1"

    @staticmethod
    def encode(text: str) -> list[int]:
        latin_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text)
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        char_est = len(text) / 4.0
        word_est = len(latin_words) * 1.3 + len(cjk_chars)
        approx = max(1, round((char_est + word_est) / 2.0))
        return [0] * approx


def estimate_qwen32_image_tokens(image_path: Path) -> dict[str, int]:
    with Image.open(image_path) as img:
        width, height = img.size
    pixels = width * height
    # Per DashScope docs for qwen3.5/qwen3-vl families: ~1 image token per 32x32 pixels.
    image_tokens = math.ceil(pixels / 1024)
    return {
        "width": width,
        "height": height,
        "pixels": pixels,
        "image_tokens": image_tokens,
    }


def compact_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False)


def format_int(n: int) -> str:
    return f"{n:,}"


def summarize(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "avg": 0.0, "max": 0}
    return {"min": min(values), "avg": mean(values), "max": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate token usage for ArtcomBench workflow.")
    parser.add_argument("--data", type=Path, default=Path("1.jsonl"))
    parser.add_argument("--image-dir", type=Path, default=Path("package_1_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--encoding", type=str, default="o200k_base")
    args = parser.parse_args()

    try:
        enc: tiktoken.Encoding | ApproxEncoding = tiktoken.get_encoding(args.encoding)
    except Exception:
        enc = ApproxEncoding()

    samples = load_jsonl(args.data)
    candidates = {row["sample_id"]: row for row in load_jsonl(args.output_dir / "candidate_texts.jsonl")}
    claims_rows = {row["sample_id"]: row for row in load_jsonl(args.output_dir / "candidate_claims.jsonl")}
    judged_rows = {row["sample_id"]: row for row in load_jsonl(args.output_dir / "judged_results.jsonl")}

    stage_a_prompt_tokens = count_tokens(Path("prompts/stage_a_candidate.md").read_text(encoding="utf-8"), enc)
    stage_b_system_tokens = count_tokens(CLAIM_EXTRACTION_SYSTEM, enc)
    stage_c_system_tokens = count_tokens(JUDGING_SYSTEM, enc)

    exact_rows: list[dict[str, int]] = []
    image_tokens_all: list[int] = []
    stage_c_gold_only_input_tokens_all: list[int] = []
    stage_a_input_tokens_all: list[int] = []
    stage_c_gold_only_by_sid: dict[str, int] = {}

    for sample in samples:
        sid = sample["id"]
        image_info = estimate_qwen32_image_tokens(args.image_dir / sample["image"])
        image_tokens_all.append(image_info["image_tokens"])

        stage_a_input_tokens_all.append(stage_a_prompt_tokens + image_info["image_tokens"])

        obs_json = compact_json(sample["observations"])
        claims_json = compact_json(sample["claims"])
        final_json = compact_json(sample["final_outputs"])
        hn_json = compact_json(sample["hard_negatives"])

        gold_only_stage_c_user = JUDGING_USER.format(
            observations_json=obs_json,
            claims_json=claims_json,
            final_outputs_json=final_json,
            hard_negatives_json=hn_json,
            candidate_claims_json="[]",
            candidate_text="",
        )
        stage_c_gold_only_input_tokens_all.append(
            stage_c_system_tokens + count_tokens(gold_only_stage_c_user, enc) + image_info["image_tokens"]
        )
        stage_c_gold_only_by_sid[sid] = stage_c_gold_only_input_tokens_all[-1]

        if sid not in candidates or sid not in claims_rows or sid not in judged_rows:
            continue

        candidate_text = candidates[sid]["candidate_text"]
        candidate_claims = claims_rows[sid]["candidate_claims"]
        judging = judged_rows[sid]

        stage_a_output_tokens = count_tokens(candidate_text, enc)

        stage_b_user = CLAIM_EXTRACTION_USER.format(candidate_text=candidate_text)
        stage_b_input_tokens = stage_b_system_tokens + count_tokens(stage_b_user, enc)
        stage_b_output_text = compact_json({"candidate_claims": candidate_claims})
        stage_b_output_tokens = count_tokens(stage_b_output_text, enc)

        user_text = JUDGING_USER.format(
            observations_json=obs_json,
            claims_json=claims_json,
            final_outputs_json=final_json,
            hard_negatives_json=hn_json,
            candidate_claims_json=compact_json(candidate_claims),
            candidate_text=candidate_text,
        )
        stage_c_input_tokens = stage_c_system_tokens + count_tokens(user_text, enc) + image_info["image_tokens"]
        stage_c_output_text = compact_json(
            {
                "precision_judgments": judging["precision_judgments"],
                "recall_judgments": judging["recall_judgments"],
            }
        )
        stage_c_output_tokens = count_tokens(stage_c_output_text, enc)

        exact_rows.append(
            {
                "sample_id": sid,
                "stage_a_input": stage_a_prompt_tokens + image_info["image_tokens"],
                "stage_a_output": stage_a_output_tokens,
                "stage_b_input": stage_b_input_tokens,
                "stage_b_output": stage_b_output_tokens,
                "stage_c_input": stage_c_input_tokens,
                "stage_c_output": stage_c_output_tokens,
                "image_tokens": image_info["image_tokens"],
                "stage_c_gold_only_input": stage_c_gold_only_by_sid[sid],
            }
        )

    print(f"encoding={getattr(enc, 'name', args.encoding)}")
    print(f"dataset_samples={len(samples)}")
    print(f"available_exact_samples={len(exact_rows)}")
    print()

    print("[fixed / dataset-known]")
    print(f"stage_a_prompt_tokens={stage_a_prompt_tokens}")
    print(f"stage_b_system_tokens={stage_b_system_tokens}")
    print(f"stage_c_system_tokens={stage_c_system_tokens}")
    image_summary = summarize(image_tokens_all)
    print(
        "image_tokens_per_sample_qwen32="
        f"min={image_summary['min']}, avg={image_summary['avg']:.2f}, max={image_summary['max']}"
    )
    stage_a_input_summary = summarize(stage_a_input_tokens_all)
    print(
        "stage_a_input_tokens_per_sample="
        f"min={stage_a_input_summary['min']}, avg={stage_a_input_summary['avg']:.2f}, max={stage_a_input_summary['max']}"
    )
    gold_stage_c_summary = summarize(stage_c_gold_only_input_tokens_all)
    print(
        "stage_c_gold_only_input_tokens_per_sample="
        f"min={gold_stage_c_summary['min']}, avg={gold_stage_c_summary['avg']:.2f}, max={gold_stage_c_summary['max']}"
    )
    print()

    if exact_rows:
        print("[exact from current outputs]")
        for row in exact_rows:
            total_in = row["stage_a_input"] + row["stage_b_input"] + row["stage_c_input"]
            total_out = row["stage_a_output"] + row["stage_b_output"] + row["stage_c_output"]
            total_all = total_in + total_out
            print(f"sample_id={row['sample_id']}")
            print(
                "  "
                f"stage_a: in={format_int(row['stage_a_input'])}, out={format_int(row['stage_a_output'])}"
            )
            print(
                "  "
                f"stage_b: in={format_int(row['stage_b_input'])}, out={format_int(row['stage_b_output'])}"
            )
            print(
                "  "
                f"stage_c: in={format_int(row['stage_c_input'])}, out={format_int(row['stage_c_output'])}"
            )
            print(f"  total_input={format_int(total_in)}")
            print(f"  total_output={format_int(total_out)}")
            print(f"  total_all={format_int(total_all)}")

        avg_in = mean(r["stage_a_input"] + r["stage_b_input"] + r["stage_c_input"] for r in exact_rows)
        avg_out = mean(r["stage_a_output"] + r["stage_b_output"] + r["stage_c_output"] for r in exact_rows)
        avg_stage_a_out = mean(r["stage_a_output"] for r in exact_rows)
        avg_stage_b_in = mean(r["stage_b_input"] for r in exact_rows)
        avg_stage_b_out = mean(r["stage_b_output"] for r in exact_rows)
        avg_stage_c_extra_in = mean(r["stage_c_input"] - r["stage_c_gold_only_input"] for r in exact_rows)
        avg_stage_c_out = mean(r["stage_c_output"] for r in exact_rows)
        projected_input_better = (
            sum(stage_a_input_tokens_all)
            + round(avg_stage_b_in * len(samples))
            + sum(stage_c_gold_only_input_tokens_all)
            + round(avg_stage_c_extra_in * len(samples))
        )
        projected_output_better = round(
            (avg_stage_a_out + avg_stage_b_out + avg_stage_c_out) * len(samples)
        )
        print()
        print("[projection if remaining samples look like observed outputs]")
        print(f"projected_total_input_tokens={format_int(round(avg_in * len(samples)))}")
        print(f"projected_total_output_tokens={format_int(round(avg_out * len(samples)))}")
        print(f"projected_total_tokens={format_int(round((avg_in + avg_out) * len(samples)))}")
        print()
        print("[projection with dataset-known image/gold inputs]")
        print(f"projected_total_input_tokens={format_int(projected_input_better)}")
        print(f"projected_total_output_tokens={format_int(projected_output_better)}")
        print(f"projected_total_tokens={format_int(projected_input_better + projected_output_better)}")


if __name__ == "__main__":
    main()
