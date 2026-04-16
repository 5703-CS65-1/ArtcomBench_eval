"""Evaluation pipeline configuration."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_JSONL = Path("1.jsonl")
IMAGE_DIR = Path("package_1_images")
OUTPUT_DIR = Path("outputs")

# ── Model settings ─────────────────────────────────────────────────────────
CANDIDATE_MODEL = "gpt-4o"
JUDGE_MODEL = "gpt-4o"
JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 4096
MAX_JSON_RETRIES = 2

# ── Thinking mode（仅对支持深度思考的模型生效，如 qwen3.6-plus）─────────────
ENABLE_THINKING = False

# ── Judge image toggle ────────────────────────────────────────────────────
JUDGE_WITH_IMAGE = True

# ── Concurrency ────────────────────────────────────────────────────────────
MAX_CONCURRENT_REQUESTS = 8

# ── Fixed user prompt (from gold data spec) ────────────────────────────────
CANDIDATE_PROMPT: str = (_PROMPTS_DIR / "stage_a_candidate.md").read_text(encoding="utf-8")
