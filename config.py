"""Evaluation pipeline configuration."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_JSONL = Path("1.jsonl")
IMAGE_DIR = Path("package_1_images")
OUTPUT_DIR = Path("outputs")

# ── Candidate model (Stage A) ──────────────────────────────────────────────
CANDIDATE_MODEL = "gpt-4o"
CANDIDATE_API_KEY: str | None = None   # None → fall back to OPENAI_API_KEY env var
CANDIDATE_BASE_URL: str | None = None  # None → OpenAI official endpoint

# ── Judge model (Stage C) ──────────────────────────────────────────────────
JUDGE_MODEL = "gpt-4o"
JUDGE_API_KEY: str | None = None       # None → fall back to OPENAI_API_KEY env var
JUDGE_BASE_URL: str | None = None      # None → OpenAI official endpoint
JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 4096
MAX_JSON_RETRIES = 2

# ── Thinking mode
CANDIDATE_ENABLE_THINKING = False  # candidate 模型（Stage A）思考模式开关
ENABLE_THINKING = False            # judge 模型（Stage C）思考模式开关

# ── Judge image toggle ────────────────────────────────────────────────────
JUDGE_WITH_IMAGE = True

# ── Concurrency ────────────────────────────────────────────────────────────
MAX_CONCURRENT_REQUESTS = 8

# ── Fixed user prompt (from gold data spec) ────────────────────────────────
CANDIDATE_PROMPT: str = (_PROMPTS_DIR / "stage_a_candidate.md").read_text(encoding="utf-8")
