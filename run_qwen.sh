#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
# ArtcomBench Faithfulness Evaluation — 启动脚本
# 修改下面的参数后直接运行: bash run.sh
# ══════════════════════════════════════════════════════════════════════════

# ── Candidate 模型配置（Stage A：多模态生成美学评价，必须支持视觉）─────────────
CANDIDATE_MODEL="qwen3.6-plus"
CANDIDATE_API_KEY=""
CANDIDATE_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# ── Judge 模型配置（Stage C：忠实度打分）────────────────────────────────────
JUDGE_MODEL="qwen3.6-plus"           # 纯文本裁判可换为 deepseek-chat 等
JUDGE_API_KEY=""
JUDGE_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
JUDGE_MAX_TOKENS=32768               # Stage C 输出大量 JSON，不能太小

# ── 数据路径 ──────────────────────────────────────────────────────────────
DATA_JSONL="data/eval_text/1.jsonl"                 # gold 数据文件
IMAGE_DIR="data/eval_text/package_1_images"         # 图片目录
OUTPUT_DIR="outputs/test_qwen_8"                 # 输出目录

# ── 运行控制 ──────────────────────────────────────────────────────────────
CONCURRENCY=50                        # 最大并发 API 请求数
LIMIT="50"                            # 只跑前 N 条（留空=全量）
ENABLE_THINKING=true                 # judge 模型深度思考（qwen3.6-plus 等支持；DeepSeek 请设 false）
CANDIDATE_THINKING=true             # candidate 模型（Stage A）深度思考开关
MAX_JSON_RETRIES=2                   # JSON 解析失败最大重试次数
JUDGE_WITH_IMAGE=false               # 是否向裁判模型提供图片（纯文本模型请设 false）

# ══════════════════════════════════════════════════════════════════════════
# 以下无需修改
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

CMD=(
    python eval_pipeline.py
    --data                "$DATA_JSONL"
    --image-dir           "$IMAGE_DIR"
    --output              "$OUTPUT_DIR"
    --candidate-model     "$CANDIDATE_MODEL"
    --candidate-api-key   "$CANDIDATE_API_KEY"
    --judge-model         "$JUDGE_MODEL"
    --judge-api-key       "$JUDGE_API_KEY"
    --judge-max-tokens    "$JUDGE_MAX_TOKENS"
    --concurrency         "$CONCURRENCY"
)

[[ -n "$CANDIDATE_BASE_URL" ]]           && CMD+=(--candidate-base-url "$CANDIDATE_BASE_URL")
[[ -n "$JUDGE_BASE_URL"     ]]           && CMD+=(--judge-base-url     "$JUDGE_BASE_URL")
[[ -n "$LIMIT"              ]]           && CMD+=(--limit "$LIMIT")
[[ "$ENABLE_THINKING"       == "true" ]] && CMD+=(--enable-thinking)
[[ "$CANDIDATE_THINKING"    == "true" ]] && CMD+=(--candidate-enable-thinking)
[[ -n "$MAX_JSON_RETRIES"   ]]           && CMD+=(--max-json-retries "$MAX_JSON_RETRIES")
[[ "$JUDGE_WITH_IMAGE"      != "true" ]] && CMD+=(--no-judge-image)

echo "════════════════════════════════════════════════════════"
echo "  ArtcomBench Faithfulness Eval"
echo "  candidate_model    : $CANDIDATE_MODEL"
echo "  candidate_url      : ${CANDIDATE_BASE_URL:-openai-official}"
echo "  candidate_thinking : $CANDIDATE_THINKING"
echo "  judge_model        : $JUDGE_MODEL"
echo "  judge_url          : ${JUDGE_BASE_URL:-openai-official}"
echo "  concurrency        : $CONCURRENCY"
echo "  limit              : ${LIMIT:-all}"
echo "  enable_thinking    : $ENABLE_THINKING"
echo "  max_json_retries   : $MAX_JSON_RETRIES"
echo "  judge_with_image   : $JUDGE_WITH_IMAGE"
echo "  output             : $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════"

"${CMD[@]}"
