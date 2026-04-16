#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
# ArtcomBench Faithfulness Evaluation — 启动脚本
# 修改下面的参数后直接运行: bash run.sh
# ══════════════════════════════════════════════════════════════════════════

# ── API 配置 ──────────────────────────────────────────────────────────────
OPENAI_API_KEY=""       # API 密钥
BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"                                                 # 留空用 OpenAI 官方，填写则用兼容接口（如 vLLM / Azure / DeepSeek）

# ── 模型配置 ──────────────────────────────────────────────────────────────
CANDIDATE_MODEL="qwen3.5-27b"            # 待测模型（Stage A 生成美学评价）
JUDGE_MODEL="qwen3.6-plus"                # 裁判模型（Stage B/C 提取+打分）
JUDGE_MAX_TOKENS=32768              # 裁判模型单次最大输出 tokens（Stage C 需输出大量 JSON，不能太小）

# ── 数据路径 ──────────────────────────────────────────────────────────────
DATA_JSONL="1.jsonl"                # gold 数据文件
IMAGE_DIR="package_1_images"        # 图片目录
OUTPUT_DIR="outputs"                # 输出目录

# ── 运行控制 ──────────────────────────────────────────────────────────────
CONCURRENCY=1                       # 最大并发 API 请求数
LIMIT="1"                            # 只跑前 N 条（留空=全量 500 条）
ENABLE_THINKING=true               # 开启深度思考模式（true/false，适用于 qwen3.6-plus 等）
JUDGE_WITH_IMAGE=true              # 是否向裁判模型提供图片（false=纯文本，仅依据 Gold reference 判定）

# ══════════════════════════════════════════════════════════════════════════
# 以下无需修改
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

export OPENAI_API_KEY

CMD=(
    python eval_pipeline.py
    --data          "$DATA_JSONL"
    --image-dir     "$IMAGE_DIR"
    --output        "$OUTPUT_DIR"
    --candidate-model "$CANDIDATE_MODEL"
    --judge-model   "$JUDGE_MODEL"
    --judge-max-tokens "$JUDGE_MAX_TOKENS"
    --concurrency   "$CONCURRENCY"
)

[[ -n "$BASE_URL" ]]               && CMD+=(--base-url "$BASE_URL")
[[ -n "$LIMIT"    ]]               && CMD+=(--limit "$LIMIT")
[[ "$ENABLE_THINKING" == "true" ]] && CMD+=(--enable-thinking)
[[ "$JUDGE_WITH_IMAGE" != "true" ]] && CMD+=(--no-judge-image)

echo "════════════════════════════════════════════════════════"
echo "  ArtcomBench Faithfulness Eval"
echo "  candidate_model : $CANDIDATE_MODEL"
echo "  judge_model     : $JUDGE_MODEL"
echo "  concurrency     : $CONCURRENCY"
echo "  limit           : ${LIMIT:-all}"
echo "  enable_thinking : $ENABLE_THINKING"
echo "  judge_with_image: $JUDGE_WITH_IMAGE"
echo "  output          : $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════"

"${CMD[@]}"
