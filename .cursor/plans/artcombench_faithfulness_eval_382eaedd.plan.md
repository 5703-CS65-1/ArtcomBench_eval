---
name: ArtcomBench Faithfulness Eval
overview: 基于 500 条 gold 数据构建一个多模态艺术评价忠实度评测框架，采用两阶段 LLM-judge 流程，输出 7 个指标（含新增 Claim-Recall），并对 instruction.md 中 gold 数据层级使用方式进行修正。
todos:
  - id: config-schema
    content: 创建 config.py 和 schemas.py（Pydantic schema 定义 + JSON 校验）
    status: completed
  - id: prompts
    content: 编写 prompts.py：Claim Extraction prompt 和 Bidirectional Scoring prompt（含 recall 评估）
    status: completed
  - id: pipeline
    content: 实现 eval_pipeline.py：Stage A/B/C 主流程 + JSON retry + 错误处理
    status: completed
  - id: metrics
    content: 实现 metrics.py：7 个指标计算 + micro-average 聚合（recall 不分层）
    status: completed
  - id: requirements
    content: 创建 requirements.txt 和 README
    status: completed
isProject: false
---

# ArtcomBench 艺术评价忠实度评测方案

## 1. 对 instruction.md 的审视：gold 数据使用方式的修正

instruction.md 提出了一个三级信息层级：`observations > claims > final_outputs`，其中 `final_outputs` 被大幅降权为"不能单独创造证据的背景语义"，且 `observations` 仅被定位为"视觉事实锚点"，不参与评价正确性判定。这种层级划分和角色分离**过于复杂且不符合 gold 数据的实际质量**。

### 核心修正：统一参照源

**observations + claims + final_outputs 三者都经过校准，共同构成统一的、完全可信的 gold reference。** 不再区分"视觉证据来源"和"评价正确性参照"两个轴——三者在所有评估维度上具有同等权威：

- **support_score（支持度）**：judge 可以依据 observations、claims、final_outputs 中的任何信息来判断 candidate claim 是否 faithful
- **evidence_score（证据锚定）**：judge 可以依据 image + observations + claims + final_outputs 来判断 candidate claim 的 grounding 强度
- **recall_score（召回）**：judge 依据 gold claims 全集 + final_outputs + observations 上下文来判断 candidate 是否覆盖了某条 gold claim

### 与 instruction.md 的关键区别

- instruction.md：`observations` 只管"看没看到"，`final_outputs` 不能创造证据，evidence_score 仅由 image + observations 决定
- 本方案：三者无层级区分，evidence_score 同样可以参考 claims 和 final_outputs 中的分析内容作为 grounding 依据

### 保留的合理设计

instruction.md 中以下设计完全保留：

- **上限规则**：gold claim 为 `weakly_supported` 时，匹配的 candidate claim 的 support_score 上限为 0.5
- **两阶段分离**：claim extraction 阶段不给 gold，避免 judge 被 gold wording 污染
- **micro-average 聚合**：比 macro sample average 更能反映真实表现
- **hard_negatives 的使用**：作为反幻觉校准器注入 judge prompt

---

## 2. 新增指标：Claim-Recall

### 为什么需要

instruction.md 的 5 个指标全部是 precision-style（只惩罚说错，不惩罚没说到）。这导致一个极端策略可以获得高分：模型只输出 1-2 句最安全的话，precision 很高但几乎什么都没评价。

Claim-Recall 衡量的是：**gold claims 中有多少被候选模型的输出覆盖了？**

### 设计

对每条 gold claim $g_j$，基于 candidate 的完整输出文本（不是提取后的 candidate claims）判断覆盖程度：

$$r_j \in 1.0, 0.5, 0.0$$

- **1.0**：candidate 输出明确覆盖了这条 gold claim 的核心判断
- **0.5**：candidate 输出部分或隐含地触及了这条 gold claim
- **0.0**：candidate 输出完全没有覆盖

**样本级 Claim-Recall**：

$$CR = \frac{1}{|G|}\sum_{j=1}^{|G|} r_j$$

**数据集级 Claim-Recall（micro-average）**：

$$CR^{dataset} = \frac{\sum_{j \in \text{all gold claims}} r_j}{\sum_{j \in \text{all gold claims}} 1}$$

### 为什么用 candidate_text 而非 candidate_claims

recall 应该基于模型的**原始输出**判断，而不是基于 Stage B 提取的 atomic claims。原因：

- extraction 可能遗漏隐含覆盖（如一段综合论述隐含了多条 gold claim）
- recall 关注"模型有没有说到"，不需要精确的 claim 对 claim 匹配

### 不做分层 Recall

Claim-Recall 只保留一个总指标，不按 perception / cognition / emotion 分层。分层精度由精确度侧的 LF-P / LF-C / LF-E 已经覆盖。

---

## 3. 完整评测流程

### Stage A: 候选模型生成

- 输入：image + 固定 user prompt（来自数据中的标准 prompt）
- 输出：`candidate_text`
- 每条样本独立调用一次待测模型

### Stage B: Claim Extraction（无 gold）

- 输入：`candidate_text`（不给 image，不给 gold）
- 模型：judge LLM（temperature=0）
- 输出：`candidate_claims[]`，每条为原子级评价判断
- 关键规则：不发明内容、跨层判断要拆分

### Stage C: Bidirectional Claim-Level Judging

这是核心阶段，**同时完成精确度打分和召回评估**。

**输入**：

- image
- candidate_claims[]（来自 Stage B）
- candidate_text（原始输出，用于 recall）
- gold observations[]
- gold claims[]
- gold final_outputs
- hard_negatives[]

**输出（精确度侧）**：对每条 candidate claim：

```json
{
  "cand_claim_id": "cand_01",
  "text": "...",
  "assigned_level": "perception | cognition | emotion",
  "matched_gold_claim_ids": ["claim_02"],
  "matched_obs_ids": ["obs_03"],
  "support_score": 1.0,
  "evidence_score": 1.0,
  "reason": "..."
}
```

**输出（召回侧）**：对每条 gold claim：

```json
{
  "gold_claim_id": "claim_01",
  "recall_score": 1.0,
  "matched_cand_claim_ids": ["cand_02", "cand_05"],
  "reason": "..."
}
```

### Stage D: Metrics Aggregation

在 Python 中完成纯计算，无 LLM 调用。 

---

## 4. 完整指标体系（7 个）

### 精确度侧（5 个，来自 instruction.md）


| 指标                        | 公式                              | 含义                     |
| ------------------------- | ------------------------------- | ---------------------- |
| Faithful-Precision (soft) | $\frac{1}{M}\sum s_i$           | candidate claims 平均支持度 |
| Evidence Grounding Score  | $\frac{1}{M}\sum s_i \cdot e_i$ | 证据锚定强度                 |
| LF-Perception             | $\frac{1}{                      | C_P                    |
| LF-Cognition              | $\frac{1}{                      | C_C                    |
| LF-Emotion                | $\frac{1}{                      | C_E                    |


### 召回侧（1 个）

- **Claim-Recall**：$CR = \frac{1}{|G|}\sum r_j$，衡量 gold claims 被候选模型输出覆盖的程度


---

## 5. 打分规则概要

### support_score

三档：1.0 / 0.5 / 0.0

- 上限规则保留：匹配到 `weakly_supported` 的 gold claim 时，上限 0.5
- observations 可以救回 faithful paraphrase（不需要原样命中 gold claim wording）
- observations + claims + final_outputs 均可作为判定依据

### evidence_score

五档：1.0 / 0.75 / 0.5 / 0.25 / 0.0

- image + observations + claims + final_outputs 均可作为 grounding 依据
- 三者经过校准，共同构成可信的证据来源

### recall_score（新增）

三档：1.0 / 0.5 / 0.0

- 基于 candidate_text 全文判断
- observations + gold claims + final_outputs 共同作为判断 recall 的参照

---

## 6. 代码实现结构

项目根目录（`[/Users/chengzecheng/Downloads/ArtcomBench_eval/](/Users/chengzecheng/Downloads/ArtcomBench_eval/)`）下新建以下文件：

- `eval_pipeline.py` — 主流程编排（Stage A/B/C/D）
- `prompts.py` — 所有 judge prompt 模板
- `metrics.py` — 指标聚合计算
- `schemas.py` — 输入输出 JSON schema 定义与校验
- `config.py` — 配置（模型名、temperature、路径等）
- `requirements.txt` — 依赖

### 关键依赖

- `openai` (或其他 LLM SDK)
- `pydantic` (schema 校验)
- `Pillow` (图片处理)
- `tqdm` (进度条)

---

## 7. 输出产物

每次评测运行保存：

- `outputs/candidate_texts.jsonl` — 候选模型原始输出
- `outputs/candidate_claims.jsonl` — 提取的 atomic claims
- `outputs/judged_results.jsonl` — per-claim 打分 + per-gold-claim recall
- `outputs/sample_metrics.jsonl` — 每条样本的 7 个指标
- `outputs/corpus_metrics.json` — 数据集级汇总

