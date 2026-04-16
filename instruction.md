可以，下面我直接按“**你已经有 500 条带 image + observations + claims + final_outputs 的 gold 数据，要评一个候选模型输出的 aesthetic commentary**”来设计一个**可直接 AI coding** 的 LLM-judge 评测框架。

你这条样本已经具备非常好的 gold 结构：有 **12 条 observations**、**11 条分层 claims**，claims 自带 **level（perception / cognition / emotion）** 和 **support_label（supported / weakly_supported）**，同时还有 `final_outputs.thinking_chain / commentary / short_summary`，以及一组 `hard_negatives`。这正适合做“**候选答案 claim-level judge → 指标聚合**”的框架。

---

## 1. 你这个 benchmark 应该怎么定义

你的评测对象不是“模型分数准不准”，而是：

**给定图像和固定用户 prompt，候选模型生成的 explanation，到底有多 faithful。**

候选模型的输入固定用你指定的 message：

```json
{
  "from": "human",
  "value": "You are an expert in evaluating the aesthetics of paintings.Dissect and analyze the aesthetic qualities manifested in the painting, with a focus on ten aspects: Layout and Composition, Space and Perspective, Light and Shadow, Color, Details and Texture, Theme and Logic, Mood, The Overall, Creativity, Sense of Order.\n<|image|>"
}
```

然后只评五个指标：

* Faithful-Precision（soft）
* Evidence Grounding Score
* LF-Perception
* LF-Cognition
* LF-Emotion

因为你只报 precision-style 指标，所以这个 benchmark **不惩罚没说到，只惩罚说错或说虚**。这点很重要。

---

## 2. gold 数据里的四层信息，分别怎么用

这里是整个框架最关键的设计。

### observations：一级事实源

`observations` 是**最高优先级的视觉事实锚点**。
它们是图像里可见的结构、颜色、空间、纹理、光影、物体等事实。judge 在给候选 claim 判分时，首先要看它能不能被 image + observations 直接支持。比如你这条样本里，“broad belt of white mist”“muted greens”“waterfall”“pavilion”“dry-brush textures”都在 observations 里有明确锚点。

### claims：二级规范判断

`claims` 是**规范化后的 gold judgment units**。
它们不只是视觉事实，还包含 level、dimension、support_label。这里最重要的不是 wording，而是：

* 这个判断属于 perception / cognition / emotion 哪一层
* 这个判断在 gold 中是 `supported` 还是 `weakly_supported`

所以 judge 在匹配到 gold claim 时，要把它当成**判分上限先验**。比如候选答案说“the mood is contemplative and secluded”，它可以匹配到 gold emotion claim，但 gold 自己就是 `weakly_supported`，那这个候选 claim 的 support 分**一般不能超过 0.5**。

### final_outputs：三级背景语义

`final_outputs` 不是 lexical target，不应该要求候选答案“复述得像”。
它的作用是：

* 给 judge 提供这条样本整体的**语义轮廓**
* 在候选 claim 比较抽象时，帮助判断它是不是和 gold 的整体 reasoning 一致
* 帮助 disambiguate paraphrase

但它**不能单独创造证据**。
也就是说：如果一句话图像里看不出来、observations 也不支持、claims 也没给依据，不能因为 `final_outputs.commentary` 里出现过类似说法就给高分。你的 `thinking_chain` 和 `commentary` 应该被视为“忠实背景信息”，不是强制参考答案。

### hard_negatives：反幻觉校准器

`hard_negatives` 很有用。
它们不是直接拿来算你这五个指标的主分，而是用于：

* judge prompt 中的“反例校准”
* 检查 judge 是否会误判 art-historical hallucination / false object / false composition
* 后续扩展到 hallucination 相关分析

比如你这条样本里的 “bright crimson sun dominates the upper sky” 就是标准 unsupported。

---

## 3. 整体评测流程：一定要做成两阶段

不要让 judge 一步同时“抽 claims + 打分”。
最稳的方式是两阶段。

### Stage A：候选模型生成答案

对每条样本：

1. 读取 image
2. 用你固定的 user prompt 调用待测模型
3. 拿到 raw commentary text

输出：

```json
{
  "sample_id": "...",
  "candidate_text": "..."
}
```

---

### Stage B：claim extraction

从 `candidate_text` 里抽 atomic claims。
这一阶段**不要给 gold**，避免 judge 被 gold 污染，导致“自动往 gold wording 靠”。

输出建议长这样：

```json
{
  "sample_id": "...",
  "candidate_claims": [
    {
      "cand_claim_id": "cand_01",
      "text": "The palette is dominated by muted greens with small blue and reddish accents."
    },
    {
      "cand_claim_id": "cand_02",
      "text": "The composition guides the eye upward from the lower waterfall toward the summit."
    },
    {
      "cand_claim_id": "cand_03",
      "text": "The mood is contemplative and secluded."
    }
  ]
}
```

抽取原则：

* 只抽**有判断含义**的内容
* 一个句子里如果有两个独立判断，要拆开
* 不要抽纯套话，比如 “this is a beautiful painting”
* 不要抽元话语，比如 “I think”, “overall speaking”
* 不要凭空补写

---

### Stage C：claim-level scoring judge

这一阶段才把以下信息一起交给 judge：

* image
* candidate claims
* gold observations
* gold claims
* gold final_outputs
* hard_negatives

judge 的任务是：**对每条 candidate claim 单独判分**。

---

## 4. claim-level 判分规则

这是整个框架的核心口径。

对每条 candidate claim，judge 最终至少要输出这些字段：

```json
{
  "cand_claim_id": "cand_01",
  "text": "...",
  "assigned_level": "perception | cognition | emotion",
  "matched_gold_claim_ids": ["claim_02"],
  "matched_obs_ids": ["obs_03", "obs_07"],
  "support_score": 1.0,
  "evidence_score": 1.0,
  "reason": "..."
}
```

---

## 5. support_score 怎么打

只用三档：

* **1.0** = supported
* **0.5** = weakly_supported
* **0.0** = unsupported / unverifiable

### 5.1 判分优先级

judge 应按这个优先级判断：

**image + observations > gold claims > final_outputs**

也就是：

1. 先看 candidate claim 能否被图像和 observations 直接支持
2. 再看是否和某个 gold claim 对齐
3. 再用 final_outputs 做语义消歧
4. final_outputs 不能单独充当证据

---

### 5.2 一个很关键的“上限规则”

如果 candidate claim 明确匹配某个 gold claim，则：

* 若 gold claim 是 `supported`，candidate 最多给到 **1.0**
* 若 gold claim 是 `weakly_supported`，candidate 最多给到 **0.5**

这个规则非常重要。
它能把你的 gold 中“高阶判断本身就不是满证据”的性质保留下来。比如你这条样本里的 mood / overall impression 都是 emotion 层且 gold 为 `weakly_supported`，judge 不该随便给 1 分。

---

### 5.3 observations 可以“救回”没撞 wording 的 faithful paraphrase

不要只按 gold claim 文本匹配。
如果 candidate claim **没有和某个 gold claim 明确对齐**，但它能被 observations 和 image 直接支持，也应该给分。

例如候选答案写：

> “A narrow waterfall appears at the lower left.”

这句话未必在 gold claims 里原样出现，但它显然被 `obs_06` 直接支持。
这种情况应该判：

* level = perception
* support_score = 1.0
* evidence_score = 1.0

否则你的 benchmark 会变成“背 gold claim wording”。

---

### 5.4 final_outputs 的正确用法

`final_outputs` 只能做这三件事：

* 判断 candidate claim 是否是在**合理 paraphrase** gold 的整体判断
* 处理比较抽象的 holistic statement
* 给 level 判定提供辅助背景

但它不能单独把一句缺乏图像证据的话抬成 supported。
比如如果候选答案写了具体历史意图、作者生平、宗教编码之类内容，而 observations / claims 都没有，`final_outputs` 也不该救它。

---

## 6. evidence_score 怎么打

你没有要求候选模型显式给 evidence span，所以这里的 EGS 最好做成**judge-based grounding strength**。

建议用五档：

* **1.0**：直接、清晰、强锚定到 image + observations
* **0.75**：较强 grounding，需要轻度综合多个 observations
* **0.5**：有 grounding，但明显带中度推断
* **0.25**：只有很弱的 grounding
* **0.0**：没有 grounding

### 推荐口径

#### 1.0

直接视觉可见，且 observations 已明确覆盖。
例如：

* muted greens with blue/reddish accents
* waterfall at lower left
* broad white mist across the middle

#### 0.75

不是单个 observation 原样复述，但能稳定地由多个 observations 合成出来。
例如：

* “the composition rises from the waterfall through the mist toward the summit”

这要综合 `obs_05 + obs_06 + obs_12`。

#### 0.5

图像和 gold 能给出一定根据，但仍有明显解释跳跃。
例如：

* “the mood is contemplative and secluded”

这类情绪判断通常不该给高 grounding。

#### 0.25

只有很弱的联系，基本靠想象或泛化。

#### 0.0

无证据，或和 hard negative 同类。

---

## 7. level 怎么判

每条 candidate claim 必须归到一个 level：

### Perception

直接视觉事实、局部形式、颜色、光影、纹理、空间可见关系。
典型词：

* pale mist
* muted green
* broken ink lines
* overlapping slopes
* waterfall
* pavilion

### Cognition

对组织、构图、主题、创意、整体整合度的分析。
典型词：

* guides the eye
* coherent landscape conception
* uneven integration
* organized lower detail
* conventional rather than inventive

### Emotion

情绪、氛围、观看感受、总体情感效应。
典型词：

* contemplative
* quiet and respectful
* serene
* deeply compelling

### 一个关键规则

如果一句话同时包含两个 level，要在 Stage B 就拆开。
例如：

> “The broad mist creates depth and a contemplative mood.”

应该拆成：

* “The broad mist contributes to spatial depth.” → cognition / perception 之间取最贴近者
* “The broad mist contributes to a contemplative mood.” → emotion

不要让一个 claim 跨层，不然后面 LF-Perception / LF-Cognition / LF-Emotion 会变脏。

---

## 8. 五个指标的最终公式

设一条样本抽出 (M) 条 candidate claims。
每条 claim 有：

* support score (s_i \in {1, 0.5, 0})
* evidence score (e_i \in {1, 0.75, 0.5, 0.25, 0})
* level (l_i \in {P,C,E})

### Faithful-Precision（soft）

[
FP_{soft} = \frac{1}{M}\sum_{i=1}^{M} s_i
]

---

### Evidence Grounding Score

[
EGS = \frac{1}{M}\sum_{i=1}^{M}(s_i \cdot e_i)
]

我建议这里一定乘上 (s_i)，因为一句 unsupported 的 claim，不该因为 judge 勉强觉得“有点像在说那块区域”就拿到 grounding 分。

---

### LF-Perception

[
LF_P = \frac{1}{|C_P|}\sum_{i \in C_P} s_i
]

### LF-Cognition

[
LF_C = \frac{1}{|C_C|}\sum_{i \in C_C} s_i
]

### LF-Emotion

[
LF_E = \frac{1}{|C_E|}\sum_{i \in C_E} s_i
]

其中：

* (C_P) = level 为 perception 的 candidate claims
* (C_C) = level 为 cognition 的 candidate claims
* (C_E) = level 为 emotion 的 candidate claims

---

## 9. 数据集级别怎么聚合

你的 500 条样本最终**不要先算每条样本均值再简单平均**。
更稳的是做 **micro-average**。

也就是：

### 数据集级 Faithful-Precision（soft）

[
FP_{soft}^{dataset} = \frac{\sum_i s_i}{\sum_i 1}
]

### 数据集级 EGS

[
EGS^{dataset} = \frac{\sum_i s_i e_i}{\sum_i 1}
]

### 数据集级 LF-P / LF-C / LF-E

分别在对应 level 的全部 candidate claims 上做 micro-average：

[
LF_P^{dataset} = \frac{\sum_{i \in P} s_i}{\sum_{i \in P} 1}
]

其余同理。

### 为什么不用 macro sample average

因为不同样本的输出长度不同。
micro-average 更真实地反映“模型说出来的 claim 总体有多 faithful”。

### 遇到某条样本没有 emotion claim 怎么办

这一条样本对 LF-Emotion **不进分母**。
不要给 0。因为你评的是 precision，不是 coverage。

---

## 10. 你的 judge 最好怎么写 prompt

下面给你可直接改造成代码模板的版本。

---

### Prompt A：Claim Extraction Judge

这个阶段不看 gold，只看 candidate answer。

```text
You are a careful evaluator extracting atomic claims from an aesthetic commentary about a painting.

Task:
Split the candidate commentary into minimal atomic evaluative claims.

Rules:
1. Do not invent content not explicitly present in the candidate commentary.
2. If one sentence contains multiple independent judgments, split them.
3. Keep each claim semantically minimal but still meaningful.
4. Ignore filler, hedging, and meta-discourse.
5. Do not score or judge yet.
6. Output JSON only.

Return schema:
{
  "candidate_claims": [
    {
      "cand_claim_id": "cand_01",
      "text": "..."
    }
  ]
}

Candidate commentary:
{{candidate_text}}
```

---

### Prompt B：Claim Scoring Judge

这个阶段看 image + gold + candidate claims。

```text
You are a strict faithfulness judge for image-grounded aesthetic commentary.

You must evaluate each candidate claim against:
1. the painting image
2. gold observations (primary factual anchors)
3. gold claims (canonical evaluative propositions with level and support prior)
4. gold final_outputs (background semantic guidance only, NOT lexical target)
5. hard negatives (examples of unsupported or unverifiable content)

Scoring priorities:
- Image + observations have the highest authority.
- Gold claims are the second authority.
- final_outputs are background only and cannot create evidence by themselves.
- If a candidate claim matches a gold claim whose support_label is weakly_supported, the candidate support_score should normally not exceed 0.5.
- If a candidate claim is directly grounded in image + observations but not explicitly phrased in a gold claim, it can still receive full support.
- Unsupported, speculative, art-historical, author-intent, or non-visible claims should receive 0.

For each candidate claim, output:
- assigned_level: perception / cognition / emotion
- matched_gold_claim_ids
- matched_obs_ids
- support_score: 1.0 / 0.5 / 0.0
- evidence_score: 1.0 / 0.75 / 0.5 / 0.25 / 0.0
- short_reason

Return JSON only.

Input:
{
  "candidate_claims": {{candidate_claims_json}},
  "gold_observations": {{observations_json}},
  "gold_claims": {{claims_json}},
  "gold_final_outputs": {{final_outputs_json}},
  "hard_negatives": {{hard_negatives_json}}
}
```

---

## 11. 你的框架里最值得加的三个工程约束

### 1) judge 温度固定为 0

因为你要做稳定 benchmark，不是 creative generation。

### 2) 所有 judge 输出必须过 JSON schema 校验

一旦格式错：

* 自动重试 1~2 次
* 仍失败则记日志，不 silently skip

### 3) 保留 per-claim 中间结果

最终不只存 summary metrics，还要存每条 claim 的 judge 结果。
后面你做 error analysis、论文 case study、或者分析 LF-E 为什么低，都靠这个。

---

## 12. 建议的数据结构

### 输入样本

```json
{
  "sample_id": "...",
  "image_path": "...",
  "gold": {
    "observations": [...],
    "claims": [...],
    "final_outputs": {...},
    "hard_negatives": [...]
  }
}
```

### 候选模型输出

```json
{
  "sample_id": "...",
  "candidate_text": "..."
}
```

### claim extraction 输出

```json
{
  "sample_id": "...",
  "candidate_claims": [...]
}
```

### judge 输出

```json
{
  "sample_id": "...",
  "judged_claims": [
    {
      "cand_claim_id": "cand_01",
      "text": "...",
      "assigned_level": "perception",
      "matched_gold_claim_ids": ["claim_02"],
      "matched_obs_ids": ["obs_03", "obs_07"],
      "support_score": 1.0,
      "evidence_score": 1.0,
      "short_reason": "Directly visible and explicitly anchored by muted greens and colored accents."
    }
  ]
}
```

### per-sample metrics

```json
{
  "sample_id": "...",
  "num_claims": 8,
  "faithful_precision_soft": 0.6875,
  "evidence_grounding_score": 0.59375,
  "lf_perception": 1.0,
  "lf_cognition": 0.625,
  "lf_emotion": 0.5
}
```

### corpus summary

```json
{
  "num_samples": 500,
  "num_total_claims": 4217,
  "faithful_precision_soft": 0.71,
  "evidence_grounding_score": 0.61,
  "lf_perception": 0.84,
  "lf_cognition": 0.68,
  "lf_emotion": 0.49
}
```

---

## 13. 结合你这条样本，judge 应该怎么判

你这条 gold 本身已经给了非常清楚的层级结构：
perception 侧重 light/color/texture/space，cognition 侧重 composition/order/theme/creativity/overall，emotion 侧重 mood/overall impression；而 `final_outputs` 又把它们串成了一条完整 reasoning。

所以如果候选模型写：

### 例 1

> “Muted greens dominate the lower slopes, with small blue and reddish accents.”

建议判：

* matched_gold_claim_ids = [`claim_02`]
* matched_obs_ids = [`obs_03`, `obs_07`]
* level = perception
* support_score = 1.0
* evidence_score = 1.0

---

### 例 2

> “The composition guides the eye upward from the waterfall toward the summit.”

建议判：

* matched_gold_claim_ids = [`claim_05`]
* matched_obs_ids = [`obs_05`, `obs_06`, `obs_12`]
* level = cognition
* support_score = 1.0
* evidence_score = 0.75 或 1.0

---

### 例 3

> “The mood is contemplative and secluded.”

建议判：

* matched_gold_claim_ids = [`claim_10`]
* matched_obs_ids = [`obs_02`, `obs_05`, `obs_12`]
* level = emotion
* support_score = 0.5
* evidence_score = 0.5 或 0.75

因为 gold 这条 mood claim 本身就是 `weakly_supported`。

---

### 例 4

> “A bright crimson sun dominates the upper sky.”

建议判：

* matched_gold_claim_ids = []
* matched_obs_ids = []
* level = perception
* support_score = 0.0
* evidence_score = 0.0

这和你的 hard negative 直接同型。

---

## 14. 最后给你一个可直接让 AI coding 开始写的实施顺序

### 第一步

写 `run_candidate_model(sample)`
输入 image + 固定 human prompt，输出 `candidate_text`

### 第二步

写 `extract_candidate_claims(candidate_text)`
调用 Prompt A，输出 `candidate_claims`

### 第三步

写 `judge_claims(sample, candidate_claims)`
调用 Prompt B，输出 `judged_claims`

### 第四步

写 `aggregate_sample_metrics(judged_claims)`
输出五个 sample-level metrics

### 第五步

写 `aggregate_dataset_metrics(all_judged_claims)`
输出 corpus-level 五个主指标 + 各 level claim 数

### 第六步

写 `save_artifacts(...)`
至少保存：

* candidate raw text
* candidate claims
* judged claims
* per-sample metrics
* corpus metrics

### 第七步

加异常处理

* JSON parse retry
* judge response schema validation
* missing image / empty output logging
* deterministic config logging

---

## 15. 我最建议你坚持的判分哲学

一句话：

**让 observations 决定“看没看到”，让 claims 决定“该归哪层、该给多大支持上限”，让 final_outputs 决定“整体语义有没有跑偏”，但永远不要让 final_outputs 单独制造证据。**

这样你的框架就会同时具备：

* 对 paraphrase 友好
* 对 hallucination 严格
* 对 perception / cognition / emotion 三层可诊断
* 能直接产出你要的五个指标

下一条我可以直接给你一版 **完整可执行的 judge 输出 JSON schema + Python 伪代码/函数骨架**。
