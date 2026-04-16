# STF 下一阶段网络改进路线图

## 1. 当前判断

当前主线问题已经比较明确：
- 模型并不只是“容量不够”，而是没有显式建模 `fine_t1` 在不同区域的可信度差异。
- 低变化区域往往适合继承 `fine_t1` 的纹理与结构。
- 高变化区域如果仍然沿用 `fine_t1`，就会出现复制旧时相细节的问题。
- 一旦粗暴压制 `fine_t1`，网络又容易在变化区域产生幻觉块，因为缺少可靠的替代生成路径。

因此，下一阶段网络改进的核心，不应只是继续堆深主干，而应转向：
- 显式建模“哪些区域该信 `fine_t1`”
- 显式拆分“静态继承”和“变化生成”
- 让 `coarse_t2` 真正参与到网络路由，而不仅仅体现在 loss 上

## 2. 主叙事

最容易讲清楚、也最贴合实际遥感场景的故事主线是：

> 时空融合的核心矛盾，不是生成能力不足，而是历史高分条件 `fine_t1` 在不同区域的可信度并不一致。  
> 因此网络不应把所有像素都当成“同一种融合任务”，而应显式区分静态继承与变化重建。

这个叙事有三个优点：
- 和当前观测到的实验现象一致
- 容易对应真实场景（农田收割、水体变化、建筑新增）
- 后续无论走 gate、双头、专家路由，都能纳入同一套故事框架

## 3. 优先候选路线

### Route A: Condition Trust Map

一句话：
- 让网络学习一个逐像素 `trust(fine_t1)`，决定每个位置该多大程度信任历史高分图像。

核心机制：
- 在 backbone 前增加一个轻量 trust head，输入 `fine_t1 / coarse_t1 / coarse_t2 / 当前 flow state`。
- 输出一张 `trust map`。
- 低变化区高 trust，允许保留 `fine_t1` 的细节注入。
- 高变化区低 trust，减少 `fine_t1` 条件，更多依赖 `coarse delta` 与 residual state。

解决的问题：
- 避免全局 `dropout` 或全局 warmup 过于粗糙。
- 把“条件是否可信”从训练技巧变成网络结构能力。

为什么好讲故事：
- 不是简单“给历史图加噪声”，而是让模型学会判断“历史信息在什么地方已经过时”。

工程切入点：
- 可先只做 feature gating，不改训练主循环。
- 适合接在当前 `PredTrajNet` 的 `fine` 分支或 early fusion 之前。

风险：
- 如果 trust map 监督完全隐式，前期可能学成保守平均策略。

推荐级别：
- `P1`

### Route B: Static/Change Dual-Head

一句话：
- 把最终预测拆成“静态继承头”和“变化生成头”，再通过 change gate 融合。

核心机制：
- `carry head`: 负责不变区域从 `fine_t1` 继承什么。
- `change head`: 负责变化区域新生成什么。
- `gate head`: 预测逐像素变化融合权重。
- 最终输出：
  - `pred = (1 - gate) * carry + gate * change`

解决的问题：
- 当前单输出头把“复制”和“生成”这两个冲突任务混在同一条路径里。
- 将问题重写为两类子任务，更符合实际场景结构。

为什么好讲故事：
- “时空融合不是单一回归，而是静态继承与变化重建的解耦。”

工程切入点：
- 保留现有 encoder-decoder 主干，只改 decoder 后端与输出头。
- 训练上可以沿用现有主损失，再考虑对 gate 增加附加约束。

风险：
- 如果 gate 不稳定，可能出现边界区抖动。

推荐级别：
- `P1`

### Route C: Coarse-Guided Routing / Mixture-of-Experts

一句话：
- 用粗分辨率变化信息把像素或区域路由到不同专家，而不是所有位置都走同一主干。

核心机制：
- 基于 `|coarse_t2 - coarse_t1|` 或 learned change cue 生成 routing map。
- 设计两个或三个专家：
  - `stable expert`
  - `change expert`
  - 可选 `uncertain expert`
- 按位置或按窗口做动态专家融合。

解决的问题：
- 低变化区和高变化区本来就需要不同特征提取逻辑。
- 统一 backbone 容易在两类任务之间折中。

为什么好讲故事：
- “变化区域是少数但高价值样本，应由专门专家负责。”

工程切入点：
- 先做轻量双专家，不必一步到完整 MoE。
- 也可以先在 bottleneck 或 decoder 中后段做路由，而不是全网络路由。

风险：
- 实现复杂度和训练不稳定性都高于 Route A/B。

推荐级别：
- `P2`

## 4. 第二梯队路线

### Route D: Coarse-to-Fine Deformable Alignment

一句话：
- 先学对齐，再做融合。

适用问题：
- 地物位移、边界错位、形状变化导致的“看起来像，但空间上不对齐”。

故事点：
- “变化不仅是辐射变化，也包含空间布局变化。”

推荐级别：
- `P2`

### Route E: Frequency-Decoupled Decoder

一句话：
- 低频结构更多看 `coarse_t2`，高频纹理有条件地继承 `fine_t1`。

适用问题：
- 细节恢复和语义真实性之间的冲突。

故事点：
- “粗分辨率决定变成什么，高分历史决定纹理长什么样。”

推荐级别：
- `P2`

### Route F: Boundary / Topology Head

一句话：
- 给变化边界和结构连续性单独一条头。

适用问题：
- 道路、河流、建筑边缘等高变化边界区域。

故事点：
- “大变化问题首先坏掉的不是均值，而是边界和拓扑。”

推荐级别：
- `P2`

## 5. 研究味更强但风险更高的路线

### Route G: Event Token Memory

一句话：
- 把“建筑新增、农田收割、水体扩张”等变化模式做成 prototype memory，让变化区域检索。

优点：
- 对稀有大变化有潜力。

缺点：
- 工程复杂度高，验证周期长。

推荐级别：
- `P3`

### Route H: Reversible Dual-Time Network

一句话：
- 同时建模 `t1 -> t2` 与 `t2 -> t1`，用时间反向一致性抑制胡编变化。

优点：
- 理论叙事强，适合后续论文扩展。

缺点：
- 当前阶段偏重，容易分散主线。

推荐级别：
- `P3`

## 6. 选择建议

如果目标是“创新性 + 实际可落地 + 容易讲故事 + 不脱离当前仓库”，建议优先级如下：

1. `Condition Trust Map`
2. `Static/Change Dual-Head`
3. `Coarse-Guided Routing`
4. `Boundary / Topology Head`
5. `Frequency-Decoupled Decoder`

原因：
- Route A 直接解决“历史条件该不该信”的核心矛盾。
- Route B 直接解决“复制”和“生成”任务冲突。
- Route C 适合作为第二阶段增强，因为它进一步把“变化区域专门处理”结构化。

## 7. 推荐主线

当前最推荐的主线不是“换一个更深的 backbone”，而是：

### 主线方案：Trust-Conditioned Dual-Head Fusion

核心组成：
- `trust map`
- `static carry head`
- `change synthesis head`
- `change gate`

一句话概括：
- 先判断历史高分条件哪里可信，再把预测显式拆成静态继承与变化生成两条路径。

这个方案的优点：
- 能完整解释当前 observed issue
- 兼容现有 `flow` / `residual-flow` 包装器
- 便于逐步实现，不必一次重写全部主干
- 后续可以自然扩展到 routing / boundary head

## 8. 最小实现路径

### Stage 1

目标：
- 先做最小侵入验证，确认结构改动方向正确。

建议实现：
- 在现有 `PredTrajNet` 前部增加 `trust map head`
- 让 `fine` 分支特征按 `trust map` 调制
- 不改 loss 主框架，只保留当前 `change_loss_weight` 与 `coarse_consistency_weight`

预期收益：
- 判断“显式条件信任”是否比全局 `dropout/warmup` 更有效

### Stage 2

目标：
- 如果 Stage 1 有效，再做静态/变化双头输出。

建议实现：
- 在 decoder 末端分出：
  - `carry head`
  - `change head`
  - `gate head`
- 输出融合结果作为最终预测

预期收益：
- 把静态继承与变化重建明确分工

### Stage 3

目标：
- 在确认双头有效后，再引入专家路由或边界头增强。

建议实现：
- 先做 bottleneck 级别双专家
- 或先加变化边界辅助头

## 9. 评估重点

下一阶段不应只看 overall scalar，还要重点看：
- `TRP`
- 高变化桶指标
- 变化边界视觉质量
- 是否减少“复制 `fine_t1`”和“幻觉地物块”

如果 Route A/B 只提高 `PSNR/SSIM` 但 `TRP` 或高变化视觉不改善，不应视为真正成功。

## 10. 当前建议

下一步优先讨论并实现的路线建议为：

1. 先做 `Condition Trust Map`
2. 若结果积极，继续推进到 `Trust-Conditioned Dual-Head Fusion`
3. 暂不优先投入 `Event Token Memory` 和 `Reversible Dual-Time Network`

原因很直接：
- 这条路径最贴问题本质
- 最容易形成完整研究叙事
- 最适合在当前工程上逐步落地和验证
