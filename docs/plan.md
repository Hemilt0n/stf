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

### Route A: Geo-Edit Residual Flow

一句话：
- 把时空融合视为“地理对齐前提下的局部编辑 + 全局季节迁移”，在 flow path 上对变化区域给予更强生成自由度，而不是对整张图统一加噪。

核心机制：
- 保持 `t1` 与 `t2` 空间对齐这一遥感先验。
- 基于 `|coarse_t2 - coarse_t1|` 构造软变化图 `m(x)`，把它作为编辑强度图，而不是硬标签。
- 在 `ResidualGaussianFlowMatching` 中不再使用统一起点分布，而是定义空间变化感知的起点：
  - 不变区域：起点接近 `0 residual` 或轻微全局季节偏移
  - 变化区域：起点接近 `coarse_delta` 附近，并叠加更高噪声
- 可写成：
  - `z_mean = (1 - m) * g + m * (lambda * coarse_delta)`
  - `sigma(x) = sigma_low + (sigma_high - sigma_low) * m`
  - `z = z_mean + sigma(x) * eps`
  - `x_t = (1 - alpha_t) * z + alpha_t * delta`
  - `u_t = alpha'_t * (delta - z)`
- 其中 `g` 表示全局/低频季节项，承接“整体风格迁移”部分。

解决的问题：
- 当前全局 warmup 是对整张图一刀切，不区分哪里该保留、哪里该重建。
- 对 residual flow 而言，如果直接在输入图上全局加噪，还会连带改变监督目标语义。
- Geo-Edit Residual Flow 改的是 flow 起点分布，而不是 ground-truth 本身，更符合流模型底层 transport 视角。

为什么好讲故事：
- 不是“整图重新生成”，而是“在地理对齐约束下对变化区域进行编辑”。
- 局部地物变化对应图像编辑 / inpainting。
- 全局季节变化对应风格迁移 / 低频迁移。
- 故事可以概括为：
  - “遥感时空融合不是一类统一生成问题，而是局部编辑与全局季节迁移的组合问题。”

工程切入点：
- 第一版只改 `ResidualGaussianFlowMatching` 的起点定义与噪声形式，不一定先改 backbone。
- 先用 `coarse change map` 作为软编辑掩码，不急着上 learned trust。
- 第一阶段避免直接改 `fine_t1` 原图，而是改 residual flow 的 `z_mean` 与 `sigma(x)`。

风险：
- `coarse` 变化图分辨率低，不能当精确标签，必须使用软图和平滑。
- 如果只做局部高噪声，不建模全局低频季节项，可能只能处理局部变化，难以解释整体风格偏移。
- 边界区域若掩码过硬，容易在编辑/非编辑交界处产生伪影。

推荐级别：
- `P1`

### Route B: Condition Trust Map

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

### Route C: Static/Change Dual-Head

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

### Route D: Coarse-Guided Routing / Mixture-of-Experts

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

### Route E: Coarse-to-Fine Deformable Alignment

一句话：
- 先学对齐，再做融合。

适用问题：
- 地物位移、边界错位、形状变化导致的“看起来像，但空间上不对齐”。

故事点：
- “变化不仅是辐射变化，也包含空间布局变化。”

推荐级别：
- `P2`

### Route F: Frequency-Decoupled Decoder

一句话：
- 低频结构更多看 `coarse_t2`，高频纹理有条件地继承 `fine_t1`。

适用问题：
- 细节恢复和语义真实性之间的冲突。

故事点：
- “粗分辨率决定变成什么，高分历史决定纹理长什么样。”

推荐级别：
- `P2`

### Route G: Boundary / Topology Head

一句话：
- 给变化边界和结构连续性单独一条头。

适用问题：
- 道路、河流、建筑边缘等高变化边界区域。

故事点：
- “大变化问题首先坏掉的不是均值，而是边界和拓扑。”

推荐级别：
- `P2`

## 5. 研究味更强但风险更高的路线

### Route H: Event Token Memory

一句话：
- 把“建筑新增、农田收割、水体扩张”等变化模式做成 prototype memory，让变化区域检索。

优点：
- 对稀有大变化有潜力。

缺点：
- 工程复杂度高，验证周期长。

推荐级别：
- `P3`

### Route I: Reversible Dual-Time Network

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

1. `Geo-Edit Residual Flow`
2. `Condition Trust Map`
3. `Static/Change Dual-Head`
4. `Coarse-Guided Routing`
5. `Boundary / Topology Head`
6. `Frequency-Decoupled Decoder`

原因：
- Route A 直接从 flow 起点分布层面重写问题，把任务从“整图生成”改写成“局部编辑 + 全局迁移”。
- Route B 解决“历史条件该不该信”的核心矛盾。
- Route C 解决“复制”和“生成”任务冲突。
- Route D 适合作为第二阶段增强，因为它进一步把“变化区域专门处理”结构化。

## 7. 推荐主线

当前最推荐的主线不是“换一个更深的 backbone”，而是：

### 主线方案：Geo-Edit Residual Flow -> Trust-Conditioned Dual-Head Fusion

核心组成：
- `soft edit mask`
- `spatially varying residual start distribution`
- `optional seasonal low-frequency term`
- `trust map`
- `static carry head`
- `change synthesis head`
- `change gate`

一句话概括：
- 先在 flow path 上把任务改写成局部编辑问题，再判断历史高分条件哪里可信，最后把预测显式拆成静态继承与变化生成两条路径。

这个方案的优点：
- 能完整解释当前 observed issue
- 兼容现有 `flow` / `residual-flow` 包装器
- 第一阶段甚至可以不先改 backbone，只改 residual flow 起点定义
- 便于逐步实现，不必一次重写全部主干
- 后续可以自然扩展到 routing / boundary head

## 8. 最小实现路径

### Stage 1

目标：
- 先从 flow 定义层做最小侵入验证，确认“局部编辑式 residual flow”是否成立。

建议实现：
- 在 `ResidualGaussianFlowMatching` 中引入软变化图 `m(x)`
- 用 `m(x)` 定义空间变化感知的 `z_mean` 与 `sigma(x)`：
  - 不变区低噪声、接近 identity residual
  - 变化区高噪声、接近 `coarse_delta`
- 不直接修改 `fine_t1` 原图，不改变 ground-truth 语义
- 暂时保留当前 `change_loss_weight` 与 `coarse_consistency_weight`

预期收益：
- 判断“局部编辑式 flow 起点”是否比全局 warmup / 统一高斯起点更有效

### Stage 2

目标：
- 如果 Stage 1 有效，再把“编辑区域该不该信历史条件”显式结构化。

建议实现：
- 在现有 `PredTrajNet` 前部增加 `trust map head`
- 让 `fine` 分支特征按 `trust map` 调制
- 继续使用 Stage 1 的空间变化感知 flow 起点

预期收益：
- 判断“显式条件信任”是否能进一步减少复制 `fine_t1`

### Stage 3

目标：
- 如果 trust gating 有效，再做静态/变化双头输出。

建议实现：
- 在 decoder 末端分出：
  - `carry head`
  - `change head`
  - `gate head`
- 输出融合结果作为最终预测

预期收益：
- 把静态继承与变化重建明确分工

### Stage 4

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

如果 Route A/B/C 只提高 `PSNR/SSIM` 但 `TRP` 或高变化视觉不改善，不应视为真正成功。

## 10. 当前建议

下一步优先讨论并实现的路线建议为：

1. 先做 `Geo-Edit Residual Flow`
2. 若结果积极，再接 `Condition Trust Map`
3. 再推进到 `Trust-Conditioned Dual-Head Fusion`
4. 暂不优先投入 `Event Token Memory` 和 `Reversible Dual-Time Network`

原因很直接：
- 这条路径最贴流模型底层原理
- 最容易把“地理对齐 + 局部编辑 + 全局季节迁移”的故事讲完整
- 最容易形成完整研究叙事
- 最适合在当前工程上逐步落地和验证
