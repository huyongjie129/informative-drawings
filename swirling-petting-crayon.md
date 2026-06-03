# 绘画机器人路径优化 — 端到端神经笔画生成器

## Context（背景与目标）

**现有方案**：语音 → ASR → Infinity (T2I) → Informative Drawing（线条风格化）→ vtracer（位图转 SVG）→ SVG 转 gcode → UltraArm P340 绘画。

**瓶颈定位**：除了机械臂绘画本身耗时数分钟外，其它环节均为秒级。vtracer 通常输出大量短小破碎的 SVG 片段，导致 **30%–50% 的总耗时浪费在 pen-up 空走（抬笔移动）**上。当前 pipeline 把"画什么"和"怎么画"分成两个互不通信的阶段：vtracer 只管几何拟合，不知道哪种笔画排布省时；下游的 SVG→gcode 也不会重新排序。

**优化思路**：训练一个**摊销（amortized）的前馈神经笔画生成器**，同时优化"画什么 + 怎么排序 + 走哪条方向"，让模型在视觉相似度损失之外承担一个由 UltraArm P340 真实运动模型推导的可微"绘画耗时损失"。

**⚠️ 与 CLIPasso 的本质区别（关键澄清）**：
- **CLIPasso 是 per-image optimization**：对每张新图都要跑 ~2000 步梯度下降通过 DiffVG，单张 ~6 分钟。这个范式如果直接套用，反而会让整个系统更慢，与我们的目标背道而驰。
- **我们要做的是 amortized feedforward model**：训练阶段一次性在大数据集上把"如何把图像变笔画序列"的能力压进网络权重；推理阶段就是一次普通前向（autoregressive 256 步 × 10ms ≈ **2.5 秒**），不做任何在线优化。
- **CLIPasso 在本方案中只出现在两处**：(1) 借用它的训练目标形式（DiffVG 渲染 + 视觉相似度损失，但只在我们训练自己的网络时用）；(2) 用 CLIPasso 离线生成一批 `(image, stroke_set)` teacher 数据，给我们的解码器做暖启（见后文"CLIPasso 教师启动"）。
- **类比**：CLIPasso 之于本方案 ≈ 经典 NeRF（per-scene optimization）之于 Instant-NGP / generalizable NeRF（feedforward）。我们做的是"generalizable" 版本。

让笔画顺序作为输出的一部分被联合优化是本方案的核心创新——不是抄 CLIPasso。

**为什么端到端优于"先生成再排序"**：纯粹的 TSP 重排只能在固定的笔画集合上找最短遍历；联合优化允许模型把一条笔画的端点挪 5mm，从而消除一段 50mm 的抬笔空走。几何与顺序的耦合是这一方案的核心收益来源。

**为什么 ML（这里用可微优化 + 可选 RL fallback）优于纯经典方法**：经典 LKH/OR-Tools 是 Phase 1 的必备 baseline，但只能优化排序、动不了几何；当 Phase 1 已经达到瓶颈后，进一步加速必须改造几何，这只能通过可微/可学习的方法做到。

---

## 推荐方案

### 总体结构

```
Infinity (保留) ──► Informative Drawing (保留) ──► 神经笔画生成器 ──► gcode ──► P340
                                                       ▲
                                                       │
                                          可微时间损失 + 视觉损失联合训练
```

替换的是 vtracer + SVG→gcode 中的"几何拟合 + 顺序"部分；style transfer 与 T2I 保留（避免一次改两个模块）。

### 模型架构

- **图像编码器**：从头训练的 ConvNeXt-T（CLIP ViT 对线稿过强）。
- **自回归 Bezier 解码器**：6 层 Transformer decoder，d_model=256，每步输出一条笔画：
  - 4 个 cubic Bezier 控制点（normalized [0,1]²，8 维）
  - 1 个方向 logit（见下）
  - 1 个 `[STOP]` logit
  - 笔宽 v1 固定为常量（P340 笔筒夹的笔粗细物理固定，无 Z 轴压力）
- **可微栅格化**：DiffVG（[github.com/BachiLi/diffvg](https://github.com/BachiLi/diffvg)，论文 [tzumao/diffvg](https://people.csail.mit.edu/tzumao/diffvg/)）
- **方向选择**：每条笔画一个 Gumbel-softmax 方向标量，控制 P0→P3 还是 P3→P0；训练时 soft（τ 从 1.0 退火到 0.1），推理时 hard 阈值 0.5。Bezier 反向是免费的（控制点倒序）。

**为什么用自回归而非 set-output + 独立 orderer**：set-output 的排列不变性与"相邻笔画端点距离"目标冲突；自回归解码在生成第 i+1 条笔画时可以看到第 i 条的真实端点，让几何能因排序而调整。Paint Transformer（[arxiv 2108.03798](https://arxiv.org/abs/2108.03798)）是相关先例。

### 损失函数

`L_total = L_LPIPS + 0.5·L_L1 + λ(t)·L_time + 0.01·L_count`

- **L_visual**（L_LPIPS + L_L1）：DiffVG 把笔画渲染到白底画布，与 Informative Drawing 输出对比。
- **L_time**：**可微梯形速度模型**（关键创新点）
  - 单笔绘制时间：`arc_length(Bezier) / v_draw`，弧长用 16-point Gauss-Legendre 积分（可微，快）
  - 笔间空走时间（梯形 profile，距离 d，最大速度 v_max，加速度 a）：
    - 距离够长（达到 cruise）：`t = d/v_max + v_max/a`
    - 短距离（三角 profile）：`t = 2·sqrt(d/a)`
    - 两支用 `torch.where`，piecewise 可微
  - 每次抬/落笔加常量 `t_pen_toggle ≈ 0.3s`（实测）
  - **必须先做机器人标定**：测 20 段不同距离的实际移动时间，最小二乘拟合 `(v_draw, v_travel, a)`。预估 `v_draw≈30mm/s, v_travel≈80mm/s, a≈200mm/s²`，但务必实测。
- **L_count**：平均笔画数，弱正则。
- **λ 退火**：epoch 0–10 用 λ=0 训纯视觉质量；epoch 10–40 线性升到 λ_target（让 L_time 梯度量级与 L_LPIPS 相当）；epoch 40–60 fine-tune。**不能上来就高 λ，否则模型会塌成"画一条线就 STOP"**。
- **方向 logit 的梯度只来自 L_time**（视觉损失对方向无感），所以方向学习完全由 λ 驱动；要单独记录"方向翻转率 vs epoch"来确认它在学。

### 训练数据

- Infinity 生成 50k–100k 张 256×256 图像，再用 Informative Drawing 得线稿（自动构造无监督训练对）。
- Quick-Draw（~5M 草图）做笔画先验的预训练。
- 训练分辨率 256×256，推理可上 512×512。

### 关键稳定性技巧：CLIPasso 教师启动（仅训练时使用，不影响推理速度）

冷启动时 decoder 不知道笔画长什么样，autoregressive 联合训练极易发散。Bootstrap 流程：

1. **离线一次性**：用现成 CLIPasso（[arxiv 2202.05822](https://arxiv.org/abs/2202.05822)，[github.com/yael-vinker/CLIPasso](https://github.com/yael-vinker/CLIPasso)）跑 5k 张图，每张 ~6 分钟，约 3 天单卡跑完，得到 `(image, stroke_set)` 对。**这是一次性预处理，不进入在线推理路径**。
2. 把笔画集按 nearest-neighbor 规范排序后，对 decoder 做 5 epoch 的 teacher-forcing 预训练。
3. 再进入 §损失函数 中的 λ 退火联合训练。

> 再次强调：上线后用户每次说一句话生成一张图，**不会再调用 CLIPasso**——只走我们训好的解码器，一次前向 ~2.5s。

### Fallback：两阶段方案（如果 Phase 2 第 6 周联合训练仍不稳定）

- **Stage A**：直接复用 CLIPasso 生成笔画集合（set output，无顺序、无方向）。
- **Stage B**：用 **POMO**（[arxiv 2010.16011](https://arxiv.org/abs/2010.16011)，比 Kool 的 Attention Model [arxiv 1803.08475](https://arxiv.org/abs/1803.08475) 更稳）训一个 learned orderer：
  - 输入 `(N, 4)` 张量（每条笔画两端点 P0、P3）
  - 输出排列 + 每条笔画方向
  - REINFORCE，reward = −T_time
  - 单卡 1–2 天可训完，N≤256 完全够用
- **不选 DIFUSCO**（[arxiv 2302.08224](https://arxiv.org/abs/2302.08224)）作为 fallback：扩散更重、推理更慢，在 N≤256 的小规模问题上 POMO 足矣。DIFUSCO 适合 N≥1000 的工业级 TSP。

代价：失去几何-顺序联合优化（凭直觉约 10–15% 的速度收益没了），但排序-only 的收益依然显著。

---

## 实施分阶段路线图（共 8 周，含 fallback 10 周）

### Phase 1（第 1 周）— Baseline，必须先做
**没有这一步，后面 ML 的提速宣称都不可信。**

1. **机器人运动标定**：写 `calibration/measure_motion.py`，让 P340 执行 20 段不同距离的移动，log 实际完成时间，拟合 `(v_draw, v_travel, a)`，输出 `motion_params.json`。
2. **LKH-TSP 重排基线**：写 `baseline/lkh_reorder.py`，读 vtracer SVG → 每个 path 拆为带两端点的笔画 → 用 §3 的标定耗时构 cost matrix → 调用 **LKH-3**（[webhotel4.ruc.dk/~keld/research/LKH-3/](http://webhotel4.ruc.dk/~keld/research/LKH-3/)，Python 封装 `elkai`）解（允许笔画反向 → 转化为 asymmetric TSP 或带配对约束的 2N-node 对称 TSP，LKH 都支持）→ 输出重排后的 gcode。
3. **Benchmark 集**：5 类（人脸/动物/物体/场景/抽象）共 50 张图。
4. **在真机上测**：Phase 1 vs 原 pipeline 的实际绘画耗时。

**预期收益：30–45% 时间下降**。这是 Phase 2 必须超越的门槛——如果 Phase 1 已达可接受水平，Phase 2 未必有必要，这本身就是有价值的发现。

### Phase 2（第 2–6 周）— 端到端神经笔画生成器
- **第 2–3 周**：实现 decoder、DiffVG 渲染模块、visual loss；用 CLIPasso teacher 数据 λ=0 预训，复现 CLIPasso 视觉质量；实现 time loss 并做单元测试（对照数值积分）。
- **第 4–6 周**：λ 退火、curriculum、完整数据集；多轮 ablation（λ 值、是否学方向、是否 curriculum）。
- **第 6 周末决策点**：联合训练产出是否比 Phase 1 再降 ≥20%？若否，走 Phase 3 fallback。

### Phase 3（第 7–8 周）— 真机集成与对比评测
- gcode 生成、真机绘制、完整对比表、用户研究。

### 可选 Phase 4（第 9–10 周）— Fallback
- 若 Phase 2 决策为负，实现 §Fallback 中的 CLIPasso + POMO 方案。

---

## 关键文件清单（新建项目结构）

| 路径 | 用途 |
|---|---|
| [calibration/measure_motion.py](calibration/measure_motion.py) | Week 1：P340 运动标定 |
| [baseline/lkh_reorder.py](baseline/lkh_reorder.py) | Phase 1：LKH-TSP 重排基线 |
| [models/stroke_decoder.py](models/stroke_decoder.py) | 自回归 Transformer Bezier 解码器 |
| [models/render.py](models/render.py) | DiffVG 批量栅格化封装 |
| [losses/time_loss.py](losses/time_loss.py) | 可微梯形耗时模型（关键、需单元测试） |
| [losses/visual_loss.py](losses/visual_loss.py) | LPIPS + L1 + 白底合成 |
| [train/pretrain_clipasso.py](train/pretrain_clipasso.py) | 生成 CLIPasso teacher 数据并预训 decoder |
| [train/train_joint.py](train/train_joint.py) | 主训练循环、λ 退火、curriculum、logging |
| [inference/stroke_to_gcode.py](inference/stroke_to_gcode.py) | 笔画序列 → P340 gcode |
| [eval/benchmark.py](eval/benchmark.py) | 所有方案在 benchmark 集上的对比表 |
| [fallback/pomo_orderer.py](fallback/pomo_orderer.py) | Phase 4 only：POMO 笔画排序器 |

---

## Verification（如何验证收益）

### 端到端测试流程
1. 跑 `calibration/measure_motion.py`，确认 `motion_params.json` 与真机一致（预测耗时与实测耗时误差 <10%）。
2. 跑 `eval/benchmark.py`，在 50 张 benchmark 图像上跑全部 5 个 variant，**在真机上实测绘画时间**（不能只看模型预测）。

### Variant 对比表（每行一种方案）
1. 原 pipeline（vtracer + 默认顺序）
2. Phase 1：vtracer + LKH-TSP 重排
3. CLIPasso（纯视觉损失）+ LKH-TSP 重排（消融：CLIPasso 笔画本身是否更易排序？）
4. Phase 2：联合自回归
5. Phase 3 fallback（如启用）：CLIPasso + POMO orderer

### 指标
- **绘画时间（秒）**：真机实测。预测 vs 实测误差也是关键指标。
- **视觉质量**：LPIPS 距 Informative Drawing 目标、CLIP image-image 相似度、5 评测人对 50 对图像的主观打分。
- **抬笔空走占比**：总时间中空走时间百分比（论文 headline）。
- **笔画数**：越少越快。

### 期望 headline
Phase 2（或 Phase 4 fallback）的总绘画时间 ≤ 原 pipeline 的 **50%**，且 LPIPS 与原 pipeline 相差 ≤ 10%。

### 诚实报告
失败用例（如 1-stroke 退化输出）必须收录在评测报告里。

---

## 已被刻意排除的方案（附理由）

- **纯扩散模型生成笔画几何**（不是用于排序）：算力贵、推理慢，在线稿场景质量收益不明；留到 v2 再考虑。
- **从头 RL（端到端 REINFORCE）**：视觉损失天然可微，放弃这一信号去做 RL 是浪费；RL 只在 fallback 的离散排序子问题中使用。
- **替换 Informative Drawing**：scope creep；线稿环节工作得好，只动真正能带回报的部分。
- **关节空间时间模型**：P340 在画纸平面上腕部锁定，等效 2-DOF 笛卡儿规划；关节空间会加 ~10× 计算只换 5% 精度提升，不值。
- **变长 polyline / 直线笔画**：cubic Bezier 在参数效率（每条 8 维 vs 长 polyline 的 2N 维）和曲线表达力上更优；直线笔画要 3–5× 笔数才能达到同等质量，反而拖累总耗时。

---

## 推理时间收支表（验证整体方案不会变慢）

| 阶段 | 当前 pipeline | 本方案 |
|---|---|---|
| ASR | ~0.3s | ~0.3s |
| Infinity T2I | ~2s | ~2s |
| Informative Drawing | ~1s | ~1s |
| vtracer | ~0.5s | — |
| SVG→gcode + 排序 | ~0.1s | — |
| **神经笔画生成器（前馈推理）** | — | **~2.5s** |
| 机器人绘画 | **N 分钟** | **N/2 分钟（目标）** |
| **总计** | ~4s + N 分钟 | ~6s + N/2 分钟 |

前段推理增加 ~2 秒，但后段绘画时间减半。N≥1 分钟时净收益巨大（绘画本身就是分钟级），用户感知是"看 AI 生成了张图，然后机器人画得很快"。

**对比若错误地直接套 CLIPasso 推理**：每张图前段会额外加 ~6 分钟优化，总耗时反而增加。这就是为什么必须做 amortized feedforward 而不是 per-image optimization。

---

## 主要风险（按严重程度排序）

1. **L_time 梯度信号过弱，λ 调参塌成纯视觉或单笔输出**。缓解：λ 退火 + 梯度量级监控 + fallback。
2. **方向 Gumbel 学不动**（梯度只来自 L_time）。缓解：方向 logit 单独 optimizer + 更高 LR；监控翻转率。
3. **预测耗时 ≠ 真机耗时**（gcode 解释器、缓冲、抬笔沉降）。缓解：积极标定、必要时加 per-trajectory 学习型偏置。
4. **Autoregressive 推理慢**（256 笔 × 10ms = 2.5s）。可接受（绘画本身要分钟级）；必要时改并行采样 + rejection。
5. **Informative Drawing 风格窄，泛化差**。缓解：多样化 benchmark、augmentation。
6. **DiffVG 在 512×512 + 256 笔下显存爆**。缓解：gradient checkpointing、训练用 256×256。

---

## 可核验的关键参考文献（请自行打开核验）

- CLIPasso：[arxiv.org/abs/2202.05822](https://arxiv.org/abs/2202.05822) / [github.com/yael-vinker/CLIPasso](https://github.com/yael-vinker/CLIPasso)
- Paint Transformer：[arxiv.org/abs/2108.03798](https://arxiv.org/abs/2108.03798)
- Stylized Neural Painter：[arxiv.org/abs/2011.08114](https://arxiv.org/abs/2011.08114)
- Learning to Paint (Huang ICCV 2019)：[arxiv.org/abs/1903.04411](https://arxiv.org/abs/1903.04411)
- Attention Model for TSP (Kool 2019)：[arxiv.org/abs/1803.08475](https://arxiv.org/abs/1803.08475)
- POMO：[arxiv.org/abs/2010.16011](https://arxiv.org/abs/2010.16011)
- DIFUSCO：[arxiv.org/abs/2302.08224](https://arxiv.org/abs/2302.08224)
- DiffVG：[people.csail.mit.edu/tzumao/diffvg/](https://people.csail.mit.edu/tzumao/diffvg/) / [github.com/BachiLi/diffvg](https://github.com/BachiLi/diffvg)
- FRIDA 机器人画家：[arxiv.org/abs/2210.00664](https://arxiv.org/abs/2210.00664)
- LKH-3 求解器：[webhotel4.ruc.dk/~keld/research/LKH-3/](http://webhotel4.ruc.dk/~keld/research/LKH-3/)
- UltraArm P340 产品页：[docs.elephantrobotics.com/docs/ultraArm/1-BriefIntroduction/2-Product/1-UltraArmP340/1-UltraArm_P340.html](https://docs.elephantrobotics.com/docs/ultraArm/1-BriefIntroduction/2-Product/1-UltraArmP340/1-UltraArm_P340.html)
- UltraArm Python API：[docs.elephantrobotics.com/docs/ultraArm/3-HowToUseultraArm/2-SoftwareControl/4-Python/2-PythonAPI.html](https://docs.elephantrobotics.com/docs/ultraArm/3-HowToUseultraArm/2-SoftwareControl/4-Python/2-PythonAPI.html)

**注**：本会话沙箱无外网访问权限，以上 URL 是我基于训练知识给出的常用入口；论文 arxiv ID 与 GitHub 仓库名我有较高把握，但页面具体内容请自行打开核验。如需我从某个站点抓取具体数据，需要在能联网的环境再跑一次。
