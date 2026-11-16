# SAM2–UNI2 半监督肾脏病理分割：方法审计、结构设计与中文 Method 初稿

**证据范围。** 本文基于 `configs/Glomeruli_uni2_spherical_hrrefine.py`、`network/sam_network.py`、`network/get_network.py`、`pl_module_sam_seg.py` 与 `losses.py`。所有“已实现”仅指当前训练路径实际调用的代码；不以模块名称替代技术事实。

---

# 阶段 A：方法审计

## A.1 任务、数据流与模块依赖

设有标注集为 `D_l={(x_i,y_i)}_{i=1}^{N_l}`，无标注集为 `D_u={x_j}_{j=1}^{N_u}`。`x∈R^{3×1024×1024}`，`y∈{0,1,2,3}^{1024×1024}`；类别 `0,1,2,3` 分别是 background、glomerulus、vessel、tubule。学生输出四通道 logits `L_s∈R^{B×4×1024×1024}`，`p_s=softmax(L_s)`；教师具有相同架构、无梯度。

```text
Labeled x_l ── Student ──> L_l, F_l, geometry_l ──> L_sup + L_proto/l + L_memory/l + L_geo/l
                       │
Unlabeled x_s ─ Student ─> L_s, F_s, geometry_s ──> L_MT + L_proto/u + L_memory/u + L_geo/u
Unlabeled x_w ─ Teacher ─> L_t, F_t, geometry_t ──> confidence / pseudo labels / EMA states

Student encoder path:
SAM2 Hiera (frozen) → [B,256,64,64]
UNI2-h Transformer (frozen) → trainable UNI neck → [B,256,64,64]
→ concat [B,512,64,64] → fusion neck → spherical residual → adapter → SAM decoder
→ [B,4,256,256] → high-resolution residual refiner → [B,4,256,256]
→ interpolation → [B,4,1024,1024].
```

SAM2 的最高分辨率 FPN 特征通常是 `[B,256,128,128]`，在 refiner 中上采样至 `256×256`。SAM2 encoder 在 `torch.no_grad()` 中运行；UNI2-h wrapper 内部冻结 Transformer，但 neck 位于 no-grad 块之外且 `requires_grad=True`。教师由 student deepcopy 初始化并在每个 batch 后 EMA 更新。

## A.2 符号表

| 符号 | 含义 |
|---|---|
| `C={1,2,3}` | 三个前景类别；背景 0 不进入 prototype/memory |
| `F^S,F^U` | SAM2、UNI neck 对齐后的 `256×64×64` 特征 |
| `F` | adapter 前的融合结构特征；`A` 为 adapter 输出 |
| `L,p` | 四类 segmentation logits 与 softmax 概率 |
| `T` / `S` | EMA teacher / 可训练 student |
| `q(i)` | 教师在像素 `i` 的最大类别概率 |
| `R` | 某个前景类别的 8-连通区域 |
| `z_R,u_R` | 教师/学生区域 token（L2 单位范数） |
| `P_c` | 类别 `c` 的 foundation prototype |
| `M_{c,k}` | 类别 `c` 的第 `k` 个 FIFO memory token |
| `τ_p,τ_m` | prototype/memory 温度，均为 0.1 |
| `β,m,ρ` | EMA=0.999、prototype momentum=0.99、memory momentum=0.9 |

## A.3 实际进入优化目标的损失

| 项 | 代码条件 | 系数/启用时刻 | 对 student 的梯度 |
|---|---|---|---|
| `L_sup` | 有标注 batch | Focal .125、Dice .875、CE .25、RC .08 | 是 |
| `L_MT` | teacher confidence ≥ .8 | 系数从 epoch 1 至 10 warm-up 到 1 | 是 |
| `L_region,L_intra,L_sep` | epoch ≥5 且至少两个 prototype ready | .05、.02、.01；无标注部分额外 ×.5 | 是 |
| `L_mem-logit,L_mem-feat` | epoch ≥10 且 memory ready | 各 .05；无标注部分 ×.5 | 是 |
| `L_boundary,L_distance` | epoch ≥10 | .20、.10 | 是 |
| `L_hard` | epoch ≥10 且存在 hard GT boundary pixels | .10 | 是 |
| `L_geo-u` | epoch ≥10 且 teacher geometry 有有效高置信像素 | .05 | 是，作用于 student boundary head |

实际总式为：

`L = L_sup + λ_mt(e)L_MT + .05L_region + .02L_intra + .01L_sep + .05L_mem-logit + .05L_mem-feat + .20L_boundary + .10L_distance + .10L_hard + .05L_geo-u`。

这里 `.05` 的两个 memory 项来自 `memory.consistency_coef` 被同时作为 `logits_coef` 和 `feature_coef` 的回退值。所有 prototype/memory 的 buffer 更新均在 `no_grad` 下，不由 AdamW 直接优化。

## A.4 实现一致性与待代码核验

1. **类别加权错误**：`vessel_boundary_weight` 实际通过 `labels==3` 施加；根据任务编码，class 3 是 tubule。因此当前是 *tubule boundary weight*，不应在论文中称为 vessel weighting，除非修复为 `labels==2`。
2. **memory 独立可靠性配置未被使用**：`memory.min_region_pixels`、`memory.min_region_purity`、`memory.conf_threshold` 被配置但 memory 更新实际复用 `proto_min_region_pixels` 和 `teacher_valid`（由 `proto_conf_threshold` 建立）。因此当前不能声称 memory 使用 32 像素、0.90 purity 的独立筛选。
3. **`broadcast_coef` 未使用**：配置有该字段，但训练模块未读入，不能纳入公式或贡献。
4. **“purity” 对已二值类别连通域近似恒为 1**：`iter_regions` 先定义 `cls_mask=(label==c)&valid` 再连通域，之后再计算 `label==c` 的比例；这并不能检测同一 mask 中的跨类混合。它仍可描述为最小区域和 valid-mask 筛选，但不宜夸大为强混合区域纯化。
5. **侵蚀核心未接入**：`eroded_core()` 已定义但没有被 prototype/memory 或 boundary 路径调用；不可声称 reliable eroded core、boundary ownership 或 core-to-boundary assignment。
6. **Spherical 分支的“球面性”有限**：两路投影和加权结果使用 L2 归一化，故 token 位于单位球面；但融合用的是归一化的欧氏加权和，不是 geodesic exponential/log map、Fréchet mean 或黎曼递推。准确名称应是 *unit-hypersphere normalized gated fusion*。
7. **高分辨率来源待形状运行核验**：代码选择 `backbone_fpn` 中面积最大的尺度，而非硬编码 128×128；“通常 128×128”应写作实现设置而非普遍理论保证。

---

# 阶段 B：Method 结构设计

建议将原先五个名称收敛为四个实质性小节，避免把可组合的正则项误包装为过多独立贡献：

1. **3.1 Problem Formulation and Framework**：定义数据、类别、student/teacher、端到端数据流；
2. **3.2 Cross-encoder Spatial Adaptation and Unit-Hypersphere Gated Residual Fusion**：把 UNI neck、64×64 对齐、concat 主径和球面残差写在一起，因为它们共同产生 decoder 特征；
3. **3.3 Confidence-filtered Region-level Semantic Regularization**：合并 Mean Teacher、prototype、semantic purification 和 instance memory。理由是它们都以 teacher confidence 和 connected region 为依赖，分开写会重复解释伪标签筛选；
4. **3.4 Geometry-aware Supervision and High-resolution Logit Refinement**：边界/距离/难边界与 refiner 都服务最终轮廓，应成一条解码修复线；
5. **3.5 Joint Objective and Optimization**：统一损失、warm-up、梯度与状态更新。

主线不是“堆叠五个模块”，而是：**用双基础编码器获得互补表征；以可靠 teacher 区域建立跨图像类别/实例参照；最后将区域语义和几何监督共同反馈给能直接改变 logits 的解码路径。** Prototype 和 memory 并不替代一致性：前者提供类别/实例级的语义坐标，后者提供时间累积的实例参照；geometry 也不替代 refiner：前者提供训练信号，后者将高分辨率信息直接注入最终 logits。

## 可用于绘制框架图的连接说明

```text
Input image → frozen SAM2 Hiera ─┐
                                 ├→ 64×64 alignment → concat neck → spherical gated residual → dual-branch adapter → SAM decoder
Input image → frozen UNI2-h → trainable neck ─┘                                                    │
SAM2 high-res FPN ────────────────────────────────────────────────────────────────────────────────┼→ logit refiner → 4-class logits
                                                                                                     │
labeled GT ───────────────────────────────────────→ supervised/geometry targets ────────────────────┤
unlabeled weak → EMA teacher → confidence mask / pseudo regions → prototype + memory states ────────┤
unlabeled strong → student ─────────────────────────→ consistency/prototype/memory losses ──────────┘
```

---

# 阶段 C：论文级中文 Method 初稿

## 3.1 问题定义与总体框架

我们研究有限标注条件下的肾脏病理语义分割。训练数据由有标注集 `D_l={(x_i,y_i)}_{i=1}^{N_l}` 和无标注集 `D_u={x_j}_{j=1}^{N_u}` 构成，其中 `x` 是 1024×1024 RGB 图像，`y` 取自类别空间 `{0,1,2,3}`，分别对应 background、glomerulus、renal vessel 和 renal tubule。学生网络 `f_θ` 输出四类 logits `L_s=f_θ(x)`；像素概率为 `p_s=softmax(L_s)`。

该框架使用冻结的 SAM2 Hiera Base+ 提取通用空间特征，并以冻结 UNI2-h Transformer 与可训练 neck 提供病理语义适配。二者先在 64×64 空间网格上对齐，再经 concat 主径和单位超球面门控残差融合。其输出送入共享 adapter 与 SAM mask decoder。为缓解低分辨率解码带来的轮廓退化，来自 SAM2 FPN 的最高分辨率特征进一步与低分辨率 logits 结合，以残差方式细化最终预测。

对于无标注图像，EMA teacher 在弱增强视图上产生概率、伪标签及区域特征；student 在强增强视图上接受一致性、prototype 和 memory 约束。有标注图像同时提供区域分割和几何监督。该安排使区域级语义正则和边界级训练信号共享同一 student feature/logit 路径，而非彼此独立。

## 3.2 跨编码器空间适配与单位超球面门控残差融合

SAM2 生成 `F^S∈R^{B×256×64×64}`。UNI2-h Transformer 主体保持冻结，trainable neck 将其输出投影到 `F^U∈R^{B×256×64×64}`。当空间维度不一致时，以双线性插值将 UNI 特征映射到 SAM2 网格。拼接主径定义为：

`F^c=[F^S;F^U]`，`F^f=N_f(F^c)`，

其中 `N_f` 是输出 256 通道的卷积 fusion neck。

为削弱不同编码器特征范数对融合的影响，分别计算 `z^S=Norm(W_SF^S)` 与 `z^U=Norm(W_UF^U)`，其中 `Norm(v)=v/(||v||_2+ε)` 沿通道维归一化。门控图为 `g=σ(G([z^S;z^U]))`，其维度为 `B×1×64×64`。在每个位置形成：

`z=Norm(g⊙z^S+(1-g)⊙z^U)`。

最终融合结果是 `F=F^f+αR(z)`，其中 `α=0.1`。`R` 的末端投影零初始化，使新增支路在初始化时不改变 concat 基线输出。该设计并非 cross-attention：没有 query-key-value 或跨特征 token 注意力矩阵；也不是固定权重/通道注意力，因为 `g` 是逐空间位置的单通道权重。称其为“单位超球面归一化的门控残差融合”比泛称“黎曼球面融合”更严格。

## 3.3 置信度过滤的教师—学生区域语义正则

教师参数在每个 batch 后按 `θ_t←βθ_t+(1-β)θ_s` 更新，`β=0.999`。给定教师弱增强 logits `L_t`，令 `p_t=softmax(L_t)`、`q(i)=max_c p_t(i,c)`。一致性有效掩码为 `M_mt(i)=1[q(i)≥0.8]`，当前 MSE 一致性为：

`L_mt=1/|M_mt| Σ_{i:M_mt(i)=1} ||p_s(i)-p_t(i)||_2^2`。

对 prototype 和 memory，更严格的候选掩码为 `M_p(i)=1[q(i)≥0.90 ∧ argmax_cp_t(i,c)>0]`。因此，半监督策略不是把所有教师 argmax 当作标签，而是仅允许高置信前景区域更新状态并产生区域约束。它目前未利用 top-1/top-2 margin、拓扑可信度或边界专属阈值；这些不应被描述为已实现。

对于任意 GT 或有效伪标签的类别 `c` 连通域 `R`，教师区域 token 定义为 `z_R=Norm(mean_{i∈R}F_t(i))`。类别 prototype 按 `P_c←Norm(mP_c+(1-m)z_R)` 更新，`m=0.99`。student 区域 token `u_R=Norm(mean_{i∈R}Norm(F_s(i)))` 与可用 prototype 集合 `C_ready` 进行温度缩放余弦匹配：

`L_region(R)=-log exp(u_R^TP_y/τ_p) / Σ_{c∈C_ready}exp(u_R^TP_c/τ_p)`，`τ_p=0.1`。

此外，`L_intra(R)=mean_{i∈R}[1-Norm(F_s(i))^Tu_R]` 约束区域内部特征紧凑。不同类别的 prototype 提供语义比较坐标，使该项补充像素级 Dice/CE 与 pixelwise consistency，而非重复它们。

## 3.4 基于可靠区域筛选的语义净化与实例记忆

本方法没有独立的 Semantic Purification network。这里的语义净化指 confidence filtering、前景连通域选择、最小面积约束、prototype matching、类内紧凑和实例记忆约束的联合过程。实际区域筛选采用前景 mask 的 8-连通组件，prototype 路径保留面积不少于 6 的区域，并限制每图最多 16 个区域、每区域最多采样 128 个像素。当前 purity 计算应视作代码层面的有效区域检查，而不是强有力的跨类混合检测（见 A.4）。

为维持跨 batch 的实例级参照，每个前景类别维护容量 16 的 FIFO memory。对于教师实例 token `z_R`，若当前位置已有 token，则执行 `M_{c,k}←Norm(ρM_{c,k}+(1-ρ)z_R)`，其中 `ρ=0.9`；随后循环指针移动到下一槽位。给定 student token `u_R`，同类记忆特征损失为：

`L_mem-feat(R)=1-max_k u_R^TM_{y,k}`。

将所有 ready memory token 拼接为 bank，类别对比损失为：

`L_mem-logit(R)=-log [Σ_{k:c_k=y} exp(u_R^TM_k/τ_m)]/[Σ_j exp(u_R^TM_j/τ_m)]`，其中 `τ_m=0.1`。

原型和 memory 分别提供类别级稳定中心和实例级历史多样性；二者可能存在功能重叠，因而必须通过 “prototype only / memory only / both” 消融判断 memory 是否具有增益。类间 hinge 分离 `mean_{c'≠c}max(0,u_R^TP_{c'}-0.2)` 以 .01 权重加入，提供温和的跨类排斥，而不保证类别间具有严格几何分离。

## 3.5 几何感知监督与高分辨率 logit 细化

区域 Dice/CE 对少量边界像素的响应有限，因此从 GT 构建膨胀语义边界带 `B` 与每个前景类的截断 signed distance map `D∈R^{3×H×W}`。边界头使用加权 BCE-with-logits，距离头使用 Smooth-L1：

`L_b=WeightedBCE(b_hat,B)`，`L_d=SmoothL1(d_hat,D)`。

在 GT 边界带内，以 `H(i)=B(i)∧[Entropy(p_s(i))≥η ∨ (L_{(1)}(i)-L_{(2)}(i))≤γ]` 定义困难位置，并在该集合计算额外 CE。教师与学生 boundary head 还在教师置信度至少 .8 的位置进行 MSE 一致性。上述损失提供边界存在、到类别区域的相对距离和困难类别竞争的训练信号；它们本身不直接替换 segmentation logits。

为直接修正最终预测，令 `F_hr` 是 SAM2 最大空间尺度的 FPN 特征。先将其插值到 `L_low∈R^{B×4×256×256}` 的空间大小，随后：

`L_ref=L_low+0.1H_r(F_hr↑,L_low)`。

refiner 最终卷积为零初始化，因此初始 `L_ref=L_low`，随后从所有 segmentation 和 geometry 相关梯度学习 residual correction。该设计关于改善粘连、边界偏移和细小结构的作用是可检验假设，而不是未经验证的结论。

## 3.6 联合优化

训练总目标按实际代码写为：

`L_total=L_sup+λ_mt(e)L_mt+λ_rL_region+λ_iL_intra+λ_sL_sep+λ_mlL_mem-logit+λ_mfL_mem-feat+λ_bL_b+λ_dL_d+λ_hL_hard+λ_uL_geo-u`。

其中 `λ_r=.05,λ_i=.02,λ_s=.01,λ_ml=λ_mf=.05,λ_b=.20,λ_d=.10,λ_h=.10,λ_u=.05`；prototype 从 epoch 5、memory 与 geometry 从 epoch 10 启动，`λ_mt(e)` 在前 10 epoch warm-up。监督项是 `L_sup=.125L_focal+.875L_dice+.25L_CE+.08L_RC`，IoU 项系数为零。

梯度仅更新 student 中的 UNI neck、fusion neck、球面门控/投影、adapter、mask decoder、geometry heads 和 refiner。SAM2、UNI2-h Transformer 主体及 teacher 不更新；prototype、memory 是 `no_grad` buffer 状态。这一区分避免把 EMA 和 memory 误称为反向传播学习层。

## 严格审稿风险与实验建议

1. **“球面融合”是否只是普通归一化门控？** 需要 concat-only、normalized fixed-average、spatial gate without normalization、full fusion 四组；报告每类 Dice、Boundary F1、HD95 和 gate 分布，而不仅是总 Dice。
2. **prototype/memory 是否真有互补性，还是伪标签噪声累积？** 做 supervised-only、MT、MT+prototype、MT+memory、MT+both；同时报告接受伪标签比例、prototype cosine、memory 类内/类间相似度及失败可视化。
3. **边界改进是否来自 refiner 而非 auxiliary geometry，且类别加权是否正确？** 做 geometry-only、refiner-only、both；修复/比较 `labels==2` 与当前 `labels==3` 的权重行为，并报告 Boundary F1@1/3、class-wise HD95、混淆矩阵、孔洞/连通性统计。
