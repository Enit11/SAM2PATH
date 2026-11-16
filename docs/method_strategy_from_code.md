# SAM2PATH 肾脏病理半监督分割：已实现策略的论文式 Method 说明

本文档只描述当前代码中已经存在并实际参与训练的机制，不把尚未接入的算法设想写成实验结论。当前正式配置为 `configs/Glomeruli_uni2_spherical_hrrefine.py`，类别编码为 `0=background/non-target`、`1=glomerulus`、`2=vessel`、`3=tubule`。

## 1. 问题定义与总体目标

输入图像记为 `x∈R^{3×1024×1024}`，像素标签为 `y∈{0,1,2,3}^{1024×1024}`。模型输出四通道 logits：

`L=f_θ(x)∈R^{4×1024×1024}`，预测概率为 `p=softmax(L)`，最终标签为 `argmax_c p_c`。

当前方法针对四类互相关联的问题：

1. SAM2 与 UNI2-h 的特征统计和语义粒度不同，直接混合可能造成信息偏置；
2. glomerulus、vessel、tubule 在局部纹理上相似，边界处容易发生类别侵入；
3. 无标注图像不能直接使用 CE/Dice，但教师模型的低置信预测可能传播错误；
4. 连通结构内部应保持语义一致，单纯像素损失对实例级紧凑性约束不足；
5. 64×64 级别的主干 token 上采样到 1024×1024 时，边界、孔洞和细小结构可能被平滑。

总体损失写为：

`L_total = L_sup + λ_mt L_mt + L_proto + L_memory + L_geo`。

其中 `L_sup` 是有标注监督，`L_mt` 是 Mean Teacher 一致性，`L_proto` 是类别 prototype 约束，`L_memory` 是实例记忆约束，`L_geo` 是边界/距离几何监督。各项的实际系数来自配置，而不是所有项都具有相同权重。

## 2. Foundation encoder 与跨编码器适配

### 2.1 针对什么问题

SAM2 Hiera Base+ 提供通用空间视觉特征，UNI2-h 提供病理语义特征。两者原始特征维度、特征范数和空间网格可能不同，直接拼接会使融合层同时承担维度压缩和统计对齐。

### 2.2 已实现策略

SAM2 输出 `F_s∈R^{B×256×64×64}`。UNI2-h 的 Transformer 主体冻结，经过可训练 UNI neck 后输出 `F_u∈R^{B×256×64×64}`。若空间大小不同，代码用双线性插值将 UNI 特征对齐到 SAM2 的 `64×64`：

`F_u' = Interpolate(F_u, size=(64,64))`。

随后进行 concat：

`F_cat=[F_s;F_u']∈R^{B×512×64×64}`，

再经 `fusion_neck` 得到 `F_f∈R^{B×256×64×64}`。

正式配置中 `extra_encoder_neck=True` 且 `extra_encoder_trainable_neck=True`。因此，UNI2-h Transformer 主体冻结，但其 1×1/3×3 投影 neck 可学习，承担从通用病理表征到 SAM2 结构空间的适配。

### 2.3 球面空间的空间自适应残差

为减少特征幅值对比较的影响，球面分支分别投影并归一化：

`z_s = W_s(F_s)/||W_s(F_s)||_2`，`z_u = W_u(F_u')/||W_u(F_u')||_2`。

空间门控图为：

`g=σ(G([z_s;z_u]))∈[0,1]^{B×1×64×64}`。

逐位置球面融合为：

`z=normalize(g·z_s+(1-g)·z_u)`。

该特征经过残差投影 `R(z)`，并加入 concat 主路径：

`F_out=F_f+αR(z)`，其中正式配置 `α=0.1`。

最后投影为零初始化，所以初始时 `F_out≈F_f`，训练过程中才逐渐学习球面语义修正。该模块是**空间自适应球面残差融合**，不是 cross-attention，也不是跨 batch 的流式特征融合。

代码位置：`network/sam_network.py` 中 `SphericalGatedFusion` 与 `PromptSAMLateFusionSAM2NoKAN.forward()`。

## 3. Confidence Filtering：可靠伪标签筛选

### 3.1 针对什么问题

Mean Teacher 的教师预测在边界、相邻结构和低对比区域可能不可靠。若所有伪标签都用于一致性、prototype 或 memory 更新，错误会被学生模型和记忆库反复放大。

### 3.2 一致性监督中的筛选

教师 logits 为 `L_t`，教师概率为：

`p_t=softmax(L_t)`。

像素置信度定义为最大类别概率：

`q(i)=max_c p_t(i,c)`。

仅保留 `q(i)≥τ_mt` 的像素：

`M_mt(i)=1[q(i)≥τ_mt]`。

一致性损失实际使用 MSE 时为：

`L_mt = mean_{i:M_mt(i)=1} ||p_s(i)-p_t(i)||_2^2`。

正式配置 `τ_mt=0.8`，并通过 `mt_coef()` 在前 10 个 epoch 逐步增加一致性权重。代码位置：`pl_module_sam_seg.py::consistency_loss`。

### 3.3 Prototype/memory 中的筛选

无标注伪标签为：

`ŷ_t(i)=argmax_c p_t(i,c)`。

代码额外要求前景类别：

`M_proto(i)=1[q(i)≥0.90 ∧ ŷ_t(i)>0]`。

低置信像素被置为背景/无效，不会更新 prototype 或 instance memory，也不会用于无标注 prototype/memory loss。Prototype 分支第 5 epoch 后激活，memory 第 10 epoch 后激活。

### 3.4 该策略的边界与局限

当前是**最大概率阈值过滤**，没有实现 top-1/top-2 margin、边界降权或校准后的类别阈值。因此它能过滤明显不确定区域，但不能保证高置信边界一定正确。`teacher_model` 使用 `torch.no_grad()`，参数由 batch 结束时 EMA 更新：

`θ_t←βθ_t+(1-β)θ_s`，正式配置 `β=0.999`。

## 4. Prototype Matching：类别级 Foundation Prototype

### 4.1 针对什么问题

像素级监督容易受局部纹理、染色变化和边界混合影响。需要一个跨图像、跨实例的类别语义中心，使同类组织在结构特征空间中保持一致。

### 4.2 Prototype 构建

对 GT 或高置信教师伪标签进行连通域提取。对类别 `c∈{1,2,3}` 的区域 `R_c`，从教师结构特征 `F_t` 提取区域平均 token：

`v_c = mean_{i∈R_c} F_t(i)`，`z_c=v_c/||v_c||_2`。

Foundation prototype 使用动量更新：

`P_c^{(t)}=normalize(mP_c^{(t-1)}+(1-m)z_c^{(t)})`，正式配置 `m=0.99`。

背景不建立 prototype；prototype 只覆盖三个前景类别。代码位置：`update_foundation_prototypes()`。

### 4.3 Prototype matching

学生区域的像素特征先归一化，再求区域 token：

`u_R=normalize(mean_{i∈R} normalize(F_s(i)))`。

对已准备好的类别 prototype 计算温度缩放余弦 logits：

`a_c(R)=u_R^T P_c/τ`，`τ=0.1`。

区域 prototype 分类损失为：

`L_region=-log exp(a_y(R))/Σ_{c∈C_ready}exp(a_c(R))`。

这里的 `y` 来自 GT 或筛选后的教师伪标签区域。该损失要求一个区域的整体 token 更接近其真实类别中心，而不是只依赖单个像素。

### 4.4 类内紧凑性

为减少同一连通区域内部的语义漂移，代码计算：

`L_intra(R)=mean_{i∈R}[1-cos(normalize(F_s(i)),u_R)]`。

它鼓励一个 glomerulus、vessel 或 tubule 连通区域的特征围绕同一语义中心聚集。

## 5. Semantic Purification：区域筛选、纯度与实例记忆

### 5.1 这里的“purification”实际指什么

当前代码没有一个名为 `SemanticPurification` 的独立神经网络模块。论文中更准确的表述是：通过**连通域筛选、面积阈值、类别纯度阈值、置信度阈值、类内紧凑损失和实例 memory 对比匹配**实现语义净化。

### 5.2 连通域与纯度筛选

对每个前景类别的 mask 做 8-邻域 connected components。仅保留面积不小于 `min_region_pixels` 的区域，并计算：

`purity(R)=#\{i∈R:label(i)=c\}/|R|`。

只有 `purity(R)≥0.70` 的区域进入 prototype region loss；实例 memory 使用更严格的 `min_region_pixels=32` 和 `min_region_purity=0.90`。每图最多保留 16 个区域，每区域最多采样 128 个像素，控制计算和异常区域影响。

这一步针对的是：粘连、碎片、小噪声和跨类混合区域不应成为稳定类别语义的来源。

### 5.3 Instance memory

每类维护 16 个 FIFO 槽位。可靠连通域产生单位范数实例 token `t_R`，写入类别 memory；若槽位已有 token，则使用 `0.9` 动量更新：

`M_{c,k}←normalize(0.9M_{c,k}+0.1t_R)`。

学生实例 token 与同类 memory 的最大相似度用于特征匹配：

`L_feat(R)=1-max_k cos(t_R,M_{y_R,k})`。

同时将所有 memory token 组成对比分类 bank：

`L_mem(R)=-log [Σ_{k:y_k=y_R}exp(t_R^TM_k/τ)]/[Σ_j exp(t_R^TM_j/τ)]`。

因此，实例 memory 既保存历史可靠结构，又要求当前学生实例靠近同类历史表示、远离异类表示。它是当前代码中最明确的跨 batch 递推语义机制。

### 5.4 类间 separation

当 `sep_coef>0` 时，对类别 `c` 的学生 token `u_c` 和其他 prototype `P_{c'}` 添加 hinge 约束：

`L_sep=mean_{c'≠c} max(0,cos(u_c,P_{c'})-δ)`。

正式配置 `δ=0.2`、`sep_coef=0.01`。该项针对的是 glomerulus/vessel/tubule 特征过近导致的类别侵入，但权重较小，避免把视觉相似类别强行推得过远。

## 6. 边界与高分辨率恢复

### 6.1 针对什么问题

主干 token 为 `64×64`，SAM decoder 产生约 `256×256` logits 后再上采样到 `1024×1024`。单纯区域损失可能允许边界膨胀、收缩、粘连或出现孔洞。

### 6.2 几何监督

代码根据 GT 构造边界带 `B` 和三个前景类别的 signed distance map `D_c`。边界由标签邻域类别变化检测后膨胀得到；距离图对每个前景类别分别计算并截断到 `[-1,1]`。

边界头使用 BCE-with-logits：

`L_boundary=BCEWithLogits(b_hat,B)`。

距离头使用 Smooth-L1：

`L_distance=SmoothL1(D_hat,D)`。

边界困难像素进一步按预测熵或 top-1/top-2 logit margin 选择：

`H(i)=B(i)∧[Entropy(p_s(i))≥η ∨ margin(L_s(i))≤γ]`，

并对这些位置计算 hard CE。代码位置：`geometry_targets()`、`supervised_geometry_losses()`、`hard_boundary_losses()`。

### 6.3 高分辨率 logits residual refinement

最高分辨率 SAM2 FPN 特征通常为 `128×128`，被插值到 decoder logits 的 `256×256`，与四类 logits 一起输入 refiner：

`L_refined=L_low+0.1H(F_FPN↑,L_low)`。

最后再插值到 `1024×1024`。refiner 最后一层零初始化，所以初始输出不改变原 decoder；训练后可以直接修正最终类别竞争 logits。它不是独立 boundary mask，而是作用于最终分割输出的边界恢复分支。

## 7. 监督、半监督与梯度路径

有标注样本使用 `SAMLoss`：

`L_sup=0.125L_focal+0.875L_dice+0.25L_CE+0.08L_RC`，正式配置中 IoU loss 系数为 0。Dice、Focal、CE 均接收 logits/probabilities 的规范实现；预测标签由四通道 logits 的 argmax 得到。

无标注样本采用 weak/strong 两个增强：teacher 处理 weak view，student 处理 strong view。teacher 使用 `no_grad`，student 的一致性和 prototype/memory loss 产生反向梯度。Prototype、memory 和 prototype separation 的 buffer 更新均在 `torch.no_grad()` 下手工完成，不通过 optimizer 反向更新。

可训练模块包括：UNI neck、fusion neck、球面投影与 gate、structure adapter、SAM mask decoder、geometry heads 和 high-resolution refiner。SAM2 主体、UNI2-h Transformer 主体以及 teacher 全部冻结。

## 8. 方法边界与不能夸大的地方

1. 当前 confidence filtering 只有最大概率阈值，没有 margin 或显式边界降权；
2. `eroded_core()` 已定义，但当前 prototype/memory 更新路径没有调用它，因此不能宣称已经实现“侵蚀得到可靠核心”；
3. Semantic Purification 是多个筛选和记忆损失的组合，不是独立的 purification decoder；
4. geometry boundary head 提供辅助边界监督，但真正影响最终 mask 的是 high-resolution logits refiner；
5. 球面融合是空间门控残差，不是 cross-attention，也不是跨样本递推融合；跨 batch 递推只发生在 EMA teacher、foundation prototype 和 instance memory。

## 9. 方法贡献的谨慎表述

在不预设实验结果的情况下，可以将当前方法概括为：

> We develop a semi-supervised SAM2–UNI2 pathology segmentation framework that combines trainable cross-encoder adaptation, spatially gated spherical residual fusion, confidence-filtered class prototypes, connected-component-based semantic purification, online instance memory, and high-resolution logit refinement. Reliable teacher regions and ground-truth regions provide normalized semantic tokens, while prototype and memory matching regularize class compactness and inter-class separation. A zero-initialized high-resolution residual decoder then recovers local boundary details without changing the initial behavior of the pretrained segmentation model.

这段描述对应当前代码；是否能改善 Dice、Boundary F1、HD95 或类别混淆，仍需由固定数据划分和统一评价协议验证。
