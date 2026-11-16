# SAM2PATH：球面特征融合与类别语义建模实验方案

## 1. 目标与范围

本方案研究 SAM2.1 Hiera Base+ 与 UNI2-h 特征融合后，是否能通过单位超球面上的度量学习改善肾脏病理语义分割中的类别混淆与边界侵入。

任务包含四个互斥类别：

| 标签 ID | 类别 |
|---:|---|
| 0 | background / non-target |
| 1 | glomerulus |
| 2 | renal vessel |
| 3 | renal tubule |

本方案只讨论以下问题：

- 肾小球、血管、小管在邻接区域的类别混淆；
- 一类组织侵入另一类组织的局部边界错误；
- Foundation Prototype 与 Instance Memory 的类别语义稳定性；
- 球面特征融合是否优于当前 `Concat → 1×1 Conv → 3×3 Conv` 融合。

本方案不将球面表示当作高分辨率边界恢复的替代方案。细血管丢失、孔洞和亚像素轮廓误差仍需要由已实现的高分辨率 refinement 分支单独消融。

## 2. 当前基线

当前训练配置使用：

```text
Frozen SAM2.1 Hiera Base+ encoder
Frozen UNI2-h encoder
SAM2/UNI channel concatenation
Fusion neck: Conv1×1 + LayerNorm + Conv3×3 + LayerNorm
SharedBackboneDualBranchAdapter
SAM mask decoder
Mean Teacher + Foundation Prototype + Instance Memory
```

当前实际融合张量为：

```text
SAM2 feature:       B×256×64×64
UNI2-h neck feature:B×256×64×64
Concatenation:      B×512×64×64
Fusion neck output: B×256×64×64
```

基线监督目标为：

\[
\mathcal L_{sup}=0.125\mathcal L_{focal}+0.875\mathcal L_{dice}+0.25\mathcal L_{CE}+0.08\mathcal L_{RC}.
\]

半监督与结构约束采用：

\[
\mathcal L_{total}=\mathcal L_{sup}+\lambda_{mt}\mathcal L_{mt}+\mathcal L_{proto}+\mathcal L_{memory}+\mathcal L_{geometry}.
\]

其中 prototype 和 memory 已经对 token 使用 `F.normalize`，因此它们在当前实现中已部分采用余弦空间。球面实验的核心是将归一化、温度、类间角度分离和融合方式系统化，而不是把已有归一化重复命名为新方法。

## 3. 研究假设

### H1：球面度量稳定类别语义

SAM2 与 UNI2-h 的特征幅值、统计分布和语义粒度不同。直接拼接后由卷积混合时，特征范数可能隐式影响融合和分类。若将实例 token、prototype 和 memory token 投影到单位超球面，类别比较将主要依赖角度：

\[
z=\frac{f}{\lVert f\rVert_2+\epsilon}, \qquad z\in\mathbb S^{d-1}.
\]

预期收益是降低 glomerulus、vessel、tubule 的跨类 token 混叠，而不是直接增加空间分辨率。

### H2：球面原型提升相邻类别排斥

令类别原型为 \(P_c\in\mathbb S^{d-1}\)，实例 token 为 \(z_k\in\mathbb S^{d-1}\)。采用余弦分类：

\[
\mathcal L_{proto}=-\log\frac{\exp(z_k^\top P_{y_k}/\tau)}{\sum_{c\in\mathcal C_{ready}}\exp(z_k^\top P_c/\tau)}.
\]

若类别原型角度更分离，则靠近结构边界的错误类别侵入应下降。

### H3：球面门控融合可保留互补信息

当前 concat fusion 没有显式控制某一位置更应相信 SAM2 的局部结构信息还是 UNI 的病理语义信息。可定义：

\[
z_{sam}=\operatorname{Norm}(W_{sam}F_{sam}),
\]

\[
z_{uni}=\operatorname{Norm}(W_{uni}F_{uni}),
\]

\[
\alpha(u)=\sigma(g([F_{sam}(u)\Vert F_{uni}(u)])),
\]

\[
z_{fuse}(u)=\operatorname{Norm}(\alpha(u)z_{sam}(u)+(1-\alpha(u))z_{uni}(u)).
\]

其中 \(\alpha(u)\) 是空间位置 \(u\) 的可学习门控权重。

## 4. 分阶段实验原则

每一次实验只修改一个独立因素。所有实验固定：

- patient-level train/val/test 划分；
- 10% labeled 与 unlabeled 数据池；
- seed；
- batch size、optimizer、learning rate、epoch budget；
- 输入 1024×1024；
- 四类 logits 的 argmax 主评价协议；
- 统一评估脚本 `evaluate_segmentation.py`；
- checkpoint 初始化策略和数据增强策略。

每个短程筛选实验建议至少运行 3 个随机种子；若 GPU 资源不足，先采用同一 seed 的短程筛选，仅将有效候选扩展到 3 seeds 和完整训练。

## 5. 实验组设计

### 5.1 Stage S：球面 prototype / memory 度量

这是风险最低的第一阶段。保持当前 `fusion_neck` 和 decoder 完全不变，只在 prototype/memory token 路径上引入显式球面度量配置与日志。

| ID | 融合 | Prototype / Memory | 目的 |
|---|---|---|---|
| S0 | 当前 concat fusion | 当前实现 | 基线 |
| S1 | 当前 concat fusion | 显式单位球面 token、prototype、memory；余弦温度固定 | 确认归一化/温度本身的贡献 |
| S2 | 当前 concat fusion | S1 + 类间 angular margin 或 prototype separation | 检查类别排斥是否降低跨类侵入 |

S1 的关键要求：必须先确认当前 `F.normalize` 的实际调用点和温度设置。若 S1 与 S0 数学上等价，S1 不应作为独立方法结果，而应只用于补充日志和数值稳定性检查。

S2 可采用原型间的铰链式角度分离：

\[
\mathcal L_{sep}=\frac{1}{|\mathcal P|(|\mathcal P|-1)}\sum_{c\neq c'}\max(0,P_c^\top P_{c'}-m).
\]

其中 \(m\) 是允许的最大跨类余弦相似度。该实验必须单独设置较小系数，避免强行把视觉相近类别推得过远。

### 5.2 Stage F：球面融合

在 S 阶段确定 prototype/memory 方案后，再改变融合方式。

| ID | SAM2/UNI 融合 | Prototype / Memory | 目的 |
|---|---|---|---|
| F0 | 当前 concat + conv neck | Stage S 最优配置 | 融合基线 |
| F1 | 投影后球面平均融合 | Stage S 最优配置 | 检查简单球面融合是否有效 |
| F2 | 空间门控球面融合 | Stage S 最优配置 | 检查位置自适应融合是否有效 |
| F3 | F2 + 高分辨率 refinement | Stage S 最优配置 | 只在 F2 有效后评估语义与边界互补性 |

F1 使用固定 \(\alpha=0.5\)：

\[
z_{fuse}=\operatorname{Norm}(0.5z_{sam}+0.5z_{uni}).
\]

F2 使用可学习门控 \(\alpha(u)\)。为避免新融合模块在训练初期破坏旧特征，建议采用残差式输出：

\[
F_{out}=F_{base}+\gamma W_o z_{fuse},
\]

其中 \(\gamma\) 初始为 0 或较小值，且输出投影层可零初始化。

### 5.3 Stage B：边界问题联合验证

只有当 F 阶段证明球面融合改善类别语义时，才与已实现的高分辨率 refinement 组合。

| ID | 球面模块 | 高分辨率 refinement | 目的 |
|---|---|---|---|
| B0 | 无 | 关闭 | 当前基线 |
| B1 | F 阶段最优 | 关闭 | 仅验证类别语义改进 |
| B2 | 无 | 开启 | 仅验证高分辨率边界细化 |
| B3 | F 阶段最优 | 开启 | 检查互补性 |

不要在 B0→B3 之间同时改变 distance head、RC、Mean Teacher consistency 或边界损失权重。

## 6. 评价指标与主要判据

主评价使用 4 类 logits 的 argmax，标签 ID 为 \(0,1,2,3\)。核心指标为：

| 类型 | 指标 | 目的 |
|---|---|---|
| 区域 | Dice-FG-Macro | 三类前景总体分割质量 |
| 区域 | IoU-FG-Macro | 对假阳性/假阴性更敏感 |
| 分类 | Dice/IoU per class | 定位 glomerulus、vessel、tubule 的获益差异 |
| 边界 | Boundary F1@1 px | 严格边界定位 |
| 边界 | Boundary F1@3 px | 容差下的边界一致性 |
| 边界 | class-wise HD95 | 检测极端边界偏移 |
| 混淆 | 4×4 confusion matrix | 量化类别侵入方向 |
| 面积 | prediction/GT area ratio | 检查膨胀或收缩 |

对于球面模块，额外保存：

- 类别原型之间的余弦相似度矩阵；
- 实例 token 到正确/错误 prototype 的相似度差；
- 边界带与内部区域的 prototype margin；
- memory bank 中每类 token 的类内相似度和跨类相似度；
- 预测类别混淆矩阵中 `glomerulus↔vessel`、`vessel↔tubule`、`glomerulus↔tubule` 的非对角元素。

主要成功判据：

1. `Dice_FG_Macro` 不下降，或下降不超过预先声明的容忍范围；
2. `Boundary F1@1px` 与 `Boundary F1@3px` 均改善；
3. 至少一个主要混淆对的互相误分类减少；
4. HD95 不恶化；
5. 三个随机种子上的趋势一致，而非仅单次波动。

## 7. 训练与 checkpoint 策略

### 7.1 当前距离头修正

当前 distance head 已从 2 通道修正为 3 通道，对应三个前景类别。旧 checkpoint 的 `distance_head` 输出层形状不兼容。加载时必须：

1. 加载所有形状匹配的参数；
2. 明确打印跳过的 `distance_head` 权重；
3. 对新的三通道 distance output layer 使用新初始化；
4. 通过 `--init_ckpt_path` 初始化新实验，不使用 `--ckpt_path` 恢复旧 optimizer state；
5. 将新实验输出到新的 run 名称，绝不覆盖原 checkpoint。

### 7.2 球面模块 checkpoint

球面融合、prototype margin 或 refiner 新增参数时：

- 旧 checkpoint 可作为 `--init_ckpt_path`；
- 新模块参数应显示为 missing/newly initialized；
- 原 backbone、fusion neck、adapter、mask decoder、boundary head 的形状匹配参数继续加载；
- 新模块以 residual/zero initialization 启动，降低分布突变；
- 每次训练保存完整配置副本和 git diff。

## 8. 运行计划

### 8.1 当前基线

```powershell
conda activate sam2path_gpu
cd C:\Users\ASUS\SAM2PATH

python main.py `
  --config configs.Glomeruli_uni2_fusion `
  --devices 0 `
  --project sampath `
  --name baseline_distance3ch `
  --init_ckpt_path "C:\Users\ASUS\SAM2PATH\outputs\glomeruli\run\glomeruli_uni2_resume2\Logs\20260916-145447\sam2path\version_0\checkpoints\epoch=64-step=10010.ckpt"
```

该实验仅用于重新训练三通道 distance head 和与其关联的 geometry 分支；它不是球面融合实验。

### 8.2 高分辨率 refinement 单因素实验

在专用配置副本中设置：

```python
config["model"]["high_res_refinement"]["enabled"] = True
```

其余参数不变。建议运行名称：

```text
sam2uni2_distance3ch_hrrefine
```

### 8.3 球面实验命名

建议为每组建立独立配置模块，而非训练命令中临时覆盖：

```text
configs/Glomeruli_s0_baseline.py
configs/Glomeruli_s1_spherical_proto.py
configs/Glomeruli_s2_spherical_proto_sep.py
configs/Glomeruli_f1_spherical_mean_fusion.py
configs/Glomeruli_f2_spherical_gated_fusion.py
configs/Glomeruli_b3_spherical_gated_hrrefine.py
```

每个配置只相对上一个已验证配置改变一个实验因素。

## 9. 建议日志

每个训练 step 或 epoch 记录：

```text
Loss/train_total
Loss/train_mt
Loss/train_proto_region
Loss/train_proto_intra
Loss/train_memory_logits
Loss/train_memory_feature
Loss/train_geometry_boundary
Loss/train_geometry_distance
Loss/train_geometry_hard
Loss/train_geometry_unsup
```

球面模块新增建议：

```text
Spherical/prototype_cos_glom_vessel
Spherical/prototype_cos_glom_tubule
Spherical/prototype_cos_vessel_tubule
Spherical/token_correct_margin_interior
Spherical/token_correct_margin_boundary
Spherical/gate_mean
Spherical/gate_std
Spherical/gate_boundary_mean
```

门控统计仅在 F2 及后续实验使用。

## 10. 风险与失败判读

| 现象 | 可能原因 | 处理方向 |
|---|---|---|
| Dice 上升、Boundary F1 下降 | 融合提高区域语义但平滑边界 | 不将该融合用于最终模型；单独检查高分辨率 refinement |
| Prototype similarity 全部接近 1 | 温度过高、投影塌缩、损失权重过强 | 降低 prototype 权重，检查 token norm 与类别样本数 |
| 类间 separation 使 vessel 变差 | 生物学外观相近类别被过度排斥 | 减小 margin/权重，使用 pair-specific 分离而非全类硬排斥 |
| Gate 全部接近 0 或 1 | 单一 encoder 主导，门控退化 | 加入 gate 熵正则或退回 F1 简单平均 |
| 训练不稳定 | `arccos` 或过小温度导致梯度不稳定 | 训练使用 cosine similarity，避免直接使用 geodesic arccos loss |
| 高分辨率 refiner 无收益 | 错误已在 decoder logits 前形成，或 FPN 分支无有效信息 | 进行 FPN/low-res logits 可视化，转向 decoder/监督策略 |

## 11. 论文表述边界

在实验没有完成前，只能声明：

> We investigate spherical feature normalization and prototype-based angular semantic regularization for stabilizing category representations in SAM2-UNI pathology segmentation.

不能声明：

> Spherical fusion improves boundary accuracy.

该结论必须由固定评价协议下的 Boundary F1、HD95、类别混淆和可视化失败案例共同支持。

## 12. 最终结果表模板

| Experiment | Fusion | Spherical prototype | Prototype separation | HR refiner | Dice-FG Macro | IoU-FG Macro | BF1@1 | BF1@3 | HD95 | Glom↔Vessel confusion | Vessel↔Tubule confusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 / F0 | concat-conv | no | no | no |  |  |  |  |  |  |  |
| S1 | concat-conv | yes | no | no |  |  |  |  |  |  |  |
| S2 | concat-conv | yes | yes | no |  |  |  |  |  |  |  |
| F1 | spherical mean | best S | no | no |  |  |  |  |  |  |  |
| F2 | spherical gate | best S | no | no |  |  |  |  |  |  |  |
| B2 | concat-conv | no | no | yes |  |  |  |  |  |  |  |
| B3 | spherical gate | best S | no | yes |  |  |  |  |  |  |  |

