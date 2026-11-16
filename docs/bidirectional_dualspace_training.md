# 双向交互—双空间原型—成对实例记忆

> 更新：后续已修正批次读写、独立Memory筛选、类别占用校正、Teacher状态和热启动缓存管理。当前完整BSR配置与行为以 [dualspace_bsr_implementation.md](dualspace_bsr_implementation.md) 为准；下述原始实现说明仅供设计沿革参考。

## 数据流与配置

独立通道投影 → 8×8 窗口双向 Cross-Attention → 通道/空间门控 → 零初始化残差注入原有 concat 主路径 → structure adapter / boundary calibration → decoder 和高分辨率 logit refinement。

新增原型投影输入是 **structure adapter 后的 256 通道特征**，不是直接取原始融合输出。该选择复用现有区域监督接口，约束实际参与解码的结构表征。独立 1×1 卷积得到 128 维欧氏特征与逐像素单位球面特征；不使用 Poincaré 几何。Teacher 包含投影层和实例门控，通过现有 EMA 更新。

## 区域与配对原型

GT 和 Teacher 高置信伪标签经最近邻对齐到特征分辨率，以前景类别连通域提取区域。无标签像素阈值为 0.9，面积与最大区域数沿用基础配置。这里的面积单位是特征网格像素，不是原始图像像素。低置信像素剔除可能拆分区域；类别连通域不等价于真实实例 ID。

令 E、S 为双分支特征，区域 token 为 e=Mean(E|R)、s=Norm(Mean(S|R))。每类每批先等权平均区域 token，再以基础配置的 momentum 更新 P_E；球面原型 EMA 后重新归一化。不按像素面积加权。无样本类别不更新，背景不参与。缓冲区占用类别0槽位但从不填充。

## 匹配与梯度

对可用类别集合 C，定义 a_c=-Mean((e-P_E^c)^2)/0.1，b_c=cos(s,P_S^c)/0.1。分别计算 q_E=softmax(a)、q_S=softmax(b)。实例门控 alpha=sigmoid(MLP([e;s]))，q=alpha*q_E+(1-alpha)*q_S，L_match=-log(q_y)。使用均方距离而不是平方和，避免维度改变导致温度失效；不是用户初始方案中指数距离与余弦的直接混合。

分离损失为 max(0,0.2+max_{k!=y}log(q_k)-log(q_y))，作用于 Student token 和门控；并非对 detached 原型向量直接优化。两类原型就绪后启用匹配。当前不叠加类内紧凑损失，避免第一版目标冗余。

## 成对 Memory

每类16个FIFO槽位，每个槽位同时保存欧氏和球面token，共享指针与有效标志。候选区域需匹配正确原型类别且融合类别概率至少0.6；至少两类原型可用才允许写入。使用Teacher投影及Teacher门控，不向教师反传。

槽位覆盖不做EMA混合：没有真实实例身份关联，旧槽位和新实例不应被假定为同一实例。类别原型仍使用EMA。

对记忆条目重新计算双空间概率混合，损失为 -log(sum_{m:class(m)=y}q_m)。同类记忆全部为正，其他类别为负，不错误排斥同类实例。至少两类记忆可用才生效。当前候选条目数量会影响类别概率质量，需后续评估类别均衡影响。

原型与Memory均是无梯度buffer；投影、门控、上游Student可训练模块接收匹配梯度。冻结编码器不更新。现有区域日志继续使用，Memory/logits日志在本配置实际表示多正例特征匹配损失，不表示分割logits蒸馏。

## 联合优化与启动

保留原有监督、Mean Teacher与已启用几何损失，新增 0.05 L_match + 0.01 L_sep + 0.02 L_memory；无标签项分别沿用基础配置的 unlabeled_weight。旧 intra/Memory feature损失系数明确置0。原型、Memory warmup继承基础配置。

```powershell
conda activate sam2path_gpu
cd C:\Users\ASUS\SAM2PATH
python scripts/test_dual_space.py
python main.py --config configs.Glomeruli_uni2_bidirectional_dualspace --devices 0 --project sampath --name glomeruli_uni2_bidirectional_dualspace
python -m tensorboard.main --logdir outputs/glomeruli/run --port 6006
```

默认不加载旧checkpoint，避免旧 spherical_fusion 同名参数被误用。需要热启动时应先检查旧权重兼容性；weights-only不能恢复优化器状态。此次不启动训练。

## 验证边界

CPU smoke test不等于完整训练验证。应继续检查真实batch、AMP、Teacher EMA与checkpoint重载；本版本不声称提升Dice/HD95，也不输出独立实例ID。训练流程沿用当前弱强增强空间对齐假设；引入独立几何增强时需显式变换标签与区域。
