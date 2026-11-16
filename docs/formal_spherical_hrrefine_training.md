# 正式方案：球面原型/记忆融合与高分辨率边界细化

配置文件：`configs/Glomeruli_uni2_spherical_hrrefine.py`。

## 目标与边界

该方案同时处理两类已知问题：

1. **跨类语义混淆**：肾小球、血管和肾小管的局部纹理存在相似性，导致相邻边界被错误类别侵入；
2. **解码空间分辨率不足**：原 SAM decoder 的 64x64 token 经上采样形成 1024x1024 掩码，局部边界、孔洞和细窄结构可能被平滑。

它不改变标签协议：输出始终为 `[B, 4, 1024, 1024]` logits，类别为 `0=background, 1=glomerulus, 2=vessel, 3=tubule`。训练和推理用四类 softmax/argmax；背景仍受监督，但 prototype/memory 只对三个前景类建模。

## 模块 A：零初始化残差球面门控融合

令 SAM2 和 UNI2-h（各经 neck 后为 256 通道）的同位置特征分别为 `F_s` 与 `F_u`。模块先投影并归一化：

`z_s = normalize(W_s F_s), z_u = normalize(W_u F_u)`。

门控图为 `g = sigmoid(G([z_s,z_u]))`，球面特征为：

`z = normalize(g z_s + (1-g) z_u)`。

它经投影得到残差 `R(z)`，最终融合结果是：

`F = FusionNeck([F_s,F_u]) + 0.1 R(z)`。

`R` 的最后投影卷积零初始化。因此训练开始时 `R(z)=0`，模型输出与原 concat + fusion neck 基线相同；新路径仅在梯度训练后贡献增量，避免破坏已有 checkpoint。空间门控允许模型在组织边界等位置更依赖 SAM2 局部结构，或在语义不确定区域更依赖 UNI2-h。

## 模块 B：球面类别 prototype 与实例 memory

教师网络由 EMA 更新且全程 `requires_grad=False`。每个 GT 连通域、以及置信度不低于 0.90 的教师伪标签连通域，产生单位范数实例 token `t`。同类 token 以动量 0.99 更新 class prototype `p_c`；可靠实例 token 以 FIFO 方式存入每类 16 个槽位的 memory。

学生 token 与 prototype 的温度缩放余弦分类损失为：

`L_proto = CE(t p^T / tau), tau=0.1`。

同一区域内的像素特征采用 `1 - cosine(pixel, region_token)` 紧凑化。跨类抑制为：

`L_sep = mean(max(0, cosine(t,p_other)-0.2))`。

instance memory 使用同类最近 memory 的余弦匹配和所有 memory token 的对比分类。它们在第 5/10 epoch 后启动，以便基础分割与 EMA teacher 先稳定。

## 模块 C：高分辨率 logits 残差细化

SAM2 encoder 的最高空间分辨率 FPN 特征 `F_hr` 与 SAM decoder 低：跨编码器空间自适应对齐 → 球面语义约束 → 高分辨率边界恢分辨率四类 logits `L_low` 对齐后共同预测：

`L_refined = L_low + 0.1 H(F_hr, L_low)`。

`H` 的输出卷积零初始化，故启用该模块不会改变初始 logits。它直接修正最终四类竞争 logits，而不是仅训练独立 boundary head；因此能够影响最终 argmax mask。随后使用双线性插值得到 `[B,4,1024,1024]`。

## 完整训练目标

`L = L_supervised + lambda_mt L_MT + L_prototype + L_memory + L_geometry`。

保留既有的监督 Dice/Focal/CE、Mean Teacher 一致性、三通道距离图、边界和 hard-boundary 监督。新组件只添加 prototype separation、球面特征融合和高分辨率 logits correction；没有删除或替换原有监督项。

## 初始化与训练

使用旧 geometry checkpoint 只能作为**初始化**，不能 `--ckpt_path` 恢复：旧 checkpoint 没有新模块参数，且旧 distance head 是两通道。加载器会显式跳过形状不匹配项，新模块保持零初始化。

```powershell
conda activate sam2path_gpu
cd C:\Users\ASUS\SAM2PATH

python main.py `
  --config configs.Glomeruli_uni2_spherical_hrrefine `
  --devices 0 `
  --project sampath `
  --name glomeruli_uni2_spherical_hrrefine `
  --init_ckpt_path "C:\Users\ASUS\SAM2PATH\outputs\glomeruli\run\glomeruli_uni2_resume2\Logs\20260916-145447\sam2path\version_0\checkpoints\epoch=64-step=10010.ckpt"
```

请保留原 checkpoint；新 run 将写到独立目录。每个 epoch 都验证一次。应重点比较 `Dice_FG_Macro/eval`、每类 Dice、边界 F1/HD95、类别混淆矩阵以及 prototype cosine 曲线。若区域 Dice 提升但边界指标下降，应保留球面语义模块而重新评估 refiner 的损失权重，而不是把两种收益混为同一结论。
