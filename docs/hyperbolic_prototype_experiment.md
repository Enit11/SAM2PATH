# Poincaré Prototype 实验

本配置 `configs.Glomeruli_uni2_hyperbolic_prototype` 在当前球面融合、separator 和高分辨率细化基线上替换原型计算。编码器融合仍为单位球面融合，Instance Memory 仍为余弦记忆。没有新增可训练投影参数。

令曲率为 -c，默认 c=1。区域特征均值 v 经尺度 s=0.25 后映射：

`h = exp_0(s*v)`。

采用 Poincaré 距离平方的负值除温度（沿用 0.1）作为类别匹配 logits。区域紧凑性使用映射后像素与区域 token 的距离平方；异类分离使用 `relu(1 - distance²)`。这项约束作用于 student token 与停止梯度的异类 prototype，而不是直接优化 buffer 中的 prototype。

Teacher 区域 token 通过原点切空间动量更新原型：`P <- exp_0(m*log_0(P)+(1-m)*log_0(h))`，m 沿用 0.99。这是切空间动量平均，不是精确 Fréchet mean。计算使用 float32 并将坐标限制在球内，距离采用稳定的 asinh 形式。

使用 `--init_ckpt_path` 时重建原型，避免将旧单位球坐标误读成双曲球坐标；memory 也遵循原有 warm-start 重建规则。新训练仍有既有 warmup。

对照：同一初始化 checkpoint、数据划分、训练预算和 separator 配置，分别训练原始配置与本配置，推荐至少三个 seeds。评估完整测试集前景 pooled Dice/IoU、逐图 Dice、BF1/HD95，以及固定20张的合并候选诊断。温度和损失尺度改变可能影响结果，不能将单次改善全部归因于双曲几何。

限制：只有三个平行类别，未显式提供层级标签；双曲表示是否优于余弦表示需实验。原始区域特征未按像素 L2 单位化后再映射，保留其幅度作为切空间半径信息。融合路径无需修改。
