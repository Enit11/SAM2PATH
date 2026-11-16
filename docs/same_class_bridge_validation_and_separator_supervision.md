# 同类实例粘连验证与分隔带监督

## 验证对象与协议

- 模型：`glomeruli_uni2_spherical_hrrefine_warmstart` 的 `last.ckpt`（epoch 46，step 7238）。
- 测试集：病人级划分的 99 张测试图像；本次仅固定抽取 seed 42 的 20 张。
- 预测：四类 logits 的 `softmax` 后 `argmax` 的 **raw prediction**，未使用形态学后处理。
- 粘连定义：一个预测类别连通域同时覆盖至少两个同类别 GT 连通域；统计其中靠近这两个 GT 实例、且 GT 为背景的候选桥区。
- 桥区半径：24 个 1024-space 像素。一个大预测连通域可产生多个 GT 实例对，故实例对数量不能解释为独立物理桥数量。

## 结果

- 20/20 张抽样图像出现至少一个候选同类合并。
- 共检测到 292 个 GT 实例对：class 2 为 290 对，class 3 为 2 对。
- 面积加权的桥区统计：目标类概率 0.7337，背景概率 0.2627，概率差 0.4710，目标减背景 logit 差 +1.3185。
- 每一个候选实例对的平均 logit 差均大于 0；143/292 对大于等于 +1。

结论：这些连接桥并非单纯由 argmax 在近似平局处产生；模型在真实背景缝隙上对目标类别已经形成稳定正响应。问题主要集中于 class 2。

原始证据：

- `outputs/bridge_diagnosis/spherical_hrrefine_epoch046_seed42_20/bridge_statistics.csv`
- `outputs/bridge_diagnosis/spherical_hrrefine_epoch046_seed42_20/panels/`

## 实现的修正：labelled same-class separator supervision

对带标注图像中类别 `c` 的不同 GT 连通实例 `I_i^c` 与 `I_j^c`，用半径 `r=6` 的膨胀区域构建仅位于真实背景的分隔带：

\[
S_{sep}=\bigcup_{c,i\ne j}\left[\operatorname{Dilate}(I_i^c,r)\cap\operatorname{Dilate}(I_j^c,r)\cap\{y=0\}\right].
\]

代码不采用两两枚举，而是对每一实例膨胀后统计覆盖次数；覆盖次数大于等于 2 且标签为背景的像素构成 `S_sep`。小于 6 像素的 GT 组件不参与生成，以避免噪声标注产生大量伪分隔带。

训练增加两项：

\[
\mathcal L_{sep-logit}=-\frac{1}{|S_{sep}|}\sum_{p\in S_{sep}}\log\operatorname{softmax}(z_p)_0,
\]

它直接对主四类 logits 施加 background 竞争；以及一通道 `separator_head` 的类别不均衡 BCE 辅助损失 `L_sep-head`。正式配置权重为 `0.10` 和 `0.05`。不从 teacher 伪标签创建 separator target，也不向 prototype/memory 写入该目标。

为避免无效监督，配置同时将 `geometry.unsup_coef=0`：此前该项比较了有标注 student 图像与无标注 teacher 图像的 geometry 输出，两者不是同一图像。

## 验证计划

训练后用同一 seed、同一 20 张和同一 bridge radius 重跑 `scripts/diagnose_same_class_bridges.ps1`（替换 checkpoint）。主要预注册检查：桥区 `z_c-z_0`、候选合并实例对/图、class-2 Dice/IoU、BF1 和 HD95。不能只依据 Dice 宣称分隔改善；须同时确认真实细小 tubule/vessel 没有被过度切断。
