# 双向融合、双空间语义建模与 BSR：修订实现

## 1. 当前正式入口

`configs.Glomeruli_uni2_bidirectional_dualspace_bsr`。保留双向窗口交互、可训练UNI neck、双空间类别原型、成对FIFO Memory、Mean Teacher和高分辨率Logit Refinement。不启动正式训练；不加载旧checkpoint作为默认行为。

## 2. 审查问题的逐项修正

1. **独立可靠性策略**：先对原始类别预测提取连通域，再计算有效像素比例。Prototype面积阈值6个特征网格像素、有效比例0.7、像素置信度0.9；Memory面积32、有效比例0.9、像素置信度0.95。GT不需要预测置信度。旧的`min_region_purity`不作为本分支的筛选依据：同类别连通域内的类别纯度天然接近1，不代表可靠性。
2. **本批只读、批末写入**：训练step中的Teacher更新调用只暂存detach token；Student的原型/Memory损失仅访问之前提交的缓存。在`on_train_batch_end`中先用旧原型验证本批Memory候选，再写入Memory，最后更新类别原型。不使用本批新token作为本批Memory正例。
3. **原型更新权重**：分别对GT和无标签区域等权平均，合并为 `(mean_GT + 0.5 mean_U)/1.5`（两源都存在时），每类每批只更新一次EMA。只有一个源时使用该源；尚未就绪类别必须有GT才能初始化。无可靠区域不更新。原型动量0.99。
4. **Memory数量偏置校正**：每个空间对条目score减去该类别当前有效条目数的log，再分别softmax并由实例门控混合。类别所有条目重复相同次数不会凭数量增加该类别概率质量。同类条目全部为正例，异类为负例。
5. **热启动状态**：`load_initial_weights`丢弃`dual_space.*`历史buffer并重置当前成对缓存，再从已加载Student同步Teacher。权重文件仍不含Teacher、优化器及调度器，不声称精确续训。普通state_dict加载仍可恢复同配置成对buffer，用于检查或推理；不能因此推断完整训练状态已恢复。
6. **Teacher模式和更新**：递归`train()`之后将Teacher重新设为eval；冻结的SAM图像编码器及UNI Transformer维持eval，UNI neck仍可训练。EMA按参数名称对应，仅在优化器global_step变化时更新可训练Student参数；Teacher buffer同步自Student。原型及Memory仍按batch提交。
7. **日志**：就绪数读取实际双空间ready；增加每类Memory占用、提交数、接受比例、门控最后一个实例的alpha、各空间最后一个实例的类别匹配、欧氏token范数及上游特征梯度范数。`*_last`明确不是整批准确率。旧分支的即时写入日志在双空间路径不再记录为零。
8. **辅助头捷径风险**：保留独立投影以维持所需双空间表达，新增上游梯度与空间匹配监控；这不保证辅助损失一定改善分割。尚未增加未经验证的额外正则，需通过实验检查门控塌缩及分割收益。
9. **其他**：修正vessel边界加权类别ID为2；新增SDF通道shape检查。单设备训练受支持，多卡因未实现跨卡区域同步而明确报错，避免缓存静默分叉。

区域均在当前64×64特征网格处理。Memory的32网格像素面积阈值可能排除很小的组织，应观察接受率，而不能将过滤后的连通区域称为真实实例标注。当前流程依赖弱强增强空间对应；若加入独立几何增强，必须另行变换区域。

## 3. BSR的三个组成部分

### 3.1 三前景 SDF 辅助预测

复用现有distance head，不新增重复头。输出 `[B,3,H,W]`，channel0/1/2分别对应Glomerulus/Vessel/Tubule，当前H=W=1024。

为兼容已有距离场定义，使用**外正、内负**：

\[
D_c(i)=\operatorname{clip}((d_{\rm outside}(i)-d_{\rm inside}(i))/R,-1,1),\quad R=32.
\]

这里outside指在类外到该类的距离，inside指类内到类外的距离，单位为**训练标签1024网格像素**，不是原始WSI像素或物理微米。空类别全+1、全图类别全-1。GT背景腔隙保留原样。

\[
L_{\rm sdf}=\frac1{3BHW}\sum_{b,c,i}|\hat D_{b,c}(i)-D_{b,c}(i)|.
\]

BSR配置将原SmoothL1替换为L1。预测头不强制tanh，训练目标截断归一化；推理导出的距离图保持同一符号，只有可视化时裁剪范围，不将SDF阈值用于填洞。

### 3.2 边界感知结构特征校准

使用`network.bsr.BoundaryStructureRefinement`替换本配置的旧校准器，不叠加两套校准。

\[
F_b=\operatorname{GELU}(\operatorname{Conv}_{3\times3}(F)),\quad
A_b=\sigma(\operatorname{Conv}_{1\times1}(F_b)),\quad
F_{\rm ref}=F+\gamma_b A_b\odot F.
\]

F为structure adapter输出的256通道特征，F_b为64通道，A_b单通道并广播。gamma_b是可训练标量，初始0，初始输出严格等于F。第一步主要学习gamma，之后梯度进入边界投影和门控。校准后特征供Decoder、SDF和双空间投影共同使用。原辅助boundary head继续由GT边界BCE监督，但A_b不被宣称为真实边界，也没有把预测边界替换成GT。

### 3.3 GT距离加权分割CE

从语义标签相邻变化提取边界，两侧像素均标为边界，包括真实背景腔隙周围的界面。计算到最近边界的欧氏距离d_i：

\[
w_i=1+4\exp(-d_i/3),\qquad
L_{\rm bw}=-\frac1{N_{\rm valid}}\sum_{i\in\Omega_{\rm valid}}w_i\log p_{i,y_i}.
\]

权重由GT计算，无梯度；分母是有效像素数，不是权重总和。同质整图无边界时w=1；忽略像素不参与。sigma=3同样使用1024标签网格像素。

仅由有标签GT生成SDF和加权CE，不借Teacher伪标签捏造这些监督。无标签仍使用原有像素一致性、可靠区域原型/Memory、Teacher可靠边界一致性。

## 4. 联合目标与启用条件

保留基础监督项（当前基础配置包括Focal、Dice、CE和区域一致性项）及Mean Teacher；双空间系数为匹配0.05、Student对原型分离0.01、Memory多正例匹配0.02，紧凑性及旧Memory feature损失均0。

BSR沿用geometry warmup=10：SDF L1系数0.10、GT边界加权CE系数0.10。已有辅助boundary BCE=0.20、同类间隙separator头=0.05、主logit间隙CE=0.10、边界一致性=0.05保留。旧hard-boundary CE系数设0，避免本版本重复强化边界CE；非法的不同图像geometry MSE明确禁用。Prototype warmup=5，Memory warmup=10。

特征校准器从训练开始参与主分割路径；几何辅助损失在warmup之后启用。高分辨率Logit Refiner继续参与最终logits预测。几何约束不代表已实现拓扑保证或实例边界归属预测。

## 5. 不自动填洞

新配置设置`inference.preserve_raw_semantics=True`，`sam2path_inference.py`遇到此开关直接保留raw argmax掩码，跳过旧的自动填洞、开闭运算和组件重标注。仅对新配置生效，不改变其他实验的默认后处理。正式对比使用同一raw评价协议。

## 6. 验证与运行

```powershell
conda activate sam2path_gpu
cd C:\Users\ASUS\SAM2PATH
python scripts/test_dual_space.py
python scripts/test_bsr.py
python scripts/test_dualspace_training_flow.py
python main.py --config configs.Glomeruli_uni2_bidirectional_dualspace_bsr --devices 0 --project sampath --name glomeruli_uni2_bidirectional_dualspace_bsr
python -m tensorboard.main --logdir outputs/glomeruli/run/glomeruli_uni2_bidirectional_dualspace_bsr --port 6006
```

三个测试分别覆盖配对缓存与梯度/类别数量校正/独立阈值、GT腔隙与精确加权CE/残差校准、真实SamSeg控制流的三批小模型训练及Teacher/缓存/检查点/热启动。第三项用微型替身编码器，不等于完整SAM2–UNI GPU训练验证。此次没有启动正式训练，没有HD95或Dice提升结论。
