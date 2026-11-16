# 双向特征交互与区域原型完整设计

配置：`configs.Glomeruli_uni2_bidirectional_prototype`。

SAM2 与 UNI2-h 经过各自可训练 1×1 投影进入256通道空间。空间尺寸不同时对 UNI 特征双线性插值。当前64×64特征被划分成8×8窗口，每个窗口64个token，以4头注意力进行同步双向交互：

`A' = A + 0.1 Attention(LN(A),LN(U),LN(U))`

`U' = U + 0.1 Attention(LN(U),LN(A),LN(A))`

通道门控由拼接特征的全局平均池化产生，空间门控由拼接特征的逐位置卷积产生。两者输出logits相加，再sigmoid，得到[B,256,H,W]的源选择权重G。`V=G*A'+(1-G)*U'`。门控初始为0.5。

融合输出为现有concat neck输出加`0.1 R(V)`。R的末层零初始化，初始保留concat路径；首步主要更新R，之后梯度进入交互与门控。该新分支没有球面归一化，属于欧氏特征上的双向窗口cross-attention，不应再称为球面融合。为兼容现有构造接口，配置暂借用`spherical_fusion`字段，mode明确设为bidirectional。

融合输出经过结构Adapter与边界特征残差校准后，同时进入SAM decoder和区域Prototype学习。GT/Teacher高置信预测的连通区域生成区域均值Token，L2归一化后与停止梯度的Teacher类别原型做温度余弦CE。Teacher用EMA更新，原型用归一化动量更新。三个前景类参与，背景不参与。当前配置关闭Prototype像素池化门控；来源门控已在融合模块显式实现。

继承现有SDF、边界hard CE、separator、高分辨率logit refinement、Mean Teacher与Instance Memory。新增模块不承担实例ID解码职责。可靠teacher边界一致性沿用已有实现；弱强增强必须共享坐标（当前只有光度差异）。

启动使用main.py、该配置与独立run名称。先从基础预训练权重开始可避免旧融合分支参数误载。若warm-start球面模型，需要剔除旧model.spherical_fusion.*，因其计算含义和新模块不同；不能只依靠shape匹配迁移。

已完成独立模块形状、零残差和双向梯度验证。尚未验证真实1024训练batch与显存开销；论文效果待实验。正式比较保留完整测试集，报告每类Dice/IoU、BF1/HD95及合并/拆分情况，排除0/1样本仅作为补充。
