# 思维创新：FACET 完整构想

# 研究定位

FACET 是面向端侧 DNN 推理的攻击驱动双平面通道级保护框架。核心目标不是把完整模型或完整 tensor 放入 TEE，而是在真实 MS/MIA 攻击约束下，只保护最小且硬件友好的敏感通道子集，并让绝大多数卷积和矩阵计算继续在 REE 的 GPU/NPU 中执行。

核心研究问题：

- 如何直接测量某个权重通道组对模型窃取（MS）的真实攻击敏感度？
- 如何直接测量某个激活通道组对成员推断（MIA）的真实泄漏敏感度？
- 如何设计既能抵抗权重恢复、又适合端侧加速器的部分通道混淆机制？
- 如何联合优化安全收益、TEE–REE 通信、延迟、能耗和安全内存？

# 基本判断

## MS 与 MIA 必须分成两个保护平面

MS 的主要秘密对象是模型权重与模型功能。若完整权重已经暴露，仅保护运行时激活无法阻止攻击者离线复制模型。

MIA 的主要泄漏对象是样本相关的中间特征、logits、置信度和其他运行时表示。即使权重不公开，只要 REE 能观察高泄漏特征，仍可能获得显著高于 label-only 黑盒基线的 membership 优势。

因此定义两个独立集合：

- W-FACET：Weight-Channel Protection，保护 MS 敏感的权重通道组。
- A-FACET：Activation-Channel Protection，保护 MIA 敏感的激活通道组。

两者通常不相同：

$S_l^W \neq S_l^A$

每个通道组可以处于四种状态：Public、W-only、A-only、W+A。

# W-FACET：权重通道保护

## 保护对象

普通卷积层权重为：

$W_l \in \mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}$

第 $o$ 个输出 filter 为：

$W_{l,o}=W_l[o,:,:,:]$

它由 $C_{in}$ 个二维 kernels 组成，并产生一个完整输出 feature channel。

W-FACET 在同一权重 tensor 内选择敏感 filter 子集：

$S_l^W \subseteq \{1,\ldots,C_{out}\}$

保护：

$W_l[S_l^W,:,:,:]$

其余 filters 保持公开并直接由 REE/NPU 执行。

## 为什么不能只隐藏权重而暴露所有边界

若攻击者知道公开权重、层输入和完整层输出：

$Y=W_{pub}X+W_{sec}X$

则可得到：

$Y-W_{pub}X=W_{sec}X$

收集足够多输入输出对后，可能通过线性方程、最小二乘或 im2col 形式恢复秘密权重。因此 W-FACET 的秘密主体是权重，但必要时还要编码相关边界激活，避免形成可解的权重恢复方程。

## 权重保护机制候选

优先比较：

- 子集内部 permutation。
- permutation + positive scaling。
- 8/16/32 通道小块稠密可逆混合。
- 权重分解与 TEE 内小型秘密补偿。
- GroupCover 类 mutual covering。
- 混淆权重常驻 REE，TEE 只保存恢复密钥和少量状态。

不能默认简单置换或缩放足够安全。需要针对公开模型辅助匹配、cosine similarity、Hungarian matching、ArrowMatch 风格方向匹配和微调恢复进行评估。

# A-FACET：激活通道保护

## 保护对象

第 $l$ 层激活为：

$H_l(x)\in\mathbb{R}^{C_l\times H_l\times W_l}$

选择 membership 泄漏敏感的激活通道：

$S_l^A \subseteq \{1,\ldots,C_l\}$

仅保护：

$H_l(x)[S_l^A,:,:]$

其他通道留在 REE。

## 动态性要求

W-FACET 可以静态或低频更新；A-FACET 必须强调每次推理的新鲜性。固定可逆变换会被攻击者通过大量样本学习，因此优先考虑：

- 每次推理的新鲜随机掩码。
- 由 TEE 种子生成的伪随机通道掩码。
- Slalom 风格 $x+r$ 掩码及离线预计算。
- 小通道组内动态编码。

A-FACET 的目标不是消除最终标签中的全部 membership 信号，而是将 REE 的额外白盒优势压低到 label-only 黑盒基线附近。

# 统一细粒度：图耦合通道组

理论上可以选择单个 filter 或单个 activation channel，但真实设备更适合结构化通道组：

$|G_{l,g}|\in\{8,16,32\}$

保护单元不能只看当前层的一个切片，还要考虑执行闭包：

$C_{l,g}=\{W_l[G,:,:,:],\ BN_l[G],\ H_l[G,:,:],\ W_{l+1}[:,G,:,:]\}$

其中：

- 安全选择单元：敏感 filter group 或 activation-channel group。
- 执行保护闭包：当前 filter、BN 参数、对应 feature channels、后继层输入通道切片及残差分支一致性。

# 四个目标 CNN 的统一映射

## VGG16_BN

最简单的链式结构。以 Conv 输出 filter groups 为 W-FACET 单元，以 Conv/BN/ReLU 后的 feature-channel groups 为 A-FACET 单元。执行闭包包括当前 Conv filter、对应 BN 参数和下一层输入通道切片。

## ResNet18

优先选择 BasicBlock 内部第一个卷积产生的隐藏通道组，避免直接跨 shortcut addition。若保护 block 输出通道，则主分支与 shortcut 必须进入兼容编码域。

## ResNet50

优先选择 bottleneck 内部低维通道组：1×1 reduce 输出、3×3 中间通道和 1×1 expand 输入切片形成耦合权重组。中间维度较小，适合局部混淆闭包。

## MobileNetV2

以 expansion channel tuple 为自然单元：

- expand 1×1 的一个输出 filter group；
- 对应 depthwise filters；
- project 1×1 的对应输入通道切片。

这三部分共同定义一组通道路径，不能独立处理。

# 隐私攻击敏感度：核心方法

## 直接攻击风险

定义保护配置 $P$ 下的攻击风险：

$R_{MS}(P)=\max_{a\in A_{MS}} Success_a(V(P))$

$R_{MIA}(P)=\max_{a\in A_{MIA}} Success_a(V(P))$

MS 指标至少包括：被盗模型 accuracy、fidelity、固定查询预算和固定辅助数据量下的结果。

MIA 指标至少包括：ROC-AUC、balanced accuracy、TPR@1% FPR，并按类别、置信度和样本难度分组。

## 以黑盒基线定义额外泄漏

$L_{MS}(P)=R_{MS}(P)-R_{MS}^{blackbox}$

$L_{MIA}(P)=R_{MIA}(P)-R_{MIA}^{blackbox}$

安全目标：

$L_{MS}(P)\le \epsilon_{MS},\quad L_{MIA}(P)\le \epsilon_{MIA}$

## 条件边际贡献

候选通道组 $g$ 在当前保护集合 $P$ 下的直接收益：

$S_g(P)=R(P)-R(P\cup\{g\})$

不能只测 $S_g(varnothing)$，因为通道组之间可能冗余或互补。

近似 Shapley：

$\phi_g=\mathbb{E}_{P}[R(P)-R(P\cup\{g\})]$

使用随机排列近似条件边际贡献。

## 多保真 attack-in-the-loop 搜索

1. 便宜代理筛选：gradient、JSD、attention transition、权重范数等，只用于缩小候选。
2. 单点真实攻击：逐组保护并运行低预算 MS/MIA。
3. 交互测量：评估 Top-K 组的两两与随机组合，识别冗余和互补。
4. 完整验证：对 Pareto 前沿配置运行完整预算、多个随机种子和未参与搜索的攻击算法。

搜索攻击器与最终评价攻击器必须部分分离，避免过拟合某一种攻击。

# 运行架构

TEE 保存：

- W-FACET/A-FACET 密钥和随机种子。
- 少量恢复参数和映射表。
- 必要的动态掩码状态。
- 最终输出控制逻辑。

REE/GPU/NPU 执行：

- 公开权重组的原始 Conv/GEMM。
- 混淆权重组的主要 Conv/GEMM。
- 非敏感激活通道的全部计算。
- 在编码域中可安全执行的算子。

在当前威胁模型中，若明确排除参数注入、结果篡改和传输篡改，则无需额外加入 Freivalds 等完整性验证；TEE 的主要职责是秘密管理、编码转换与信息释放控制。

# 联合优化目标

$\min_{S^W,S^A,\Pi} C_{device}(S^W,S^A,\Pi)$

约束：

$R_{MS}(S^W)\le R_{MS}^{BB}+\epsilon_{MS}$

$R_{MIA}(S^A)\le R_{MIA}^{BB}+\epsilon_{MIA}$

$M_{TEE}\le M_{max}$

$\Pi$ 包括通道分组、混淆机制、TEE/REE 放置和边界布局。

真实成本不能简单相加，需实测：

- TEE–REE world switch。
- shared-memory copy 和 cache maintenance。
- GPU/NPU kernel splitting。
- 通道对齐和 tensor packing。
- operator fusion 破坏。
- DMA、设备同步和尾延迟。
- 峰值安全内存、能耗与 thermal throttling。

# 与已有工作的差异

## 对 TensorShield

- TensorShield 主要以完整 weight tensor 和 intermediate tensor 为选择/布局单位。
- FACET 进入 tensor 内部，独立选择结构化 weight-filter groups 与 activation-channel groups。
- FACET 以真实 MS/MIA 条件边际风险下降为目标，并与实测硬件成本联合搜索。

## 对 Phantom

- Phantom 选择 Top-K 敏感层位置插入混淆。
- FACET 在单层内部只保护部分通道组，并区分 MS 权重与 MIA 激活两个平面。

## 对 NNSplitter

- NNSplitter 可细到少量标量权重，但目标主要是使未授权模型失效，且不规则稀疏位置不利于 NPU/GPU。
- FACET 采用硬件友好的连续 filter groups，直接面向 MS 攻击成功率，并保留 REE 加速执行。

## 对 ShadowNet / GroupCover / Game of Arrows

- FACET 不是默认采用简单置换，而是把抗权重恢复攻击作为 W-FACET 的核心约束。
- 必须验证公开模型辅助匹配、方向保持、统计相关性和长期密钥复用风险。

# 核心实验计划

## Baselines

- 全模型公开。
- 全模型 TEE 保护。
- TensorShield tensor 级保护。
- Phantom layer 级保护。
- NNSplitter 标量权重保护。
- ShadowNet / GroupCover 权重混淆。
- Slalom 完整激活 masking。

## W-FACET 实验

- 单 filter、8/16/32 filter-group 的 MS 条件边际贡献。
- accuracy 与 fidelity 双指标。
- Knockoff、fine-tuning、public-model-assisted recovery。
- cosine/Hungarian/ArrowMatch 权重恢复。
- 不同混淆强度下的安全—延迟曲线。

## A-FACET 实验

- 单 activation channel 与通道组的 MIA 风险。
- label-only、logit、白盒中间特征 MIA 对比。
- AUC、balanced accuracy、TPR@1% FPR。
- 固定变换与动态掩码的长期观测攻击。
- 未保护通道预测受保护通道的旁路恢复。

## 硬件实验

目标模型：MobileNetV2、ResNet18、ResNet50、VGG16_BN。

需要报告：

- 平均与 p95/p99 延迟。
- TEE–REE 数据量和边界次数。
- GPU/NPU kernel 数量及融合变化。
- 峰值 TEE 内存。
- 能耗与持续推理温升。
- 通道组大小、保护比例和真实吞吐。

# 当前最重要的开放问题

1. 哪种 W-FACET 混淆在只保护部分 filter groups 时仍能抵抗公开模型辅助恢复？
2. 只隐藏少量权重组时，是否必须同步隐藏其输入、输出或 residual 接口？
3. 如何避免 A-FACET 的固定编码被长期样本观测学习？
4. 未保护通道是否足以重建被保护通道或恢复 membership 信号？
5. 通道级拆分是否会严重破坏 NPU 的算子融合和通道对齐？
6. attack-in-the-loop 搜索如何降低计算成本并保持对未知攻击的泛化？
7. W-FACET 与 A-FACET 是否可以共享通道分组、密钥和数据搬运，从而减少额外边界？

# 推荐的第一阶段最小可行原型

1. 先使用 VGG16_BN 和 ResNet18。
2. 将卷积输出 filters 按 16 通道连续分组。
3. W-FACET 先使用 blockwise dense mixing 与 permutation+scaling 两种机制。
4. A-FACET 先使用部分通道一次性加法掩码。
5. 对每个通道组运行低预算 MS/MIA 单点干预。
6. 选择 20–40 个候选组做近似 Shapley 与组合搜索。
7. 在一台真实 ARM TrustZone + GPU/NPU 设备上实测边界和 kernel splitting 成本。
8. 先证明相对 TensorShield 的安全—延迟 Pareto 改进，再扩展至 MobileNetV2 与 ResNet50。