# TSDP 项目交接

本文面向一个完全没有历史上下文的新会话。开始工作前必须先阅读本文件、`AGENTS.md`、`README.md`、`STRUCTURE.md` 和 `FLOW.md`。如本文与实际代码或正式实验最近一层的 `README.md` 不一致，以代码和最近一层已经固化的正式协议为准，并先查明差异，不能直接运行实验。

## 1. 我们在做什么

TSDP 是一个围绕 TEE 场景下模型参数保护的研究项目。当前只推进 Model Stealing（MS）部分，Membership Inference Attack（MIA）和真实 TEE 部署暂不展开。

研究目标不是发明新的模型窃取算法，而是在统一、较强的攻击协议下比较不同参数保护粒度对 MS 的抑制效果。最终创新方向是把保护粒度从已有工作的完整层、完整 tensor 或离散标量进一步细化到通道块，并提取适合保护的关键路径。

当前阶段的具体任务是：

1. 固定统一 MS 数据、查询、初始化、训练和评估协议。
2. 在 `ResNet18+CIFAR-100` 上完成已有保护策略 baseline。
3. 以该组合为验证基准，下一步实现本项目的通道块保护策略。
4. 确认通道块策略有效后，再扩展到四个数据集和四个模型。

## 2. 当前有效的统一 MS 协议

### 2.1 数据划分

当前采用与公开工作一致的随机重叠思路，不再使用早期讨论过的 `strict_disjoint`：

```text
victim_train   官方训练集全量，用于训练 victim
query_pool_ms  从同一官方训练集随机无放回抽取 1%
eval_ms        官方 test 或 validation split；普通 victim 逐轮评估，surrogate 用于攻击评估
```

query 因而可能与 victim 训练数据重叠，这是当前论文 baseline 协议的有意选择，不是数据泄漏 bug。普通 victim 不另划 validation split，而是与 TensorShield、TEESlice 两个参考仓库保持一致，在 `eval_ms` 上逐轮评估并保存 accuracy 最高的 `best.pth`；该 checkpoint 用于生成 query 输出。这个约定只适用于 victim 固定，不能用于选择 surrogate 正式结果。单随机种子正式结果当前统一为 `42`。普通 MS surrogate 必须按 `formal_victim_then_public_v1` 重放目标类别 victim→ImageNet public→目标类别任务头的构造轨迹，并为 query DataLoader 使用由同一实验 seed 显式构造的 generator；完整规则见 `AGENTS.md`。后续多随机种子实验只替换预先固化的 seed，不改变构造顺序。

正式 surrogate 在当前 budget 的 query 前缀内按 seed 42 与固定 offset 100 随机拆成
80% query train 和 20% query validation。以 `C100` 的 500 条为例，400 条只用于
梯度更新，100 条只用于选 checkpoint；`eval_ms` 不参与 surrogate 选模。

四个数据集的正式 query budget：

```text
c10    500
c100   500
s10     50
t200  1000
```

数据协议只允许通过 `dataset/MS/<dataset>/manifest.json` 的 `query.split=query_pool_ms` 引用 `splits.tsv`。不得重新建立 `query.tsv`，不得恢复 `dataset/query/` 或 `dataset/auxiliary/`。

### 2.2 victim 与查询输出

- 普通 victim 使用 `weights/MS/victim/<model>/<dataset>/best.pth` 生成查询输出。
- 普通 victim checkpoint 使用 `best.pth` 和 `end.pth`；正式 surrogate 只保存
  validation-best `best.pth`。两者都不要恢复误导性的 `target.pth`。
- 正式部分保护策略和 `full_protection` 的攻击者可以观察完整 softmax posterior，标签模式为 `soft`。
- `full_protection + soft` 是 soft-posterior 黑盒；`hard_blackbox` 是相同完整保护 mask 下的 label-only 黑盒。两者都是正式黑盒边界，汇总图必须同时展示。
- posterior 生成、soft surrogate 和正式 hard_blackbox 读取 query 时都必须使用确定性的 test transform。
- soft posterior 绝对不能与 `RandomCrop`、随机翻转等训练增强后的图像绑定，否则标签与输入错配。

### 2.3 surrogate 初始化与训练

普通固定 victim baseline 的规则：

```text
暴露状态          从 victim best.pth 精确复制
受保护 backbone   保留官方 ImageNet 初始化
分类头完整暴露    复制 victim 分类头，head_mode=exposed
分类头部分保护    按标量 mask 混合复制，head_mode=mixed
分类头完整保护    随机替换为目标类别 Linear，head_mode=replace
训练方式          除无保护恒等组外，全部参数共同 finetune
epochs            100
batch size         64
optimizer          SGD(lr=0.01, momentum=0.5, weight_decay=5e-4)
scheduler          StepLR(step_size=60, gamma=0.1)
```

100 轮是当前统一的强攻击协议。TensorShield 原仓库正式扫描本身使用 100 轮；TEESlice 部分旧脚本虽然只训练 10 轮，但现有日志表明 10/20 轮会明显低估攻击者。不要只把 epoch 改成 50：当前 `lr_step=60` 与 100 轮配套，单独截断会使学习率从不衰减，并令现有全部正式结果失效。

### 2.4 checkpoint 与评估

- victim query 固定读取按上述规则产生的 `best.pth`；普通 victim 的 `end.pth` 只保存训练终点。
- 正式 surrogate 只保存 `best.pth`，不再保存或消费 surrogate `end.pth`。
- soft 攻击按 validation soft cross-entropy 最低的 epoch 选模，hard 攻击按 validation hard cross-entropy 最低的 epoch 选模；数值并列保留更早 epoch。
- checkpoint 固定后才构造并遍历 `eval_ms`，每个 artifact 只完整评估一次。
- `no_protection` 是 epoch 0 恒等控制组，其 `best.pth` 逐状态等于 victim。
- 正式保存原始 `accuracy`、`fidelity`、`posterior KL` 及计数。下降值、倍数和归一化指标只在绘图时派生。

## 3. 基础保护单位

`ResNet18` 以 `state_dict` 稳定顺序定义 122 个基础 tensor unit，索引为 `0..121`。每个 unit 对应一个参数或 buffer 条目。所有完整层、TensorShield 和未来通道块方案都应能回溯到这一权威状态空间，不能再发明相互冲突的底层编号。

完整层 baseline 使用官方 18 层：stem conv 为第 1 层，8 个 BasicBlock 的 16 个主卷积为第 2 至 17 层，分类头为第 18 层。BN、downsample 和池化归属详见 `exp/MS/train_surrogate/README.md`。

TensorShield 图中的 L1-L7 只是论文展示分段，不是新的网络深度，也不是第三级映射。不要用 L1-L7 替换 122 unit 或官方 18 层。

## 4. 已经完成的工作

### 4.1 数据、victim 与查询产物

- 四个数据集的 MS manifest 和 split 已建立在 `dataset/MS/`。
- 普通 victim 训练、`best/end` checkpoint 和查询标签流程已经实现。
- `get_label.py` 支持普通 victim 和 `teeslice_r18`。
- posterior、labels 和 victim checkpoint SHA256 在 manifest 中绑定。

### 4.2 普通 ResNet18+C100 上下界

正式主结果：

| 策略 | epoch | surrogate accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|
| no protection | 0 | 0.6182 | 1.0000 | 1.0591e-9 |
| full protection（soft posterior） | 45 | 0.1390 | 0.1463 | 3.039817 |
| hard blackbox（label-only） | 3 | 0.0890 | 0.0969 | 3.387234 |

当前 `full_protection` 表示所有 victim 参数不可见，但查询接口仍暴露 posterior。不要把它悄悄改回 label-only 黑盒；label-only 使用独立的 `hard_blackbox` artifact。soft 与 hard 都是正式黑盒参考：前者用于和 posterior-visible 部分保护策略进行同接口比较，后者用于展示 label-only 查询能力边界。两者都按各自 query validation cross-entropy 选择最早的最优 `best.pth`。

### 4.3 三种完整层 baseline

以下策略均已完成 8 个保护规模，共 24 组正式结果：

- `shallow_02..16`：从输入侧连续保护 2、4、...、16 个官方层。
- `middle_02..16`：围绕网络中部连续、对称扩展，对应 SOTER 思路。
- `deep_02..16`：连续保护到官方第 18 层。

三者按相同层数扫描，但实际参数比例差异很大。这是原有完整层策略的真实属性，不要为了“看起来公平”篡改保护集合。绘图和与本方法比较时必须同时使用实际 `protected_param_ratio` 或 FLOPs 成本。

新协议下完整层结果不严格单调：`middle_14` 相比 `middle_12`、`deep_16` 相比 `deep_14` 出现小幅反弹。`deep_14` 为 `0.1184/0.1300/3.158334`，个别指标越过 soft 黑盒；这只说明单 seed、有限 query 和攻击选模存在波动。攻击者可以忽略暴露状态并回退到 soft 黑盒，不能将其解释为信息意义上强于黑盒。

### 4.4 全局大权重标量 baseline

`large_01..08` 已按 `0.01、0.1、0.3、0.5、0.7、0.8、0.9、0.95` 来源锚点转换为绝对整数预算并全部运行。

该策略按所有 eligible weight 标量的绝对值全局排序，不是比较整个卷积 tensor 的绝对值和。mask 可以在同一 tensor 内逐标量离散保护。

分类头出现部分标量保护时必须使用 `mixed`：未保护标量复制 victim，保护标量保留随机初始化。绝对不能因为分类头只保护了一个标量，就丢弃整个分类头的其他暴露参数。这个错误曾导致保护成本与实际不可见参数不一致，已经修复并重跑。

新协议下该扫描的 accuracy/fidelity 随保护比例严格下降，posterior KL 严格上升。`large_01` 在 `1.0411%` 下为 `0.3637/0.4204/1.486967`，明显强于成本接近的 `shallow_04`；但直到约 80% 以上保护比例才接近 soft 黑盒。`large_08` 为 `0.1356/0.1424/3.095049`，略越过 soft 黑盒同样只能按攻击训练波动解释。

### 4.5 仅分类头保护控制组

`head_only` 已在普通 `ResNet18+C100` victim 上完成。该策略只隐藏 `last_linear.weight` 和 `last_linear.bias`，完整复制其余 backbone 状态；保护 `2/122` 个 unit、`51,300/11,227,812` 个参数，参数比例为 `0.4569%`，分类头使用 `replace`，随后对全部参数共同 finetune。

正式 validation-best 结果：

```text
run id                  d2d4a36a3208
best epoch              93
accuracy                0.3985
fidelity                0.4621
posterior KL            1.616573
```

它比参数比例相近的 `shallow_02`（`0.5608/0.7215/0.412557`）更能抑制 MS，但仍明显没有达到普通 victim 的 soft `full_protection`。该实验用于量化分类头不可见的独立贡献，不重复 Lab 02 的 replace/adapter 选择实验。

### 4.6 TensorShield baseline

当前实现只使用论文作者直接提供的 `ResNet18+CIFAR-100` 最终 rank，不再保留 importance、Grad-CAM 或公式估计代码。

权威文件：

```text
exp/MS/train_surrogate/selector/tensorshield.py
exp/MS/train_surrogate/defense/tensorshield.py
```

正式 mask 是作者 eligible rank 的 Top-10 weight 集合，并联动保护 `last_linear.bias`：

```text
protected unit          11/122
protected params        1,009,764 / 11,227,812
protected ratio         8.9934%
mask SHA256             1e3aa38124f084dd39eab42a4d3f1ddf1ca86807812796c66a8318c05e7aa2cb
head mode               replace
```

正式 validation-best 结果：

```text
run id                  5057ffe55a3e
best epoch              93
accuracy                0.1728
fidelity                0.1865
posterior KL            2.694492
```

该点明显强于相近成本的浅层与大权重扫描点，但三项仍未达到 soft 黑盒 `0.1390/0.1463/3.039817`。只有 `ResNet18+C100` 获得了作者 rank。不得把该 rank 复用到其他数据集或模型，也不得重新启用公式估计冒充作者结果。没有对应作者 rank 的组合应标记为不适用。

### 4.7 TEESlice standalone 复现

TEESlice 改变 victim 结构和训练过程，不能写入普通 ResNet18 固定 victim 的主 `metrics.tsv`，也不能与普通 mask 策略假装是完全同条件点。

当前 `ResNet18+C100` 已实现四阶段 defended victim：

```text
source   在私有训练数据上监督训练 CIFAR-stem ResNet18
teacher  使用 source posterior 蒸馏同结构 teacher
full     冻结公开 backbone，训练 private proxy、alpha、分类头并适配 BN
prune    只在 victim_train 内部验证集动态删除低 alpha proxy
```

最终 victim：

```text
checkpoint              weights/MS/victim/teeslice_r18/c100/best.pth
eval_ms accuracy         0.7578
active proxy             8
private params           703,092
private param ratio      5.9223%
private FLOPs            27,756,644
```

剪枝判断只能使用 victim_train 内部验证集，不能用 `eval_ms` 调节拓扑。

TEESlice 黑盒攻击者能力已经固定为：知道最终 pruned topology 和保护策略，只复制 `keep_flags` 连接关系与官方 ImageNet backbone；private proxy、alpha、分类头和任务 BN 状态 fresh 初始化，然后全部可执行路径参数 finetune。完整状态白盒重新加载 victim 全部状态并实际评估，不再手工填解析上界。

正式结果：

| 能力 | 主评估点 | accuracy | fidelity | posterior KL |
|---|---|---:|---:|---:|
| blackbox known topology | surrogate best epoch 92 | 0.1580 | 0.1698 | 3.342776 |
| whitebox full state | victim 实际评估 | 0.7578 | 1.0000 | 3.5986e-10 |

TEESlice 结果只在 `results/MS/resnet18/c100/teeslice/` 保存，并标记为 `standalone_reproduction`。不要覆盖普通预训练模型的 `no_protection` 或 `full_protection`。

### 4.8 Lab 结论

训练普通 surrogate 的 `lab/02`、`04`、`05`、`06`、`07`、`08` 已统一复用
`lab/protocol.py`：500 条 soft query 固定拆为 400 train / 100 validation，最多训练
100 epoch，按 validation soft cross-entropy 选择最早 best，checkpoint 固定后每组只
评估一次完整 `eval_ms`。`lab/03_baseline` 继续作为只读正式结果汇总入口。

有效 Lab 必须保留：

- `lab/01_kmeans`：预训练特征聚类验证。
- `lab/02_head`：分类头 replace/adapter 与 frozen/finetune 消融。两种保护范围的
  finetune 对照中，replace 三指标均强于 adapter，因此普通 MS 继续使用分类头替换；
  随机拼接 victim/public 状态后冻结暴露权重会使攻击几乎失效，不能作为主攻击者。
- `lab/03_baseline`：四种通用 baseline 曲线、分类头与 TensorShield 单点、TEESlice standalone 单点、普通 victim 的 soft no/full 主参考线，以及 hard-label 全保护辅助参考线的三指标总图。
- `lab/04_tensorshield`：作者 eligible rank 的 Top-1 至 Top-17 前缀曲线、Top-12
  完整 leave-one-out 与五组联合删除、四策略十种子候选，以及前/后/分散十项位置
  对照。Top-10 与正式 TensorShield 逐值一致，为
  `0.1728/0.1865/2.694492`；Top-12 为 `0.1403/0.1519/2.856598`，仍未同时达到
  soft 黑盒 `0.1390/0.1463/3.039817`。新 2×2 中删除 rank-6、rank-7 或同时删除
  都使攻击反弹，不再支持“unit 48 已使 unit 36 条件冗余”的旧串行解释。前十项只
  保护 `14.25%` 参数即强于分散十项和后十项，说明保护位置比纯参数量更重要。
  十种子实验保留四根柱子：Top-10、Top-10+全部 BN gamma、删除 rank-5/8/10
  +BN gamma，以及额外删除 rank-6；四组均保护分类头 weight/bias。四个均值依次为
  `0.17277/0.18638/2.66467`、`0.13136/0.13994/3.01441` 和
  `0.11975/0.12755/3.13450`、`0.11259/0.12141/3.17270`。BN gamma 相对 Top-10
  在 `10/10` seed 上三指标
  同向改善；给定 BN gamma 后删除 rank-5/8/10 也在 `10/10` seed 上未引发攻击
  反弹，并将比例从 `9.0362%` 降到 `7.0662%`。额外删除 rank-6 后比例降至
  `5.7529%`，相对原候选在 `8/10` seed 三项同时改善。相对逐 seed soft 黑盒，
  三项同时达到的次数依次为 `0/10`、`8/10`、`9/10`、`10/10`。这支持优先保护
  Top-10 中五个 `conv1.weight` 而不保护四个 `conv2.weight` 的当前后验观察，
  但不能推广为跨模型先验。该集合仍为读取 MS 反馈的后验有效集合；暴露更多信息后
  的指标改善只能解释为固定攻击优化负迁移。
- `lab/05_state`：五种完整 state 类型与十三种参数语义组。只保护主路径 Conv
  即使占 `97.84%` 参数仍为 `0.2210/0.2468/2.206965`，明显弱于全部 weight 的
  `0.1435/0.1490/2.902834`；Stem、downsample、BN affine 与分类头 weight 不能从
  图模型中排除。相同参数量下 BN gamma 明显强于 beta；分类头作用几乎全部来自
  weight；BN buffer 只作为执行闭包状态记录。新增固定五个候选 conv1 与完整分类头
  的 BN gamma 十种子分组消融：No/All gamma 均值分别为
  `0.18249/0.19780/2.61784` 与 `0.11259/0.12141/3.17270`，后者在 10/10 seed
  三项同时改善。相对 All gamma，删除 Stem 在 9/10 seed 三项反弹；删除 Block BN2
  或 Downsample BN 都在 10/10 seed 三项反弹，平均变化分别为
  `+0.02225/+0.02312/-0.20992` 和 `+0.02224/+0.02374/-0.20939`。删除 Block BN1
  反而得到 `0.10654/0.11475/3.21235`，7/10 seed 三项同时改善并保持 10/10
  黑盒三线；这应解释为 protected public conv1 与 exposed victim BN1 的固定拼接
  失配，不代表主动泄露 BN1 对适应性攻击安全。
  另以 seed 42 完成从 No gamma 分别单独加入四类 gamma 的对偶实验：No gamma 为
  `0.1798/0.1947/2.642482`，Stem 为 `0.1546/0.1699/2.782362`，Block BN1 为
  `0.1785/0.1907/2.645210`，Block BN2 为 `0.1481/0.1633/2.803683`，Downsample
  BN 为 `0.1376/0.1509/2.939023`。相对基线的保护改善排序为 Downsample >
  Block BN2 > Stem >> Block BN1，与 drop 必要性方向一致；但 Downsample 也未同时
  达到 seed-42 soft 黑盒三线，说明单组不能替代跨层 gamma 闭包。
- `lab/06_weight`：在 Top-10 至 Top-17 上分别补充全部 BN gamma、downsample
  Conv、二者组合、Stem Conv 与三类并集。新协议下 BN gamma 在全部八个 k 上都
  一致改善三项指标；`Top-11/12/13/17 + BN gamma` 严格越过 soft 黑盒三线。
  `Top-10 + BN gamma + Downsample` 保护 `10.5684%` 参数并得到
  `0.1240/0.1336/3.072730`。Downsample 单项始终有保护方向但未独立达到三线；
  组合与并集存在优化非单调性。越过黑盒只能解释为当前攻击无额外收益，攻击者仍可
  退回黑盒训练；结论仍需多随机种子和跨模型复现。
- `lab/07_structure`：除逐 unit 结构表和 TensorShield Top-17 标注外，已对
  5.7529% 基础集合中的五个前中层 `conv1.weight` 完成 seed 43–52 的逐一暴露
  配对实验。暴露 rank-9 `layer3.0.conv1.weight` 的平均反弹最大，为
  `+0.03705/+0.04144/-0.34032`，且 10/10 seed 三指标同时反弹、0/10 seed
  仍同时维持 matched soft 黑盒边界；rank-2 次之，为
  `+0.02969/+0.03119/-0.22151` 和 10/10 同向反弹。rank-7、rank-4、rank-1
  反弹较弱，其中后两者分别只有 9/10、8/10 seed 三指标同时反弹。当前证据说明
  五个 `conv1.weight` 并非等价成员，rank-9 是最强条件依赖，rank-2 是第二强；
  结论只针对其他保护项固定时的后验 leave-one-out 条件，不能直接外推为跨模型先验。
  又在相同五个 BasicBlock 内将五个 `conv1.weight` 一一换成对应
  `conv2.weight`，保持分类头与全部 BN gamma 不变。尽管保护比例由 `5.7529%`
  增至 `9.0362%`，攻击指标仍平均变化
  `+0.01216/+0.01331/-0.06859`，accuracy/fidelity 在 10/10 seed 均反弹；
  对应 conv2 不能保持 conv1 的保护效果。这是位置替换反例，不是等成本比较，
  也不能单独证明坐标转换机制。
- `lab/08_leakage`：固定同一 5.7529% 保护集合，将已泄露浮点状态从 public 向
  victim 取 0%/25%/50%/75%/100% 利用强度。100% 端点的 epoch-0 validation
  soft CE 为 `11.47155`，远高于黑盒 `4.76294`，并在 10/10 seed 上最终三指标
  都劣于黑盒，证明直接完整拼接确实造成严重初始化负迁移。但三个中间强度均在
  10/10 seed 上同时优于黑盒，50% 最强，达到
  `0.19574/0.20706/2.56165`。只按 query-validation loss 选择时，适应性攻击者
  在 10/10 seed 全部选中 50%，因此标准 100% 混合攻击低于黑盒是可被状态收缩
  绕开的初始化陷阱，不能视为稳健黑盒等效证据。后续机制问题应转向：为什么当前
  被保护接口会使完整 public/victim 拼接在端点产生如此强的非线性不兼容。
- `lab/09_mechanism`：已用 seed 43–52 的 query-validation 做完不训练、
  不读取 `eval_ms` 的前向因果分析。100% 端点的特征/logit RMS 均约为 0% 的
  `8×`；只把分类头输入逐样本缩回 public 特征范数，posterior KL 就从
  `10.92288` 降到 `4.22468`，说明端点“失明”主要是随机私有头与过大暴露特征的
  幅值失配。全部 BN gamma 在 near-victim 上造成 `12.69042` KL，其中四类分别
  单独替换的损伤和仅 `4.29184`，证明主要作用是跨层尺度闭包；BasicBlock BN2
  的 Shapley 损伤 `6.71501` 最大。五个 conv1 中只有 `layer2.0` 和 `layer3.0`
  的 near-victim 损伤明显较大，并与 Lab07 MS 反弹排序一致；八块对照中只有
  3/8 的 conv1 weight 损伤超过相邻 conv2，因此不存在普遍的 conv1 优势。
  当前准确机制是分类头幅值、BN 跨层尺度和两个高敏感 conv1 的联合接口割，而非
  每个候选都独立承载攻击信息；该接口失配仍可被 Lab08 的状态收缩绕过。
- `lab/10_pair`：在 `layer1.0/1.1`、`layer2.0/2.1` 和 `layer3.0` 五个固定块上，
  用 seed 42 比较两种局部配对保护。五个 `conv1.weight` + 对应 `bn2.weight` +
  完整分类头保护 `5.7158%` 参数，得到 `0.1532/0.1662/2.808921`；对应五个
  `conv2.weight` + `bn1.weight` + 完整分类头保护 `8.9991%`，得到
  `0.1562/0.1694/2.784013`。前者三项略优且成本低 3.2833 个百分点，但二者都
  未达到 seed-42 soft 黑盒 `0.1390/0.1463/3.039817`；局部五个 gamma 不能替代
  Stem、Block BN2 与 Downsample BN 构成的跨层闭包。该比较只有单 seed，不能
  表述为稳定排序。
- `test/MS/01_cross`：数据侧计算已精简为固定 500 张 `query_pool_ms` 的单一协议，
  不再计算 50,000 张全训练集对照。候选为 20 个 Conv weight 与 20 个 BN affine
  参数组；每个 BN 组同时包含 weight/gamma 与 bias/beta，并从同一结果抽取 16 个
  BasicBlock 主分支卷积。每张图片计算 `z_pp/z_pv/z_vp/z_vv`、
  紧凑交叉项 `I` 和自然残差 `z_uu-z_pp`，再以通道为列计算六类谱熵有效秩。
  最终九项为两个残差幅值、`r(I)`、`r(z_uu-z_pp)`、三组 rank gap、rank
  interaction，以及两个残差幅值的乘积。40/16 项各生成九张图；所有图按指标
  绝对值降序，柱子和文字保留真实符号。40 项交叉幅值首位仍为
  `layer4.1.bn2=1.464529`，交叉有效秩首位为
  `layer1.1.conv2=15.273210`，rank interaction 绝对值首位也是
  `layer1.1.conv2=+4.793478`；自然残差首位是 stem `conv1=1.311637`，但其
  交叉项、交叉秩和乘积严格为零。乘积前三为 `layer2.0.conv1=1.061349`、
  `layer1.1.conv1=0.982852` 和 `layer1.0.conv1=0.896433`。beta 在四路交叉差分中
  抵消，因此 BN 交叉项不变；但自然残差、相关有效秩和乘积分数均已重算。数据侧产物只保留
  `metrics.json`、`all.tsv`、`main.tsv` 和 18 张当前图片；另有 `sweep.py`
  固定保护完整分类头，按乘积分数扫描完整 tensor 前缀；使用统一 400/100 soft
  validation-best 协议，在 Top-4 第一次 accuracy 反弹后选择 Top-3。Top-3 保护
  比例 1.770211%，accuracy/fidelity/KL 为 0.2882/0.3239/2.096797，未达到
  TensorShield Top-10 或双黑盒。该停止规则读取 `eval_ms`，只能作为 post-hoc
  排名诊断，不能作为正式先验 selector。
  同协议的 16 项 `main_product` 扫描在 Top-7 首次反弹并选择 Top-6；Top-6 固定
  完整分类头并保护 `layer2.0/layer1.1/layer1.0/layer3.0/layer2.1/layer4.0` 的
  `conv1.weight`，保护比例 16.216624%，accuracy/fidelity/KL 为
  0.1519/0.1626/2.794303。三项均优于 TensorShield Top-10，但保护成本高于其
  8.9934%，且尚未达到 soft/hard 黑盒。
  另有 Main Conv 与对应 BN affine 的最小绑定实验：每个卷积增加对应
  `bn1.weight/bias`，不保护 BN 运行状态或 downsample。它同样在 Top-7 反弹并选择
  Top-6，保护比例 16.237144%，accuracy/fidelity/KL 为
  0.1579/0.1664/2.781295；相对纯 Conv Top-6 的 0.1519/0.1626/2.794303 三项回退。
  局部 affine 绑定不能替代 Lab04 的全部 20 个 BN gamma 跨层闭包。当前按要求不执行
  对应 BN affine 的十种子配对验证，也不保留该多 seed 入口或结果。
- `test/MS/02`：已在与 Test01 相同的 500 张 query 和 40 个 Conv/BN affine
  候选上计算“相同 victim 输入”的高斯二阶表征传输分数。分类头因
  ImageNet/CIFAR 类别坐标不对应而固定为私有边界，不进入 backbone 排名。
  Test02 与 Test01 的 Spearman/Kendall 为 `0.596998/0.458974`，Top-5/10 只重合
  `1/2` 项；`layer4.1.bn2` 在两种定义下均位于前五。但对称二阶能量归一化
  会把 `layer3.1` 的四个低能量算子推到前四，既没有聚集已有五个后验
  `conv1`，也无法表达全部 BN affine 的跨层闭包。因此当前单算子相对 RT 只是
  可解释的局部表征迁移量，还不是最终保护位置选择器。该测试同样不生成
  mask、不训练 surrogate，也不读取 `eval_ms`。

Lab 结果不能混入正式主实验索引，但也不能以“清理历史”为由删除仍承担独立结论的 Lab。

## 5. 当前卡在哪里

当前没有代码运行阻塞，`ResNet18+C100` 的 baseline 组合已经完成。真正未决的是本项目方法的设计与扩展边界：

1. 尚未形成能够先验选择攻击依赖 filter、跨层连通通道块或关键路径的统一算法；
   Lab09 已否定“统一优先选每个 BasicBlock 的 conv1”这一简单规则。
2. 尚未确定通道块大小、跨层成本归一化、保护预算扫描点和相同成本比较方式；Lab06 表明 BN gamma 应纳入通道块联动规则的候选状态，downsample Conv 应保留为图中候选。
3. `ResNet50`、`VGG16-BN`、`MobileNetV2` 尚未定义各自的基础 unit 与官方层映射。
4. TensorShield 缺少其他模型/数据集的作者 rank，不能自动扩展。
5. TEESlice 当前只实现 `ResNet18+C100`，其他组合需要单独适配 topology、训练和剪枝。

这些是研究和实现工作，不是通过复制当前 ResNet18 mask 就能解决的问题。

## 6. 下一步计划

### 第一阶段：在 ResNet18+C100 实现本项目方法

1. 先定义通道块保护的唯一权威 mask 表示，必须能够回溯到 122 个基础 state unit。
2. 明确定义卷积、BN、downsample 和分类头在通道块级别的联动规则。
3. 明确保护成本：至少记录受保护参数数、参数比例，并为后续 TEE 阶段预留 FLOPs/通信成本。
4. 固化若干绝对保护预算，不把比例作为训练入口的隐式参数。
5. 在正式运行前先更新最近一层中文 `README.md`，核对默认值、命令和协议完全一致。
6. 使用当前 500-query、固定 400/100 query 划分、soft posterior、最多 100 epoch、validation-best checkpoint 协议运行；hard 黑盒另用相同划分与 hard label。
7. 与浅层、中间层、深层、大权重、仅分类头和 TensorShield 在相近实际成本处比较。
8. 随机通道块、排序规则和块大小变化属于本方法消融，放在 Lab 或独立消融中，不要混成公开 baseline。

### 第二阶段：判断是否值得扩展

只有当通道块策略在 `ResNet18+C100` 上同时表现出更低 surrogate accuracy/fidelity、更高 KL，并且 victim utility 与保护成本可接受时，才进入 4x4 扩展。

### 第三阶段：扩展数据集和模型

1. 同一 ResNet18 扩展到 `c10/s10/t200` 时可复用数据与训练协议，但必须重新训练 victim、生成 posterior 和运行结果。
2. 新模型必须先建立自己的 state unit 与完整层映射，不能复用 ResNet18 的 122-unit mask。
3. `no/full/large_weight` 可优先扩展；完整层 baseline 在完成模型映射后扩展。
4. TensorShield 只在取得对应作者 rank 的组合展示。
5. TEESlice 只在完成结构忠实适配和独立上下界后展示，并始终标为 standalone。

### 最终绘图

当前三指标总图已经展示：

```text
普通预训练模型 no/full bounds
shallow / middle / deep / large_weight
head_only
TensorShield
TEESlice standalone point
```

后续把本项目通道块方法加入同一图。TEESlice 必须继续使用不同标记并注明 victim 结构不同，不能与普通 victim 曲线连成同一条成本曲线。

## 7. 绝对不要再踩的坑

1. **不要使用 `eval_ms` 选择正式 surrogate checkpoint。** 主结果只能使用 query validation loss 选择的最早最优 `best.pth`；`eval_ms` 在选模后只完整评估一次。
2. **不要把 surrogate 和 victim 的 best/end 混淆。** query 使用 victim `best.pth`；正式攻击使用 surrogate validation-best `best.pth`，且 surrogate 不再保存 `end.pth`。
3. **不要恢复 `target.pth`。** victim 使用 `best.pth`/`end.pth`；正式 surrogate
   只使用 validation-best `best.pth`。
4. **不要让 soft posterior 对应随机增强后的图像。** soft 模式只能使用确定性 test transform。
5. **不要为不同保护位置随意切换 hard/soft。** 正式主保护策略全部 posterior-visible soft；hard 只允许完整保护的 `hard_blackbox`。soft `full_protection` 与 hard `hard_blackbox` 都必须作为正式黑盒参考绘制。
6. **不要按测试集最优结果选择 frozen/finetune 或 replace/adapter。** 当前正式协议已固定为全模型 finetune；分类头按可见性使用 exposed、mixed 或 replace，不再逐实验选择。
7. **不要在分类头部分保护时丢弃整个头。** 必须逐标量混合复制，否则保护成本被低估。
8. **不要重新计算 TensorShield rank。** 只使用作者给出的 `ResNet18+C100` 固定列表；不要恢复已删除的 importance 公式代码。
9. **不要把 TensorShield rank 迁移到其他组合。** rank 与模型、数据集和 victim 有关。
10. **不要把 L1-L7 当成 ResNet18 官方层或新 unit。** 基础是 122 state unit，完整层是官方 18 层。
11. **不要把完整层 baseline 描述成通道块保护。** shallow/middle/deep 选择的是完整层。
12. **不要把大权重理解成 tensor 范数。** 它是全局逐标量绝对值排序。
13. **不要把 TEESlice 加回普通 victim `metrics.tsv`。** 它改变 victim 结构，只能 standalone。
14. **不要用 `eval_ms` 调整 TEESlice 剪枝拓扑。** 动态剪枝只看 victim_train 内部验证集。
15. **不要把 TEESlice 私有状态复制给黑盒攻击者。** 黑盒只知道 topology 和公开 backbone；完整状态只属于白盒。
16. **不要只改训练轮数而保留不匹配的 scheduler。** 修改正式协议意味着旧权重、结果和索引全部失效，必须清理并重跑。
17. **不要保留 v1/v2/old/backup/日期目录。** 同语义实现和结果直接覆盖或删除，历史交给 Git。
18. **不要保留失效脚本、结果、缓存或兼容入口。** 但仍承担结论的 Lab 不能误删。
19. **不要新增冗长目录名。** 遵守 `AGENTS.md` 的目录层级和单目录最多一个下划线约束。
20. **不要新增或修改实验后漏掉文档。** `README.md` 必须中文；涉及实验结构或流转时同步 `STRUCTURE.md` 与 `FLOW.md`。
21. **不要先跑正式实验再补协议。** 必须先固化 README，再核对代码默认值和命令，最后运行。
22. **不要创建独立 query 文件或 ratio/seed/run 子目录。** 元数据写入 manifest/params，目录保持稳定。

## 8. 关键路径与命令

### 文档

```text
AGENTS.md
README.md
STRUCTURE.md
FLOW.md
exp/MS/train_surrogate/README.md
exp/MS/train_surrogate/teeslice/README.md
results/MS/README.md
```

### 数据与权重

```text
dataset/public/                              原始公开数据
dataset/MS/<dataset>/                        MS split manifest
dataset/MS/c100/resnet18/                    普通 victim query 输出
dataset/MS/c100/teeslice_r18/                TEESlice query 输出
weights/pre_train/                           官方 ImageNet 权重
weights/MS/victim/                           victim checkpoint
weights/MS/surrogate/                        surrogate checkpoint 与 mask
results/MS/resnet18/c100/metrics.tsv         普通固定 victim 主索引
results/MS/resnet18/c100/teeslice/            TEESlice standalone 结果
```

### 验证

```bash
make env
make gpu
make unit
make verify
make results
```

TSDP 只认 `~/venvs/dl-py310-torch210-cu121`。`requirements.txt` 保存直接依赖，`requirements.lock.txt` 保存完整解析版本；不得退回系统 `/usr/bin/python3`。`make gpu` 会严格核对 WSL GPU 桥接并运行真实 CUDA 前向和反向计算，正式实验前必须通过。当前 `make unit` 应通过 43 个测试，`make results` 应通过正式 MS、Lab02-10 与 Test01/02 的协议、指标、history、mask、来源哈希及图片核对。运行后清理任何意外生成的 `__pycache__` 和 `*.pyc`。

### 普通 baseline

```bash
bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense head_only \
  --budget 500 --training-mode finetune --label-mode soft

bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense tensorshield \
  --budget 500 --training-mode finetune --label-mode soft

"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_surrogate/sweep.py layers --jobs 4
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_surrogate/sweep.py large_weight --jobs 4
```

### TEESlice

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_victim/teeslice/train.py resnet18 c100
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/transfer/get_label.py teeslice_r18 c100
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_surrogate/teeslice/attack.py resnet18 c100 \
  --budget 500 --training-mode finetune --label-mode soft
```

## 9. 外部参考仓库

当前服务器上参考代码位于 TSDP 同级目录：

```text
../Demo/TEESlice-artifact/
../Demo/TensorShield/
```

这些目录不是 TSDP Git 仓库的一部分。迁移到新机器后如需重新核对原实现，应单独复制或重新获取。TSDP 正式实验不能在运行时依赖 Demo 目录；作者固定 rank 已固化在 TSDP 内部。
