# 实验 05：State 与参数语义保护对比

本实验在 `ResNet18+CIFAR-100` 上分别只保护一种 `state_dict` 条目类型或参数语义组，观察不同状态不可见时的 MS 效果。各组都是相互独立的保护方案，不构成累计保护序列。

## 固定协议

```text
数据划分          dataset/MS/c100/manifest.json 中的 query_pool_ms 与 eval_ms
victim            weights/MS/victim/resnet18/c100/best.pth
surrogate 初始化  formal_victim_then_public_v1：ImageNet-1K backbone + 固定随机分类头
攻击者可观测输出  victim soft posterior
query transform   确定性的 test transform
query budget      500，即 CIFAR-100 训练集的 1%
query 划分        seed 42、offset 100 固定拆为 400 train / 100 validation
保护组            五种完整 state 类型，加十三种参数语义组
保护语义          只保留所选组的公开初始化，其余 victim 状态全部复制
分类头            严格按 mask 混合复制，不额外隐藏未保护类型
训练方式          全部 surrogate 参数共同微调；BN buffer 按正常 train 语义更新
训练轮数          100
优化器            SGD，lr=0.01，momentum=0.5，weight_decay=5e-4
学习率调度        StepLR，step_size=60，gamma=0.1
选模               validation soft cross-entropy 最低的最早 epoch
主要评估点        checkpoint 固定后只评估一次完整 eval_ms
原始指标          surrogate accuracy、fidelity、posterior KL
随机种子          42
```

每组均把当前实验 seed 传给共享 canonical 初始化器；public surrogate 初态与正式 MS 入口及 Lab04/Lab06 使用同一构造轨迹，不依赖此前的 RNG 消耗。

完整 state 类型使用 `state_dict` 名称最后一个字段精确匹配：

```text
weight
bias
running_mean
running_var
num_batches_tracked
```

其中 `weight` 同时包含 Conv weight、BN gamma 和分类头 weight；`bias` 包含 BN beta 和分类头 bias。当前 ResNet18 的 Conv 均使用 `bias=False`。

十三种参数语义组为：

```text
main_conv          16 个残差主路径 Conv weight，不含 Stem 和 downsample
stem_conv           1 个 Stem Conv weight
downsample_conv     3 个 downsample Conv weight
bn_gamma           20 个 BN weight
bn_beta            20 个 BN bias
bn_affine          全部 BN gamma 与 beta
stem_bn_affine     首个 BN 的 gamma 与 beta
downsample_bn_affine
                   三个 downsample BN 的 gamma 与 beta
head_weight        last_linear.weight
head_bias          last_linear.bias
head               last_linear.weight 与 last_linear.bias
downsample_branch  三个完整 downsample 分支的 Conv、BN 参数与 BN buffer
stem_branch        Stem Conv 和首个 BN 的参数与 BN buffer
```

这些语义组允许重叠。例如 `bn_affine` 是 `bn_gamma` 与 `bn_beta` 的并集，`head` 是 `head_weight` 与 `head_bias` 的并集，`downsample_branch` 也包含对应 downsample BN 的 gamma、beta 和 buffer。重叠组用于回答不同的独立问题，不能沿横轴解释为累计保护。

当只保护 `weight` 或 `head_weight` 时，分类头 weight 使用公开模型替换 C100 分类头后的随机初始化，分类头 bias 从 victim 复制；只保护 `bias` 或 `head_bias` 时则相反。只有显式选中的分类头状态不可见，不额外绑定 weight 与 bias。

`downsample_branch` 和 `stem_branch` 把对应 BN buffer 纳入完整执行状态；这不预设 BN buffer 单独具有持久保护作用，而是用于比较单个 weight 与完整计算图分支的差异。

## 成本口径

`running_mean`、`running_var` 和 `num_batches_tracked` 不是可训练参数，因此不能使用现有 `protected_param_ratio` 作为唯一横坐标。本实验保存四种互补比例：

```text
protected_unit_ratio           受保护 state tensor 数量 / 122
protected_param_ratio          受保护可训练标量 / 全部可训练标量
protected_state_element_ratio  受保护 state 标量数 / 全部 state 标量数
protected_state_byte_ratio     受保护 tensor payload 字节数 / 全部 state payload 字节数
```

三张图统一使用 `protected_state_byte_ratio` 作为横坐标，并采用对数刻度。各保护组只绘制独立散点，不把相互重叠的语义组连接成曲线；图中同时绘制 no-protection 白盒、soft full-protection 黑盒和 hard-label 黑盒参考线。

## 运行方式

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/05_state/run.py
```

运行前只核对十八组 mask、统计与输入，不写结果：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/05_state/run.py --dry-run
```

完整运行会覆盖同一语义入口，十八组均独立按统一协议训练。

## 输出

```text
results/lab/05_state/metrics.json               十八组协议、保护统计、选模与单次 eval_ms 指标
results/lab/05_state/history.tsv                十八组各 100 轮 query train/validation 记录
results/lab/05_state/data.tsv                   绘图使用的正式原始点
results/lab/05_state/accuracy.png               保护存储比例与 accuracy
results/lab/05_state/fidelity.png               保护存储比例与 fidelity
results/lab/05_state/posterior_kl.png            保护存储比例与 posterior KL
results/lab/05_state/<group>_mask.pt             各组的 122-unit 紧凑保护掩码
```
