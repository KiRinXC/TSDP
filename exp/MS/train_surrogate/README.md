# surrogate 训练入口

本目录实现 MS baseline 的 surrogate 初始化、伪标签训练和 `eval_ms` 评估。所有运行都读取 victim 的 `best.pth`，并使用 `query_pool_ms` 固定顺序的预算前缀。

内部代码分为两部分：`core/` 保存数据读取、训练、评估和产物写入等公共流程；`defense/` 保存保护策略插件、掩码协议、模型初始化与冻结机制。`train.py` 是唯一正式入口，所有策略共用同一套训练和评估代码。

```text
train_surrogate/
├── train.py              统一入口
├── run.sh                命令行封装
├── core/                 公共训练、评估和产物管理
└── defense/              策略注册、mask、unit 映射和初始化
```

## 保护策略

```text
no_protection    所有 victim 权重暴露，作为攻击上界
full_protection  所有 victim 权重不可见，作为纯黑盒下界
shallow          从官方第 1 层开始保护连续完整层
middle           保护不接触首尾的连续完整官方层
deep             连续保护到官方最后一层
custom           直接指定任意连续或离散 unit
large_weight     按公开预训练权重绝对值保护全局大权重标量
```

ResNet18 以 `state_dict` 的稳定顺序定义 122 个基础 tensor unit，索引范围为 `0` 至 `121`。每个 unit 对应一个参数或缓冲区条目；后续所有完整层和细粒度策略均由这些 unit 聚合，不再另外定义底层保护单位。

下表给出 ResNet18 官方 18 层与 unit 的归属关系，用于在需要完整层边界时选择 unit：`conv1` 为第 1 层，8 个 BasicBlock 中的 16 个主分支卷积为第 2 至 17 层，`last_linear` 为第 18 层。BatchNorm 状态归入前置卷积；三个 downsample 分支归入对应 stage 首个 BasicBlock 的 `conv1` 层；`maxpool` 关联第 1 层，`avgpool` 关联第 18 层。池化层没有 `state_dict` 条目，因此不会增加 unit 数量。训练入口只把明确指定的官方层映射为 unit，不会把比例隐式转换成 unit。

```text
官方层  主模块                   unit 索引
1       conv1                    0-5
2       layer1.0.conv1           6-11
3       layer1.0.conv2           12-17
4       layer1.1.conv1           18-23
5       layer1.1.conv2           24-29
6       layer2.0.conv1           30-35, 42-47
7       layer2.0.conv2           36-41
8       layer2.1.conv1           48-53
9       layer2.1.conv2           54-59
10      layer3.0.conv1           60-65, 72-77
11      layer3.0.conv2           66-71
12      layer3.1.conv1           78-83
13      layer3.1.conv2           84-89
14      layer4.0.conv1           90-95, 102-107
15      layer4.0.conv2           96-101
16      layer4.1.conv1           108-113
17      layer4.1.conv2           114-119
18      last_linear              120-121
```

保护方案不接收比例。`shallow`、`middle` 和 `deep` 使用 `--protected-layers` 指定 1-based 官方完整层，例如 `1-3`、`8-11`、`16-18`。入口会把完整层映射为 unit；downsample 随所属官方层一起选择，因此最终 unit 不要求连续。也可以通过 `--protected-units` 提供映射后的精确 unit，但必须恰好组成符合策略方向的完整官方层。任意细粒度连续或离散 unit 使用 `custom`，例如 `3,6,9`。`no_protection` 和 `full_protection` 自动生成空集合与全集。分类头的 `last_linear.weight` 和 `last_linear.bias` 必须同时保护或同时暴露。

每种策略最终统一生成一个 `protection_mask.pt`。文件按模型 `state_dict` 顺序包含一个 unit mask：`True` 表示对应状态不可见，`False` 表示可被攻击者复制。完整 unit 使用紧凑的 `all` 或 `none` 标记；只保护部分标量时保存压缩位图。该掩码是保护方案的唯一权威表示，JSON 只记录掩码路径、SHA256 和统计量，不重复保存层列表或 unit 索引列表。

TensorShield 论文 Figure 12 将同一批索引 `0` 至 `121` 画成 L1 至 L7 七段，用于展示 Serdab 和 DarkneTZ 的四层选择。L1 至 L7 是该图对网络位置的展示分段，不是额外的 tensor unit，也不是 ResNet18 官方的深度定义；本实验不会用这七段替代官方 18 层。

`large_weight` 忠实采用逐标量混合：通过 `--protected-scalars` 指定绝对标量数量，对 Conv、BatchNorm 和 Linear 的公开预训练 `weight` 绝对值进行全局排序，排名靠前的标量保持公开初始化，其余标量复制 victim。BatchNorm 的 bias 和运行统计量沿用对应 weight 的掩码。

新增保护策略时，在 `defense/` 中实现 mask builder，并注册到 `DEFENSE_REGISTRY`。builder 只负责返回 victim 状态空间中的布尔掩码、分类头模式和策略统计量；公共初始化器负责组合公开权重与暴露的 victim 权重，训练器不包含具体策略分支。

## 初始化与训练

未受保护的状态直接从 victim `best.pth` 复制。受保护的 backbone 状态保留官方 ImageNet 初始化。

分类头未受保护时直接复制 victim 分类头；完整分类头受保护时保留 ImageNet-1K 分类头，并追加映射到当前数据集类别数的适配层。`large_weight` 需要在同形状张量内混合标量，因此分类头使用同形状的 `scalar_mix`，该例外会写入 `head_mode`。

`--training-mode frozen` 只训练未从 victim 暴露的状态；`finetune` 允许所有状态共同训练。逐标量冻结会在每次优化后恢复暴露权重，避免 weight decay 和 momentum 改写它们。

除 `full_protection` 外默认使用 `posteriors.pt` 的 soft posterior 训练，hard pseudo label 可通过 `--label-mode hard` 作为消融配置启用。`full_protection` 模拟只能查询输入和输出类别的黑盒，强制使用 `--label-mode hard`，训练数据管线只读取 `labels.tsv`，不会加载 `posteriors.pt`。

## 运行方式

运行一个 `ResNet18+C100` 配置：

```bash
bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense shallow \
  --protected-layers 1-9 \
  --budget 500 \
  --training-mode frozen \
  --label-mode soft
```

其他 unit 选择方式：

```bash
bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense deep --protected-layers 10-18 \
  --budget 500 --training-mode frozen --label-mode soft

bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense middle --protected-layers 8-11 \
  --budget 500 --training-mode frozen --label-mode soft

bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense custom --protected-units 3,6,9 \
  --budget 500 --training-mode frozen --label-mode soft

bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense large_weight --protected-scalars 100000 \
  --budget 500 --training-mode frozen --label-mode soft
```

正式 baseline 使用各数据集 manifest 中训练集 1% 对应的 `max_budget`：`c10=500`、`c100=500`、`s10=50`、`t200=1000`。较小预算仅用于后续 Lab 中的 query budget 敏感性实验。

不同配置根据输入、victim checkpoint 和训练参数生成稳定的短 `run_id`，不会把预算、比例或随机种子编码进目录名。

## 输出

```text
weights/MS/surrogate/<model>/<dataset>/<run_id>/
  best.pth                surrogate_acc 最高的 checkpoint
  end.pth                 最后一个 epoch 的 checkpoint
  protection_mask.pt      当前策略的 unit 保护掩码
  params.json             输入、训练参数和掩码摘要
  train.log.tsv           每个 epoch 的训练观测与 eval_ms 原始指标

results/MS/<model>/<dataset>/
  metrics.tsv             同一模型和数据集下所有 run 的原始指标索引
  <run_id>/metrics.json   best 与 end 的完整原始指标
```

`best.pth` 按 `surrogate_acc` 选择。`metrics.json` 同时保留 best 和 end，冻结与微调运行也分别保留，不在训练阶段删除较弱配置。

正式保存的 MS 指标为：

```text
eval_count
victim_correct
surrogate_correct
agreement_count
victim_acc
surrogate_acc
fidelity
posterior_kl_sum
posterior_kl
```

准确率下降、fidelity 下降、相对黑盒倍数和归一化保护效果均不写入正式结果，由后续绘图脚本从这些原始值计算。
