# 实验 04：TensorShield Rank 前缀曲线

本实验在 `ResNet18+CIFAR-100` 上依次保护 TensorShield 作者确认 eligible rank 的 Top-1 至 Top-17，测量 MS accuracy、fidelity 与 posterior KL 随前缀扩大产生的变化。实验用于判断作者最终 Top-10 是否位于稳定的前缀收敛区间，并观察全部 eligible tensor 的累计作用。曲线横轴统一使用实际保护参数比例，各点标注对应的 Top-k 编号。

前缀曲线只能证明这组有序前缀的累计效果，不能单独证明每个 tensor 的相对次序优于所有同规模替代集合。若要检验全局排序准确性，还需要在相同 k 下增加随机集合或低排名替换对照。

## Rank 定义

输入是 TensorShield 作者确认用于论文 Figure 12 的 41-weight rank。按照论文最终候选规则排除 BatchNorm、downsample 和 attention transition 排除的 `conv1.weight` 后，得到 17 个 eligible weight。该顺序的前 10 项与 Figure 12(d) 发布的 10-weight 集合完全相同；Figure 12 按网络位置展示集合，不表示 importance 顺序。

本实验使用全部 17 项，并在每个前缀中固定加入 unit 121 `last_linear.bias`：

```text
rank  新增 weight                    unit   累计 unit  累计参数    累计比例
1     layer1.1.conv1.weight          18          2      36,964     0.3292%
2     layer2.0.conv1.weight          30          3     110,692     0.9859%
3     last_linear.weight            120          4     161,892     1.4419%
4     layer1.0.conv1.weight           6          5     198,756     1.7702%
5     layer1.1.conv2.weight          24          6     235,620     2.0985%
6     layer2.0.conv2.weight          36          7     383,076     3.4118%
7     layer2.1.conv1.weight          48          8     530,532     4.7252%
8     layer1.0.conv2.weight          12          9     567,396     5.0535%
9     layer3.0.conv1.weight          60         10     862,308     7.6801%
10    layer2.1.conv2.weight          54         11   1,009,764     8.9934%
11    layer3.0.conv2.weight          66         12   1,599,588    14.2467%
12    layer4.0.conv1.weight          90         13   2,779,236    24.7531%
13    layer4.0.conv2.weight          96         14   5,138,532    45.7661%
14    layer4.1.conv1.weight         108         15   7,497,828    66.7791%
15    layer4.1.conv2.weight         114         16   9,857,124    87.7920%
16    layer3.1.conv2.weight          84         17  10,446,948    93.0453%
17    layer3.1.conv1.weight          78         18  11,036,772    98.2985%
```

unit 121 在所有 k 中固定保护。`last_linear.weight` 从 k=3 起进入前缀，因此 k=1、k=2 是分类头 weight 暴露、bias 保护的 `mixed` 控制；k=3 至 k=17 同时保护分类头 weight 与 bias，使用 `replace`。该 mixed-head 处理只存在于 Lab04，不改变正式 `exp/MS/train_surrogate` 中 custom unit 必须成对选择分类头 weight/bias 的约束。

## 固定协议

```text
数据划分          dataset/MS/c100/manifest.json 中的 query_pool_ms 与 eval_ms
victim            weights/MS/victim/resnet18/c100/best.pth
surrogate 初始化  ImageNet-1K 官方预训练 ResNet18
攻击者可观测输出  victim soft posterior
query transform   确定性的 test transform
query budget      500，即 CIFAR-100 训练集的 1%
保护策略          作者确认 rank 的 eligible Top-k，k=1,...,17；每组固定加入 unit 121
暴露状态          从 victim 复制；保护状态保留公开预训练/随机初始化值
分类头            k=1、2 暴露 weight、保护 bias；k=3 起 weight 与 bias 均保护
训练方式          所有 surrogate 参数共同微调，不冻结暴露权重
训练轮数          100
优化器            SGD，lr=0.01，momentum=0.5，weight_decay=5e-4
学习率调度        StepLR，step_size=60，gamma=0.1
主要评估点        第 100 轮 end；不使用 eval_ms 选点或选择 k
原始指标          surrogate accuracy、fidelity、posterior KL
随机种子          每个 k 均重置为 42
```

为避免把 `eval_ms` 用作训练过程中的选择信号，每个 k 只在 100 轮训练结束后评估一次。无保护和全保护不重复训练，只作为图中的当前正式参考线。Lab 输出不写入正式 `results/MS` 索引，也不保存 surrogate checkpoint。

## 运行方式

先验证 rank、17 个 mask、参数量与分类头模式：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/04_tensorshield/run.py --dry-run
```

运行完整实验并覆盖旧 Lab 04 结果：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/04_tensorshield/run.py
```

## 输出

```text
results/lab/04_tensorshield/metrics.json       固定协议、rank、17 组保护统计与 end 指标
results/lab/04_tensorshield/history.tsv        17 组各 100 轮 query 训练记录
results/lab/04_tensorshield/data.tsv           绘图使用的 Top-k 原始指标
results/lab/04_tensorshield/accuracy.png       参数占比与 surrogate accuracy 断轴曲线，放大 0–15% 区间
results/lab/04_tensorshield/fidelity.png       参数占比与 fidelity 断轴曲线，放大 0–15% 区间
results/lab/04_tensorshield/posterior_kl.png   参数占比与 posterior KL 断轴曲线，放大 0–15% 区间
results/lab/04_tensorshield/top_01_mask.pt      Top-1 紧凑保护掩码
...
results/lab/04_tensorshield/top_17_mask.pt      Top-17 紧凑保护掩码
```

## Top-12 内 Rank-5/Rank-10 冗余消融

Top-12 已取得较好的 MS 防护效果；为判断 rank-5 `layer1.1.conv2.weight` 和 rank-10 `layer2.1.conv2.weight` 在该集合中是否仍有独立保护作用，使用以下四组集合消融：

```text
full_top12   作者 eligible rank 的完整 Top-12
drop_05      从完整 Top-12 删除 rank-5
drop_10      从完整 Top-12 删除 rank-10
drop_05_10   从完整 Top-12 同时删除 rank-5 和 rank-10
```

四组均保留 rank-3 分类头 weight，并同步保护 bias，因此分类头模式全部为 `replace`。`full_top12` 直接读取当前 `metrics.json` 中使用相同初始化协议得到的 Top-12；删除 rank-5、删除 rank-10 和同时删除两者三组重新训练。每组在 surrogate 初始化前重置相同随机状态，并使用与前缀曲线一致的 query 顺序、优化器、100 轮训练和 end-only 评估。图中同时显示统一的白盒 `no_protection` 与黑盒 `full_protection` 边界。

该消融比较的是从完整 Top-12 删除 tensor 后的条件贡献。若删除后 accuracy/fidelity 基本不升、KL 基本不降，只能说明该 tensor 在当前集合和攻击协议下存在功能冗余；不能推导它在其他 k、模型或数据集上始终无用。

先验证四组集合、mask 和复用输入：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/04_tensorshield/ablate.py --dry-run
```

运行三组新增训练并覆盖同语义消融结果：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/04_tensorshield/ablate.py
```

新增输出为：

```text
results/lab/04_tensorshield/ablation.json          四组集合、黑白盒边界、保护统计与 end 指标
results/lab/04_tensorshield/ablation.tsv           相对完整 Top-12 的原始差值
results/lab/04_tensorshield/ablation_history.tsv   三组新增训练共 300 轮 query 记录
results/lab/04_tensorshield/ablation.png           三项 MS 指标及黑白盒边界的集合消融对比
results/lab/04_tensorshield/drop_05_mask.pt         删除 rank-5 的 mask
results/lab/04_tensorshield/drop_10_mask.pt         删除 rank-10 的 mask
results/lab/04_tensorshield/drop_05_10_mask.pt      同时删除 rank-5/rank-10 的 mask
```

## Eligible rank 窗口消融

从 17 个 eligible weight 中排除 `last_linear.weight`，得到 16 个非分类头候选。窗口实验选择该顺序的前 10 和后 10，并为两组固定加入 `last_linear.weight` 与 `last_linear.bias`，从而只比较非分类头候选位置，避免分类头状态成为混杂因素：

```text
窗口       候选位置   保护的 10 个非分类头 eligible weight
first_10   1-10       layer1.1.conv1.weight, layer2.0.conv1.weight,
                      layer1.0.conv1.weight, layer1.1.conv2.weight,
                      layer2.0.conv2.weight, layer2.1.conv1.weight,
                      layer1.0.conv2.weight, layer3.0.conv1.weight,
                      layer2.1.conv2.weight, layer3.0.conv2.weight
last_10    7-16       layer1.0.conv2.weight, layer3.0.conv1.weight,
                      layer2.1.conv2.weight, layer3.0.conv2.weight,
                      layer4.0.conv1.weight, layer4.0.conv2.weight,
                      layer4.1.conv1.weight, layer4.1.conv2.weight,
                      layer3.1.conv2.weight, layer3.1.conv1.weight
```

两组均额外保护分类头 weight 与 bias，并使用 `replace`。每组只训练一次，均沿用前缀曲线的 CIFAR-100 数据划分、500 条 soft posterior query、确定性 test transform、公开预训练初始化、种子 42、全参数微调、100 轮训练和 end-only 评估。

两组都选择 10 个非分类头候选并使用相同分类头控制，但 tensor 尺寸不同，因此并非等参数成本比较。结果使用三联直方图，横轴类别标签显示实际保护参数比例，同时报告原始三项 MS 指标。

先验证两个窗口、mask 和分类头模式：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/04_tensorshield/window.py --dry-run
```

运行两组各一次训练并覆盖同语义结果：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/04_tensorshield/window.py
```

新增输出为：

```text
results/lab/04_tensorshield/window.json          两个 eligible 窗口、保护统计与 end 指标
results/lab/04_tensorshield/window.tsv           两组原始指标与保护成本
results/lab/04_tensorshield/window_history.tsv   两组各 100 轮 query 训练记录
results/lab/04_tensorshield/window.png           两个窗口的三项 MS 指标三联直方图
results/lab/04_tensorshield/first_10_mask.pt      前 10 候选加分类头的紧凑保护掩码
results/lab/04_tensorshield/last_10_mask.pt       后 10 候选加分类头的紧凑保护掩码
```
