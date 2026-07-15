# 实验 04：TensorShield Rank 前缀曲线

本实验在 `ResNet18+CIFAR-100` 上依次保护 TensorShield 作者确认 rank 的 Top-1 至 Top-12，测量 MS accuracy、fidelity 与 posterior KL 随前缀扩大产生的变化。实验用于判断作者最终 Top-10 是否位于稳定的前缀收敛区间，并观察每个新增 tensor 的边际作用。

前缀曲线只能证明这组有序前缀的累计效果，不能单独证明每个 tensor 的相对次序优于所有同规模替代集合。若要检验全局排序准确性，还需要在相同 k 下增加随机集合或低排名替换对照。

## Rank 定义

输入是 TensorShield 作者确认用于论文 Figure 12 的 41-weight rank。按照论文最终候选规则排除 BatchNorm、downsample 和 attention transition 排除的 `conv1.weight` 后，得到 17 个 eligible weight。该顺序的前 10 项与 Figure 12(d) 发布的 10-weight 集合完全相同；Figure 12 按网络位置展示集合，不表示 importance 顺序。

本实验使用前 12 项：

```text
rank  新增 weight                    unit   累计 unit  累计参数    累计比例
1     layer1.1.conv1.weight          18          1      36,864     0.3283%
2     layer2.0.conv1.weight          30          2     110,592     0.9850%
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
```

`last_linear.weight` 从 k=3 起进入前缀。分类头 weight 与 bias 必须联动保护，因此 k=3 至 k=12 的累计 unit 和参数量均包含 unit 121 `last_linear.bias`；k=1、k=2 的分类头完整暴露。

## 固定协议

```text
数据划分          dataset/MS/c100/manifest.json 中的 query_pool_ms 与 eval_ms
victim            weights/MS/victim/resnet18/c100/best.pth
surrogate 初始化  ImageNet-1K 官方预训练 ResNet18
攻击者可观测输出  victim soft posterior
query transform   确定性的 test transform
query budget      500，即 CIFAR-100 训练集的 1%
保护策略          作者确认 rank 的 eligible Top-k，k=1,...,12
暴露状态          从 victim 复制；保护状态保留公开预训练/随机初始化值
分类头            k=1、2 完整暴露；k=3 起 weight 与 bias 联动保护并随机初始化
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

先验证 rank、12 个 mask、参数量与分类头模式：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/04_tensorshield/run.py --dry-run
```

运行完整实验并覆盖旧 Lab 04 结果：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/04_tensorshield/run.py
```

## 输出

```text
results/lab/04_tensorshield/metrics.json       固定协议、rank、12 组保护统计与 end 指标
results/lab/04_tensorshield/history.tsv        12 组各 100 轮 query 训练记录
results/lab/04_tensorshield/data.tsv           绘图使用的 Top-k 原始指标
results/lab/04_tensorshield/metrics.png        accuracy、fidelity、KL 三联曲线
results/lab/04_tensorshield/top_01_mask.pt      Top-1 紧凑保护掩码
...
results/lab/04_tensorshield/top_12_mask.pt      Top-12 紧凑保护掩码
```

## Rank-5/Rank-10 冗余消融

前缀曲线显示，加入 rank-5 `layer1.1.conv2.weight` 和 rank-10 `layer2.1.conv2.weight` 时，MS 指标变化较小。为判断这两个高排名 tensor 是否在完整 Top-10 集合中仍然提供独立保护作用，增加以下四组集合消融：

```text
full_top10   作者 eligible rank 的完整 Top-10
drop_05      从完整 Top-10 删除 rank-5
drop_10      从完整 Top-10 删除 rank-10，等价于已有 Top-9
drop_05_10   从完整 Top-10 同时删除 rank-5 和 rank-10
```

四组均保留 rank-3 分类头 weight，并同步保护 bias，因此分类头模式全部为 `replace`。`full_top10` 和 `drop_10` 直接读取当前 `metrics.json` 中使用相同初始化协议得到的 Top-10/Top-9；只新增训练 `drop_05` 与 `drop_05_10`，每组在 surrogate 初始化前重置相同随机状态，并使用与前缀曲线一致的 query 顺序、优化器、100 轮训练和 end-only 评估。

该消融比较的是从完整集合删除 tensor 后的条件贡献。若删除后 accuracy/fidelity 基本不升、KL 基本不降，只能说明该 tensor 在当前集合和攻击协议下存在功能冗余；不能推导它在其他 k、模型或数据集上始终无用。

先验证四组集合、mask 和复用输入：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/04_tensorshield/ablate.py --dry-run
```

运行两组新增训练并覆盖同语义消融结果：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/04_tensorshield/ablate.py
```

新增输出为：

```text
results/lab/04_tensorshield/ablation.json          四组集合、保护统计与 end 指标
results/lab/04_tensorshield/ablation.tsv           相对完整 Top-10 的原始差值
results/lab/04_tensorshield/ablation_history.tsv   两组新增训练共 200 轮 query 记录
results/lab/04_tensorshield/ablation.png           三项 MS 指标的集合消融对比
results/lab/04_tensorshield/drop_05_mask.pt         删除 rank-5 的 mask
results/lab/04_tensorshield/drop_05_10_mask.pt      同时删除 rank-5/rank-10 的 mask
```

## 原始 rank 窗口消融

为观察作者完整排序中部和末尾 tensor 的 MS 保护效果，直接基于作者提供的原始 41-weight rank 选取两个窗口。这里不应用前述 Figure 12 候选筛选，也不逐个扩展窗口：

```text
窗口          原始 rank 位置   保护的 10 个 ranked weight
rank_11_20    11-20            layer3.0.bn1.weight, layer3.0.bn2.weight,
                                layer3.0.downsample.1.weight, conv1.weight,
                                layer2.0.downsample.0.weight, layer4.0.bn1.weight,
                                layer1.1.conv1.weight, layer2.0.conv1.weight,
                                last_linear.weight, layer1.0.conv1.weight
rank_32_41    32-41            layer3.0.conv2.weight, layer3.1.bn2.weight,
                                layer4.1.bn2.weight, layer4.0.conv1.weight,
                                layer3.1.bn1.weight, layer4.0.conv2.weight,
                                layer4.1.conv1.weight, layer4.1.conv2.weight,
                                layer3.1.conv2.weight, layer3.1.conv1.weight
```

`rank_11_20` 包含 `last_linear.weight`，因此联动保护 `last_linear.bias` 并使用 `replace`；`rank_32_41` 不包含分类头，分类头完整暴露、从 victim 复制。每组只训练一次，均沿用前缀曲线的 CIFAR-100 数据划分、500 条 soft posterior query、确定性 test transform、公开预训练初始化、种子 42、全参数微调、100 轮训练和 end-only 评估。

两组都选择 10 个原始 ranked weight，但 tensor 尺寸和分类头状态不同，因此必须同时报告保护参数量、参数比例和 `head_mode`。该消融用于观察两个指定 rank 窗口的实际攻击结果，不是等成本的纯排序检验。

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
results/lab/04_tensorshield/window.json          两个窗口、保护统计与 end 指标
results/lab/04_tensorshield/window.tsv           两组原始指标与保护成本
results/lab/04_tensorshield/window_history.tsv   两组各 100 轮 query 训练记录
results/lab/04_tensorshield/window.png           两个窗口的三项 MS 指标对比
results/lab/04_tensorshield/rank_11_20_mask.pt    rank 11-20 紧凑保护掩码
results/lab/04_tensorshield/rank_32_41_mask.pt    rank 32-41 紧凑保护掩码
```
