# 实验 04 结果

本目录保存 TensorShield 作者确认 eligible rank 的 Top-1 至 Top-17 MS 前缀曲线，以及 rank-5/rank-10 冗余消融和 eligible rank 位置集合对照。三个实验均使用 `formal_victim_then_public_v1`、seed 42，并在每个独立 surrogate 初始化前重置随机状态。

## Top-1 至 Top-17 前缀曲线

每个 Top-k 均固定保护 unit 121 `last_linear.bias`。Top-1、Top-2 的分类头模式为 `mixed`；Top-3 起同时保护分类头 weight 与 bias，分类头模式为 `replace`。

```text
k   新增 weight                    参数比例   accuracy  fidelity  posterior KL
1   layer1.1.conv1.weight           0.3292%     0.5980    0.8153       0.167875
2   layer2.0.conv1.weight           0.9859%     0.5499    0.6977       0.447697
3   last_linear.weight              1.4419%     0.3493    0.3923       1.757093
4   layer1.0.conv1.weight           1.7702%     0.3282    0.3681       1.844142
5   layer1.1.conv2.weight           2.0985%     0.3333    0.3721       1.822863
6   layer2.0.conv2.weight           3.4118%     0.3011    0.3331       1.975075
7   layer2.1.conv1.weight           4.7252%     0.2727    0.2992       2.094104
8   layer1.0.conv2.weight           5.0535%     0.2732    0.2992       2.101054
9   layer3.0.conv1.weight           7.6801%     0.1984    0.2160       2.479365
10  layer2.1.conv2.weight           8.9934%     0.1915    0.2106       2.505739
11  layer3.0.conv2.weight          14.2467%     0.1757    0.1910       2.607458
12  layer4.0.conv1.weight          24.7531%     0.1655    0.1757       2.690443
13  layer4.0.conv2.weight          45.7661%     0.1600    0.1732       2.707384
14  layer4.1.conv1.weight          66.7791%     0.1611    0.1732       2.714185
15  layer4.1.conv2.weight          87.7920%     0.1617    0.1759       2.729092
16  layer3.1.conv2.weight          93.0453%     0.1633    0.1783       2.721452
17  layer3.1.conv1.weight          98.2985%     0.1650    0.1790       2.723661
```

当前正式参考线为：无保护 accuracy `0.6182`、fidelity `1.0000`、KL 约为 `0`；全保护 accuracy `0.1545`、fidelity `0.1610`、KL `2.835290`。

Top-10 的保护 mask 与 Figure 12(d) 固定集合一致，逻辑 SHA256 为 `1e3aa38124f084dd39eab42a4d3f1ddf1ca86807812796c66a8318c05e7aa2cb`。当前曲线和正式入口都使用 `formal_victim_then_public_v1`、seed 42、同一 query 顺序和训练协议。当前 RTX 4060 Laptop 上得到 `0.1915/0.2106/2.505739`，旧正式 CUDA 主机上的单点为 `0.1913/0.2099/2.505831`；两者只差 2 个 accuracy 样本、7 个 agreement 样本和约 `0.000091` KL。该微小差异属于跨 CUDA 硬件数值差异，不再是随机分类头轨迹不同。

曲线总体随保护范围扩大而增强，但不是严格单调。Top-3 加入 `last_linear.weight` 后，accuracy 相比 Top-2 降低 `20.06` 个百分点，fidelity 降低 `30.54` 个百分点，KL 增加 `1.309396`，说明早期最大跳变仍来自分类头 weight，而固定加入的 100 个 bias 参数没有消除这一现象。

Top-10 后仍存在改善：Top-11 和 Top-12 分别把 accuracy 降至 `0.1757` 和 `0.1655`。但 Top-12 之后进入高成本、低边际收益区间，参数比例从 `24.7531%` 增至 `98.2985%`，accuracy 只再降低 `0.05` 个百分点，fidelity 反而提高 `0.33` 个百分点，KL 增加 `0.033218`。Top-13 之后多次出现反向变化，说明 eligible rank 的细粒度次序不能由单次前缀曲线视为严格边际贡献排序。

Top-17 与全保护仍相差 `1.05` 个百分点 accuracy、`1.80` 个百分点 fidelity 和 `0.111629` KL。也就是说，在当前攻击协议下，保护全部 17 个 eligible weight 已非常接近全保护，但仍未严格达到全保护参考线。

```text
metrics.json       作者 rank、输入哈希、17 组保护统计与 end 原始指标
history.tsv       1,700 条 query 训练记录，不包含中途 eval_ms 指标
data.tsv          以参数占比为横轴的 Top-k 原始绘图数据
accuracy.png      参数占比与 surrogate accuracy 断轴曲线，放大 0–15% 区间
fidelity.png      参数占比与 fidelity 断轴曲线，放大 0–15% 区间
posterior_kl.png  参数占比与 posterior KL 断轴曲线，放大 0–15% 区间
top_01_mask.pt    Top-1 紧凑保护掩码
...
top_17_mask.pt    Top-17 紧凑保护掩码
```

## Top-12 内 Rank-5/Rank-10 冗余消融

该消融使用统一的先训练后分区边界：`no_protection` 为白盒边界，`full_protection` 为黑盒边界。

```text
边界                    accuracy  fidelity  posterior KL
白盒（no protection）     0.6182    1.0000       0.000000
黑盒（full protection）   0.1545    0.1610       2.835290
```

```text
方案          保护参数比例  accuracy  fidelity  posterior KL
完整 Top-12      24.7531%     0.1655    0.1757       2.690443
删除 rank-5      24.4248%     0.1635    0.1745       2.714643
删除 rank-10     23.4398%     0.1623    0.1753       2.704415
同时删除 5/10    23.1115%     0.1579    0.1705       2.718431
```

删除 rank-5 `layer1.1.conv2.weight` 后，accuracy 和 fidelity 分别降低 `0.20`、`0.12` 个百分点，KL 增加 `0.024200`。删除 rank-10 `layer2.1.conv2.weight` 后，accuracy 和 fidelity 分别降低 `0.32`、`0.04` 个百分点，KL 增加 `0.013972`。三个指标均指向删除任一项后当前 MS 攻击反而变弱。

同时删除两项后，accuracy 和 fidelity 分别降低 `0.76`、`0.52` 个百分点，KL 增加 `0.027988`；保护参数减少 `184,320`，相当于完整 Top-12 保护参数的 `6.63%`。因此该实验不能把 rank-5/rank-10 解释为当前 Top-12 集合中的正向安全贡献。更准确的结论是：在固定初始化和当前 finetune 攻击器下，保护范围与经验 MS 指标不保证单调；攻击者的优化路径会随公开/随机状态组合改变。

完整 Top-12 距黑盒仍差 `1.10` 个百分点 accuracy、`1.47` 个百分点 fidelity 和 `0.144847` KL；同时删除两项后对应差距为 `0.34`、`0.95` 个百分点和 `0.116859` KL。四组均未严格达到黑盒边界，且这里观察到的是单次固定攻击训练的经验非单调性，不能推出“少保护在信息论上更安全”。

```text
ablation.json          四组集合、黑白盒边界、相对完整 Top-12 的差值与输入哈希
ablation.tsv           可直接绘图和统计的原始指标
ablation_history.tsv   三组删除实验共 300 轮 query 训练记录
ablation.png           accuracy、fidelity、posterior KL 与黑白盒边界三联柱状图
drop_05_mask.pt        删除 rank-5 的紧凑保护掩码
drop_10_mask.pt        删除 rank-10 的紧凑保护掩码
drop_05_10_mask.pt     同时删除 rank-5/rank-10 的紧凑保护掩码
```

## Eligible rank 位置集合消融

从 17 个 eligible weight 中排除分类头 weight 后得到 16 个候选。三组均选 10 个非分类头候选，并额外保护 `last_linear.weight` 和 `last_linear.bias`；`spread_10` 固定选择候选位置 `1,2,3,5,7,9,11,13,15,16`。

```text
方案       候选位置                       非头候选  保护 unit  参数比例  head mode  accuracy  fidelity  posterior KL
first_10   1-10                                  10         12   14.2467%  replace      0.1757    0.1910       2.607458
spread_10  1,2,3,5,7,9,11,13,15,16              10         12   46.7511%  replace      0.2121    0.2342       2.419835
last_10    7-16                                  10         12   94.0303%  replace      0.2120    0.2283       2.420454
```

`first_10` 的 mask 等价于前缀曲线 Top-11，统一轨迹后两者三个指标逐值相同，这同时校验了两条 Lab 入口的初始化、数据顺序和 mask 语义。相较 `first_10`，`spread_10` 虽保护约 `3.28` 倍参数，但 accuracy 和 fidelity 分别高 `3.64`、`4.32` 个百分点，KL 低 `0.187622`；`last_10` 虽保护约 `6.60` 倍参数，accuracy 和 fidelity仍分别高 `3.63`、`3.73` 个百分点，KL 低 `0.187003`。因此 eligible rank 前部的 10 个候选明显强于分散或后部候选，保护效果不随参数成本单调增强。

`spread_10` 与 `last_10` 的 accuracy 几乎相同，前者 fidelity 高 `0.59` 个百分点、KL 低 `0.000619`。二者不存在值得解释为稳定排序的三指标优势，核心结论只应是它们均明显弱于 `first_10`。

三组之间存在候选重叠，且参数成本不同。结果仅来自当前模型、数据集、攻击协议和固定种子，因此不能视为等成本比较、单个候选的全局贡献排序或统计性证明。

```text
window.json          三个 eligible 位置集合、保护统计、输入哈希和 end 原始指标
window.tsv           三组保护成本与三项 MS 原始指标
window_history.tsv   三组各 100 轮、共 300 轮 query 训练记录
window.png           横轴标注参数占比的 accuracy、fidelity、posterior KL 三联直方图
first_10_mask.pt     前 10 候选加完整分类头的紧凑保护掩码
spread_10_mask.pt    分散 10 候选加完整分类头的紧凑保护掩码
last_10_mask.pt      后 10 候选加完整分类头的紧凑保护掩码
```
