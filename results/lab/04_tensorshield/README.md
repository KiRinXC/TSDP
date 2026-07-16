# 实验 04 结果

本目录保存 TensorShield 作者确认 eligible rank 的 Top-1 至 Top-17 MS 前缀曲线，以及 rank-5/rank-10 冗余消融和 eligible rank 窗口对照。所有主要结果均为相同初始化与 query 顺序下第 100 轮的 `end` 指标。

## Top-1 至 Top-17 前缀曲线

每个 Top-k 均固定保护 unit 121 `last_linear.bias`。Top-1、Top-2 的分类头模式为 `mixed`；Top-3 起同时保护分类头 weight 与 bias，分类头模式为 `replace`。

```text
k   新增 weight                    参数比例   accuracy  fidelity  posterior KL
1   layer1.1.conv1.weight           0.3292%     0.5984    0.8164       0.167886
2   layer2.0.conv1.weight           0.9859%     0.5494    0.6969       0.447626
3   last_linear.weight              1.4419%     0.4056    0.4657       1.456595
4   layer1.0.conv1.weight           1.7702%     0.3919    0.4436       1.535802
5   layer1.1.conv2.weight           2.0985%     0.3910    0.4432       1.523687
6   layer2.0.conv2.weight           3.4118%     0.3627    0.4076       1.669490
7   layer2.1.conv1.weight           4.7252%     0.3374    0.3738       1.783606
8   layer1.0.conv2.weight           5.0535%     0.3279    0.3646       1.803998
9   layer3.0.conv1.weight           7.6801%     0.2594    0.2824       2.138012
10  layer2.1.conv2.weight           8.9934%     0.2566    0.2783       2.168109
11  layer3.0.conv2.weight          14.2467%     0.2200    0.2353       2.332578
12  layer4.0.conv1.weight          24.7531%     0.1934    0.2121       2.453752
13  layer4.0.conv2.weight          45.7661%     0.1862    0.2048       2.507269
14  layer4.1.conv1.weight          66.7791%     0.1835    0.1982       2.536271
15  layer4.1.conv2.weight          87.7920%     0.1835    0.1983       2.588010
16  layer3.1.conv2.weight          93.0453%     0.1845    0.2013       2.577924
17  layer3.1.conv1.weight          98.2985%     0.1804    0.1959       2.584539
```

当前正式参考线为：无保护 accuracy `0.6182`、fidelity `1.0000`、KL 约为 `0`；全保护 accuracy `0.1545`、fidelity `0.1610`、KL `2.835290`。

Top-10 的保护 mask 与 Figure 12(d) 固定集合一致，逻辑 SHA256 为 `1e3aa38124f084dd39eab42a4d3f1ddf1ca86807812796c66a8318c05e7aa2cb`。当前曲线在每个 surrogate 初始化前重置种子，Top-10 accuracy 为 `0.2566`。该值与正式单点结果的初始化 RNG 不同，因此当前受控曲线只用于比较各前缀，不替代正式 baseline 指标。

曲线总体随保护范围扩大而增强，但不是严格单调。Top-3 加入 `last_linear.weight` 后，accuracy 相比 Top-2 降低 `14.38` 个百分点，fidelity 降低 `23.12` 个百分点，KL 增加 `1.008969`，说明早期最大跳变仍来自分类头 weight，而固定加入的 100 个 bias 参数没有消除这一现象。

Top-10 后仍存在明显改善：Top-11 和 Top-12 分别把 accuracy 降至 `0.2200` 和 `0.1934`。但 Top-12 之后进入高成本、低边际收益区间，参数比例从 `24.7531%` 增至 `98.2985%`，最终只再降低 `1.30` 个百分点 accuracy 和 `1.62` 个百分点 fidelity。Top-15 到 Top-16 三项指标还出现小幅反向变化，说明 eligible rank 的细粒度次序不能由单次前缀曲线视为严格边际贡献排序。

Top-17 与全保护仍相差 `2.59` 个百分点 accuracy、`3.49` 个百分点 fidelity 和 `0.250751` KL。也就是说，在当前攻击协议下，保护全部 17 个 eligible weight 已非常接近全保护，但仍未严格达到全保护参考线。

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
完整 Top-12      24.7531%     0.1934    0.2121       2.453752
删除 rank-5      24.4248%     0.1966    0.2147       2.462021
删除 rank-10     23.4398%     0.1985    0.2154       2.452102
同时删除 5/10    23.1115%     0.2011    0.2166       2.457284
```

删除 rank-5 `layer1.1.conv2.weight` 后，accuracy 和 fidelity 分别提高 `0.32`、`0.26` 个百分点，但 KL 反而增加 `0.008268`。删除 rank-10 `layer2.1.conv2.weight` 后，accuracy 和 fidelity 分别提高 `0.51`、`0.33` 个百分点，KL 只降低 `0.001651`。两项删除均使 accuracy/fidelity 指向攻击小幅增强，但 KL 没有形成同等幅度的一致证据。

同时删除两项后，accuracy 和 fidelity 分别提高 `0.77`、`0.45` 个百分点，KL 增加 `0.003532`；保护参数减少 `184,320`，相当于完整 Top-12 保护参数的 `6.63%`。三组删除结果说明 rank-5/rank-10 在 Top-12 内对 accuracy/fidelity 仍有小幅条件贡献，但当前差异不足以形成三项指标一致的强结论。

完整 Top-12 距黑盒仍差 `3.89` 个百分点 accuracy、`5.11` 个百分点 fidelity 和 `0.381538` KL；同时删除两项后对应差距为 `4.66`、`5.56` 个百分点和 `0.378006` KL。Top-12 已明显接近黑盒，但四组均未严格达到黑盒边界。结论仅适用于当前模型、数据集、攻击协议和固定种子。

```text
ablation.json          四组集合、黑白盒边界、相对完整 Top-12 的差值与输入哈希
ablation.tsv           可直接绘图和统计的原始指标
ablation_history.tsv   三组删除实验共 300 轮 query 训练记录
ablation.png           accuracy、fidelity、posterior KL 与黑白盒边界三联柱状图
drop_05_mask.pt        删除 rank-5 的紧凑保护掩码
drop_10_mask.pt        删除 rank-10 的紧凑保护掩码
drop_05_10_mask.pt     同时删除 rank-5/rank-10 的紧凑保护掩码
```

## Eligible rank 窗口消融

从 17 个 eligible weight 中排除分类头 weight 后得到 16 个候选。`first_10` 保护候选第 1 至 10 项，`last_10` 保护候选第 7 至 16 项；两组均额外保护 `last_linear.weight` 和 `last_linear.bias`。

```text
方案       非头候选  保护 unit  参数比例  head mode  accuracy  fidelity  posterior KL
first_10         10         12   14.2467%  replace      0.2200    0.2353       2.332578
last_10          10         12   94.0303%  replace      0.2324    0.2514       2.305942
```

`first_10` 的 mask 等价于前缀曲线 Top-11，因此两者指标完全相同。`last_10` 保护参数约为 `first_10` 的 `6.60` 倍，但 accuracy 和 fidelity 分别高 `1.24`、`1.61` 个百分点，KL 低 `0.026636`，即攻击效果反而略强。这说明在固定完整分类头和相同候选数量后，eligible rank 前部候选的保护成本效率明显高于后部候选。

两组窗口有 4 个候选重叠，且参数成本并不相同，因此该结果只能作为前后窗口的直接对照，不能视为等成本或统计意义上的全局排序证明。

```text
window.json          两个 eligible 窗口、保护统计、输入哈希和 end 原始指标
window.tsv           两组保护成本与三项 MS 原始指标
window_history.tsv   两组各 100 轮、共 200 轮 query 训练记录
window.png           横轴标注参数占比的 accuracy、fidelity、posterior KL 三联直方图
first_10_mask.pt     前 10 候选加完整分类头的紧凑保护掩码
last_10_mask.pt      后 10 候选加完整分类头的紧凑保护掩码
```
