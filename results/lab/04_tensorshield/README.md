# 实验 04 结果

本目录保存 TensorShield 作者确认 eligible rank 的 Top-1 至 Top-17 MS 前缀曲线、Top-12 完整 leave-one-out 与联合删除消融，以及 eligible rank 位置集合对照。三个实验均使用 `formal_victim_then_public_v1`、seed 42，并在每个独立 surrogate 初始化前重置随机状态。

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

## Top-12 完整 Leave-one-out 与联合删除消融

该消融使用统一的先训练后分区边界：`no_protection` 为白盒边界，`full_protection` 为黑盒边界。
所有 case 都固定保护 unit 121 `last_linear.bias`。rank-3 是 `last_linear.weight`，
因此 `drop_03` 只暴露分类头 weight，分类头模式为 `mixed`；其余 case 的分类头模式
为 `replace`。

```text
边界                    accuracy  fidelity  posterior KL
白盒（no protection）     0.6182    1.0000       0.000000
黑盒（full protection）   0.1545    0.1610       2.835290
```

```text
方案              删除的 weight                   保护比例  head     accuracy  fidelity  posterior KL
完整 Top-12       -                                24.7531%  replace    0.1655    0.1757       2.690443
删除 rank-1       layer1.1.conv1.weight            24.4248%  replace    0.1698    0.1811       2.661706
删除 rank-2       layer2.0.conv1.weight            24.0965%  replace    0.1718    0.1842       2.641335
删除 rank-3       last_linear.weight               24.2971%  mixed      0.2320    0.2589       2.250162
删除 rank-4       layer1.0.conv1.weight            24.4248%  replace    0.1692    0.1832       2.673958
删除 rank-5       layer1.1.conv2.weight            24.4248%  replace    0.1635    0.1745       2.714643
删除 rank-6       layer2.0.conv2.weight            23.4398%  replace    0.1627    0.1786       2.677902
删除 rank-7       layer2.1.conv1.weight            23.4398%  replace    0.1622    0.1746       2.683795
删除 rank-8       layer1.0.conv2.weight            24.4248%  replace    0.1638    0.1784       2.705771
删除 rank-9       layer3.0.conv1.weight            22.1265%  replace    0.1739    0.1891       2.608790
删除 rank-10      layer2.1.conv2.weight            23.4398%  replace    0.1623    0.1753       2.704415
删除 rank-11      layer3.0.conv2.weight            19.4999%  replace    0.1710    0.1830       2.646769
删除 rank-12      layer4.0.conv1.weight            14.2467%  replace    0.1757    0.1910       2.607458
同时删除 5/10    两项                             23.1115%  replace    0.1579    0.1705       2.718431
同时删除 5/8/10  三项                             22.7832%  replace    0.1624    0.1740       2.725131
同时删除 5/6/8/10  四项                           21.4699%  replace    0.1623    0.1743       2.704000
同时删除 5/7/8/10  四项                           21.4699%  replace    0.1678    0.1841       2.666556
同时删除 5/6/7/8/10  五项                         20.1566%  replace    0.1762    0.1936       2.608998
```

各 unit 的条件贡献并不相同。删除 rank-3 分类头 weight 后，accuracy 和 fidelity
分别反弹 `6.65` 和 `8.32` 个百分点，KL 降低 `0.440281`，是三项指标中影响最大的
单项；但该项同时把分类头从 `replace` 改成 `mixed`，不能与卷积项视为完全同类的
结构比较。

在 11 个卷积 weight 中，删除 rank-12 和 rank-9 产生最大的三指标一致反弹：
rank-12 使 accuracy/fidelity 分别提高 `1.02`/`1.53` 个百分点、KL 降低
`0.082985`；rank-9 分别提高 `0.84`/`1.34` 个百分点、KL 降低 `0.081653`。
rank-1、rank-2、rank-4 和 rank-11 也表现为三指标一致的正向保护贡献，但幅度更小。

rank-5 和 rank-10 则相反：单独删除任一项都会让 accuracy/fidelity 降低且 KL
增加，即当前攻击在三项指标上都变弱。rank-6、rank-7、rank-8 的三项指标方向
不一致，不能归为稳定正贡献或稳定负贡献。由此可见，作者 eligible rank 的 Top-12
顺序不等于本实验中的条件边际贡献顺序；尤其 rank-9/rank-12 的作用明显大于若干
更早进入前缀的项。

联合删除结果继续显示非加性：删除 5/10 在 accuracy 和 fidelity 上最接近黑盒，
差距分别为 `0.34` 和 `0.95` 个百分点；删除 5/8/10 在 KL 上最接近黑盒，差距
为 `0.110159`，但其 accuracy/fidelity 不优于删除 5/10。

以删除 5/8/10 为共同基准，对 rank-6 的 unit 36 `layer2.0.conv2.weight` 和
rank-7 的 unit 48 `layer2.1.conv1.weight` 补齐了完整 2×2：

```text
unit 36  unit 48  accuracy  fidelity  posterior KL
保护     保护       0.1624    0.1740       2.725131
暴露     保护       0.1623    0.1743       2.704000
保护     暴露       0.1678    0.1841       2.666556
暴露     暴露       0.1762    0.1936       2.608998
```

当 unit 48 仍受保护时，只暴露 unit 36 的 accuracy/fidelity 变化仅为
`-0.01`/`+0.03` 个百分点；当 unit 36 仍受保护时，只暴露 unit 48 已使二者反弹
`0.54`/`1.01` 个百分点，并使 KL 降低 `0.058576`。当 unit 48 已暴露后，再暴露
unit 36 又使 accuracy/fidelity 反弹 `0.84`/`0.95` 个百分点、KL 降低
`0.057558`。按
`I=y(36暴露,48暴露)-y(36暴露,48保护)-y(36保护,48暴露)+y(36保护,48保护)`
计算，交互项为 accuracy `+0.85` 个百分点、fidelity `+0.92` 个百分点和 KL
`-0.036426`，三项方向一致地表明同时暴露二者产生了额外攻击收益。

在 ResNet 计算图中，unit 36 位于 `layer2.0` 主分支末端，经过 BN、与 downsample
shortcut 相加和 ReLU 后进入下游 unit 48。当前结果呈现非对称的串行割点模式：
保护下游 unit 48 时，上游 unit 36 的暴露几乎不能转化为 accuracy/fidelity 收益；
一旦下游 unit 48 暴露，上游 unit 36 才表现出明显条件贡献。unit 48 因而是这对
候选中的主要下游安全割点，unit 36 是在该割点失守后才增强攻击的上游条件节点。
这比独立 unit 分数更支持沿真实计算图寻找攻击依赖路径或最小割，但仍只是当前模型、
攻击协议和单一 seed 下的局部因果证据；BN 与 downsample 状态尚未纳入这组
TensorShield eligible unit，不能把两点直接宣称为完整闭合路径。

五项联合删除把保护比例降至 `20.1566%`，但相对删除 5/8/10，accuracy/fidelity
分别反弹 `1.38`/`1.96` 个百分点，KL 降低 `0.116134`。因此 rank-6/7 不能随
rank-5/8/10 一并视为可安全删除项。18 组均未严格达到黑盒三项边界。

该结果说明 TensorShield Top-12 中存在明显不均匀和条件冗余，但不能据此直接断言
某个 unit 完全不被攻击者依赖：leave-one-out 测到的是其余 Top-12 已保护时的条件
贡献，可能受冗余路径、tensor 大小和 surrogate 优化轨迹影响。结果仍来自单一固定
种子，后续若据此重新选择保护集合，需要用独立规则选集合并以多种子 MS 验证，不能
直接把本次 MS 反馈当作正式选择器。

```text
ablation.json              18 组集合、固定 bias、黑白盒边界、2×2 设计与输入哈希
ablation.tsv               可直接绘图和统计的原始指标
ablation_history.tsv       17 组删除实验共 1,700 轮 query 训练记录
ablation_accuracy.png      accuracy 独立断轴柱状图
ablation_fidelity.png      fidelity 独立断轴柱状图
ablation_posterior_kl.png  posterior KL 独立断轴柱状图
drop_01_mask.pt            删除 rank-1 并固定保护分类头 bias 的紧凑掩码
...
drop_12_mask.pt            删除 rank-12 并固定保护分类头 bias 的紧凑掩码
drop_05_10_mask.pt         同时删除 rank-5/rank-10 的紧凑保护掩码
drop_05_08_10_mask.pt      同时删除 rank-5/rank-8/rank-10 的紧凑保护掩码
drop_05_06_08_10_mask.pt   在 2×2 基准上额外删除 rank-6 的紧凑保护掩码
drop_05_07_08_10_mask.pt   在 2×2 基准上额外删除 rank-7 的紧凑保护掩码
drop_05_06_07_08_10_mask.pt  同时删除 rank-5/6/7/8/10 的紧凑保护掩码
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
