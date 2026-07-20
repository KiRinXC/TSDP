# 实验 09：攻击依赖位置的接口机制

本实验不再重复比较某个固定保护集合能否让标准 MS 低于黑盒，而是解释 Lab08 已经
确认的现象：为什么完整复制未保护 victim 状态时，5.7529% 候选会形成严重的初始化
失配；以及前中层 `conv1.weight` 是否确实比相邻 `conv2.weight` 更容易形成这种
跨层接口失配。

实验只做未经训练的前向因果干预，不训练 surrogate，也不读取 `eval_ms`。所有分析
均固定使用 seed 43–52 各自的 100 条 query-validation 与 victim soft posterior。
主结果统一使用 posterior KL：它直接度量干预模型输出相对 victim posterior 的
偏离，越小表示接口越兼容。Soft cross-entropy 同时保留，用于和 Lab08 的
validation-best 训练目标对应。

## 分析一：分类头接口的范数反事实

重建 Lab08 的 0%、25%、50%、75% 和 100% 五个初始化。每个强度的分类头都保持
同 seed、同一次 canonical 初始化得到的受保护随机头，只改变未保护状态从 public
向 victim 的利用强度。

除直接记录分类头输入特征、logit 范数和 posterior KL 外，还构造一个不修改特征
方向的反事实：把当前模型每张图片的分类头输入向量缩放到同图片 public 特征的
L2 范数，再送入原来的随机分类头。

- 若 100% 的 KL 在只校正范数后大幅恢复，说明“低于黑盒”的训练前失配主要来自
  victim 特征幅值与受保护随机头不兼容；
- 若范数校正几乎无效，则失配主要来自特征方向或更深的坐标系变化；
- 该反事实不拟合参数，不使用标签，也不会被当成新的攻击结果。

## 分析二：七组受保护状态的完整 oracle-reveal

把 27 个受保护 state 按计算语义分成七组：

```text
head          last_linear.weight + last_linear.bias
bn_gamma      全部 20 个 BatchNorm gamma
layer1.0      layer1.0.conv1.weight
layer1.1      layer1.1.conv1.weight
layer2.0      layer2.0.conv1.weight
layer2.1      layer2.1.conv1.weight
layer3.0      layer3.0.conv1.weight
```

起点是 100% 混合初始化：未保护 state 来自 victim，上述七组保持 public/随机状态。
随后枚举七组的全部 `2^7=128` 种组合，把指定组临时恢复为 victim 真值。空集合必须
等于 100% 混合端点，七组全部恢复后必须逐 state 等于 victim。

这是 oracle 机制分析，不是可部署攻击：攻击者实际看不到被恢复的 state。完整枚举
用于避免把某个组只在单一上下文中的偶然效果解释为独立贡献。对每组报告：

- 从混合端点单独恢复该组带来的 KL 恢复；
- 在其余六组已恢复时，仅隐藏该组造成的 KL 损失；
- 在全部 128 个上下文中平均边际恢复得到的精确 Shapley 贡献；
- 与分类头共同恢复时，相对二者各自单独恢复的交互变化。

所有贡献都由同一个 posterior KL 因果结果计算，不把多个代理指标加权为新分数。

## 分析三：八个 BasicBlock 的 `conv1/conv2` 对照

为避免只研究 TensorShield 已经选中的位置，从完整 victim 出发，对 ResNet18 的
八个 BasicBlock 逐一构造六种单接口干预：

```text
只把 conv1.weight 换成 public
只把 bn1 gamma 换成 public
把 conv1.weight 与对应 bn1 gamma 换成 public
只把 conv2.weight 换成 public
只把 bn2 gamma 换成 public
把 conv2.weight 与对应 bn2 gamma 换成 public
```

其余 state 全部保持 victim。这样 `conv1` 与 `conv2` 在同一个残差块、相同数据和
相同 public/victim 来源下直接配对。gamma-only 对照进一步区分卷积坐标变化和残差
分支尺度变化。若前中层的 `conv1+bn1 gamma` 持续产生更大的 posterior KL，而
后层差异减弱，才能支持“块入口坐标转换比块出口更敏感”的结构性解释；若该规律
不成立，就不能把 Lab07 的五个后验候选推广成先验规则。

## 分析四：BN gamma 的跨层闭包

单个 BN gamma 的影响可能很小，但多层同时保持 public gamma 会连续改变主分支和
identity 分支的相对尺度。为区分独立效应与跨层交互，把全部 20 个 BN gamma 再
划成四类：

```text
stem              输入 stem 的 bn1 gamma
block_bn1         八个 BasicBlock 的 bn1 gamma
block_bn2         八个 BasicBlock 的 bn2 gamma
downsample        三个阶段转换 downsample BN gamma
```

从完整 victim 出发，枚举四类 gamma 的全部 `2^4=16` 个 public 替换组合。空集合
必须等于 victim，四类全部替换必须等于七组实验中的“仅隐藏全部 BN gamma”端点。
每类报告精确 Shapley KL 损伤；若完整组合远大于四类独立损伤之和，则 BN gamma
保护的主要作用是维持跨层尺度闭包，而不是某一个 gamma 本身承载大量任务信息。

最后只做一项跨实验核对：将五个候选 `conv1` 在近-victim 上的 KL 损伤，与 Lab07
同一 tensor 的十种子 MS 反弹均值计算 Pearson 相关。该结果只说明两个独立干预的
排序是否一致；样本只有五个且候选来自后验消融，不能用作先验选择器或显著性证明。

## 运行

先核对来源、七组状态、128 个组合端点和八个残差块：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/09_mechanism/run.py --dry-run
```

执行 seed 43–52 的完整前向分析：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/09_mechanism/run.py
```

输出固定为：

```text
results/lab/09_mechanism/metrics.json
results/lab/09_mechanism/lambda.tsv
results/lab/09_mechanism/lattice.tsv
results/lab/09_mechanism/attribution.tsv
results/lab/09_mechanism/seam.tsv
results/lab/09_mechanism/bn.tsv
results/lab/09_mechanism/metrics.png
```
