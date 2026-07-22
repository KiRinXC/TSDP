# 实验 08 结果

`resnet18_c100.tsv` 保存当前项目 ResNet18 在 `3×32×32` CIFAR-100 输入下的完整
逐算子、逐 unit 结构。表中省略 batch 维，并显式展开残差块的主分支、
identity/downsample 分支及逐元素相加。

需要注意：

- 最左列是正式 122-unit 注册表中的编号，不是表格行号；
- 每个数字只出现一次，并通过 `state名称` 唯一对应一个状态张量；
- H 列按作者 eligible rank 标出全部 17 个 weight，值为 `Top-1` 至 `Top-17`；
- H 列只表示 rank 前缀，不额外标记固定保护的 `last_linear.bias`；
- BatchNorm 展开为五行，分别对应 weight、bias、running mean、running variance
  和 `num_batches_tracked`；
- ReLU、池化、identity、残差相加和 flatten 没有 `state_dict` unit，统一标为 `—`；
- 当前模型沿用 ImageNet ResNet18 的 `7×7, stride=2` stem 和最大池化；
- `layer2.0`、`layer3.0`、`layer4.0` 使用 `1×1, stride=2` downsample；
- 同一 BasicBlock 的 `relu` 模块会调用两次，表中分别标记为“第1次”和“第2次”；
- `last_linear` 输出 100 类 logits，模型图内没有 Softmax。

尺寸已经由 `lab/08_structure/run.py` 使用虚拟输入逐模块核对。

## 实验结论

结构表与十种子干预共同说明，保护效果取决于状态在残差计算图中的具体位置，不能把
同一 BasicBlock 内的卷积视为可互换参数。五个 `conv1.weight` 中，重新暴露
`layer3.0.conv1.weight` 的攻击反弹最大，`layer2.0.conv1.weight` 次之，表明成员
贡献具有明显条件差异。

把五个 `conv1` 全部换成对应 `conv2` 后，保护参数反而增加，但 accuracy 和 fidelity
在 10/10 seed 上都回升；因此 `conv1` 集合的保护效果不是“保护任意五个大卷积”造成
的，而包含位置效应。局部 `conv1+BN2` 与 `conv2+BN1` 配对都没有达到 seed-42 soft
黑盒，且只有单 seed，不能据此建立稳定的局部配对排序。所有成员结论都是在其他保护
项固定时的条件依赖，不是跨模型的无偏选择规则。

## 五个 conv1 的条件攻击依赖

以 Lab06 的 5.7529% 候选集合作为基础保护：五个前中层 `conv1.weight`、全部 BN gamma
和分类头 weight/bias。实验在 seed 43–52 上分别重新暴露一个 `conv1.weight`，
其余保护项保持不变。表中指标为十个 seed 的均值 ± 样本标准差；`Δ` 是逐 seed
计算“重新暴露减基础集合”后再取均值。攻击反弹方向是 accuracy 和 fidelity 上升、
posterior KL 下降。

| 重新暴露的 tensor | 保护比例 | MS accuracy | Fidelity | Posterior KL | Δ accuracy | Δ fidelity | Δ KL | 三指标同时反弹 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无（基础集合） | 5.7529% | 0.11259 ± 0.00466 | 0.12141 ± 0.00495 | 3.17270 ± 0.05087 | — | — | — | — |
| rank-1 / unit 18 / `layer1.1.conv1.weight` | 5.4246% | 0.11848 ± 0.00377 | 0.12711 ± 0.00493 | 3.11384 ± 0.02991 | +0.00589 | +0.00570 | -0.05886 | 8/10 |
| rank-2 / unit 30 / `layer2.0.conv1.weight` | 5.0962% | 0.14228 ± 0.00650 | 0.15260 ± 0.00736 | 2.95119 ± 0.05313 | +0.02969 | +0.03119 | -0.22151 | 10/10 |
| rank-4 / unit 6 / `layer1.0.conv1.weight` | 5.4246% | 0.12202 ± 0.00585 | 0.13045 ± 0.00710 | 3.09298 ± 0.05983 | +0.00943 | +0.00904 | -0.07971 | 9/10 |
| rank-7 / unit 48 / `layer2.1.conv1.weight` | 4.4396% | 0.12563 ± 0.00381 | 0.13417 ± 0.00514 | 3.05501 ± 0.04585 | +0.01304 | +0.01276 | -0.11769 | 10/10 |
| rank-9 / unit 60 / `layer3.0.conv1.weight` | 3.1263% | 0.14964 ± 0.00553 | 0.16285 ± 0.00568 | 2.83238 ± 0.04063 | +0.03705 | +0.04144 | -0.34032 | 10/10 |
| matched soft 黑盒 | 100.0000% | 0.14775 ± 0.00613 | 0.15421 ± 0.00559 | 2.98499 ± 0.06360 | — | — | — | — |

结果说明五个 `conv1.weight` 的贡献并不相同：

- `layer3.0.conv1.weight` 的条件依赖最强。暴露后反弹幅度最大，且十个 seed 均未在
  三指标上同时维持 matched soft 黑盒边界。
- `layer2.0.conv1.weight` 的反弹幅度第二，十个 seed 全部同向；但仍有 4/10 seed
  在三指标上同时处于或弱于对应黑盒，说明它主要削弱保护裕量，不像
  `layer3.0.conv1.weight` 那样稳定跨过黑盒边界。
- `layer2.1.conv1.weight` 虽有 10/10 同向反弹，9/10 seed 仍同时处于或弱于黑盒；
  `layer1.0.conv1.weight` 和 `layer1.1.conv1.weight` 的反弹更小且分别只在 9/10
  和 8/10 seed 三指标同向。它们是当前集合中的弱条件贡献项，不能仅凭本实验称为
  维持黑盒等效所必需。

该结论只适用于“其他四个 `conv1.weight`、全部 BN gamma 和分类头仍受保护”的
leave-one-out 条件，不证明这些 tensor 在任意组合中独立必要。由于基础集合来自既有
MS 消融，本实验是后验机制验证，也不能直接作为跨模型的无偏先验选择规则。

## `conv1` 与对应 `conv2` 的直接保护替换

固定分类头 weight/bias、全部 20 个 BN gamma 和相同五个 BasicBlock，只把基础
集合中的五个 `conv1.weight` 一一换成对应的 `conv2.weight`。指标为 seed 43–52
的均值 ± 样本标准差：

| 保护位置 | 保护比例 | MS accuracy | Fidelity | Posterior KL | 三指标同时达到或弱于 matched soft 黑盒 |
|---|---:|---:|---:|---:|---:|
| 五个 `conv1.weight` | 5.7529% | 0.11259 ± 0.00466 | 0.12141 ± 0.00495 | 3.17270 ± 0.05087 | 10/10 |
| 对应五个 `conv2.weight` | 9.0362% | 0.12475 ± 0.00676 | 0.13472 ± 0.00614 | 3.10410 ± 0.05297 | 9/10 |
| matched soft 黑盒 | 100.0000% | 0.14775 ± 0.00613 | 0.15421 ± 0.00559 | 2.98499 ± 0.06360 | — |

逐 seed 配对后，`conv2` 组相对 `conv1` 组的平均变化为：

```text
MS accuracy    +0.01216
Fidelity       +0.01331
Posterior KL   -0.06859
```

accuracy 和 fidelity 在 10/10 seed 上都上升，说明攻击均得到恢复；KL 在 8/10
seed 上下降，另外两个 seed 小幅上升。尽管 `conv2` 组多保护了 368,640 个参数，
其整体保护效果反而更弱。因此，在当前五个块、分类头和 BN gamma 固定的直接拼接
攻击下，保护位置不能由“同一残差块内任一卷积”互换，`conv1` 的位置效应更强。

这项实验只回答直接替换后保护效果是否保持，不比较额外利用强度，也不单独证明
`conv1` 完成了坐标转换。两组参数成本不同，因此结论是位置反例，不是等成本效率
估计。

## 文件

```text
resnet18_c100.tsv       完整逐算子、逐 unit 结构表
tensorshield_top17.tsv  从完整表抽出的 17 个 TensorShield weight，按 unit 排序
dependency.json         完整协议、70 条结果、配对效应和十种子聚合
dependency.tsv          六种保护集合的逐 seed 指标与相对基础集合差值
dependency_history.tsv  五十组 leave-one-out 的 5,000 轮训练/验证历史
dependency.png          三指标均值、样本标准差、逐 seed 点和 matched soft 黑盒
dependency_*_mask.pt    基础、五个 leave-one-out 与完整保护的紧凑 mask
swap.json               conv1/conv2 直接替换协议、30 条结果和配对聚合
swap.tsv                三组逐 seed 指标及相对 conv1 组的差值
swap_history.tsv        conv2 组十个 seed 的 1,000 轮训练/验证历史
swap.png                conv1、conv2 和 matched soft 黑盒的三指标对照
swap_conv2_mask.pt      对应 conv2、全部 BN gamma 与分类头的紧凑 mask
pair.json               两种局部卷积/BN 配对的协议与结果
pair.tsv                两个 seed-42 原始点
pair_history.tsv        两组各 100 轮训练/验证历史
pair.png                两组配对与双黑盒参考线
conv1_bn2_mask.pt       五个 conv1 与局部 BN2 gamma 的 mask
conv2_bn1_mask.pt       五个 conv2 与局部 BN1 gamma 的 mask
```

局部配对中，`conv1+BN2` 保护 5.7158% 参数并得到
`0.1532/0.1662/2.808921`；`conv2+BN1` 保护 8.9991% 并得到
`0.1562/0.1694/2.784013`。前者三项略优且成本更低，但该比较只有单 seed。
