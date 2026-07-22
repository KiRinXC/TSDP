# TEESlice 独立复现结果

本目录只记录本次有效运行的结果文件和原始指标。TEESlice 改变了 victim 的结构、容量与训练过程，因此结果标记为 `standalone_reproduction`，不写入固定普通 victim 的主 `metrics.tsv`。

## 实验结论

本次复现将 active proxy 从 21 条剪到 8 条、private parameter ratio 从 `9.6803%`
降到 `5.9223%`，同时把 full victim 的 `eval_ms` accuracy 从 `0.7620` 保持到
`0.7578`。这说明当前剪枝准则能够显著降低私有执行成本，并将任务效用损失控制在
既定的内部验证容忍范围内。

即使攻击者知道最终剪枝拓扑并可训练全部可执行路径参数，黑盒 surrogate 仍只有
`0.1580` accuracy、`0.1698` fidelity 和 `3.342776` posterior KL，而完整状态白盒
恢复到 `0.7578/1.0000/3.5986e-10`。因此在本次已知拓扑攻击协议下，真正形成能力
差距的是未公开的 private proxy、alpha、分类头和任务 BN 状态，而不是把拓扑本身当作
秘密。

本次 8-proxy 拓扑与作者发布拓扑的 Jaccard 仅为 `0.3077`，所以这里验证的是方法流程
和当前剪枝结果，不是作者最终拓扑的逐项复现；同时由于 victim 架构与普通 ResNet18
不同，不能把该黑盒点直接用于普通策略的同条件优劣排序。

## 结果文件

```text
victim.json    source、teacher、full、pruned 的效用、剪枝判断与成本
metrics.json   已知剪枝拓扑的 validation-best 黑盒攻击与实际白盒评估
```

## 受保护 Victim

| 阶段 | checkpoint | eval_ms accuracy | 相对 source fidelity |
|---|---|---:|---:|
| source | `source/best.pth` | 0.7938 | - |
| teacher | `teacher/end.pth` | 0.7955 | 0.8517 |
| full | `full/best.pth` | 0.7620 | 0.7926 |
| pruned | `best.pth` | 0.7578 | 0.7874 |

full 模型的内部验证 accuracy 为 `0.7632`，容忍阈值为 `0.755568`。最终选择第 10 轮保存的 `last_tolerable`，内部验证 accuracy 为 `0.7568`，绝对下降 `0.0064`、相对下降 `0.8386%`，满足严格小于 1% 的剪枝容忍条件。

| 成本 | full | pruned |
|---|---:|---:|
| active proxy | 21 | 8 |
| private parameter | 1,197,057 | 703,092 |
| private parameter ratio | 9.6803% | 5.9223% |
| private FLOPs | 55,658,596 | 27,756,644 |
| private FLOPs ratio | 9.0900% | 4.7496% |

作者发布的 `ResNet18+C100` 拓扑包含 9 条 proxy，对应 task parameter `711,524`、task FLOPs `29,868,032`。本次严格按内部验证剪枝得到 8 条 proxy，与发布拓扑的 Jaccard 为 `0.3077`，并非精确匹配；实验没有使用 `eval_ms` 反向调整拓扑。

## MS 指标

黑盒 surrogate 使用最终 8 条活跃 proxy 的相同剪枝拓扑、公开 ImageNet backbone 和 fresh 私有状态。白盒重新加载最终 victim 的完整状态并实际评估。

| 能力与 checkpoint | epoch | validation loss | accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|---:|
| black-box `best.pth` | 92 | 3.674704 | 0.1580 | 0.1698 | 3.342776 |
| white-box full state（实际评估） | - | - | 0.7578 | 1.0000 | 0.0000000004 |

黑盒在 500 条 query 内使用与普通正式 surrogate 相同的固定 400/100 划分，按 validation soft cross-entropy 选择最早的最优 `best.pth`；`eval_ms` 不参与选模，只在 checkpoint 固定后完整评估一次。正式产物不再保存 surrogate `end.pth`。白盒 posterior KL 原始值为 `3.598613115940452e-10`，来自相同状态重复前向时的浮点误差。
