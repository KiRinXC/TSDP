# TEESlice 独立复现结果

本目录只记录本次有效运行的结果文件和原始指标。TEESlice 改变了 victim 的结构、容量与训练过程，因此结果标记为 `standalone_reproduction`，不写入固定普通 victim 的主 `metrics.tsv`。

## 结果文件

```text
victim.json    source、teacher、full、pruned 的效用、剪枝判断与成本
metrics.json   已知剪枝拓扑的黑盒攻击、诊断 checkpoint 与实际白盒评估
```

## Defended Victim

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

| 能力与 checkpoint | epoch | accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|
| black-box `end.pth`（主结果） | 100 | 0.1619 | 0.1784 | 3.251131 |
| black-box `best.pth`（训练诊断） | 69 | 0.1638 | 0.1797 | 3.258412 |
| white-box full state（实际评估） | - | 0.7578 | 1.0000 | 0.0000000005 |

主结果固定读取第 100 轮 `end.pth`。`best.pth` 只用于观察训练过程，不参与正式选模。白盒 posterior KL 原始值为 `4.814928395546758e-10`，来自相同状态重复前向时的浮点误差。
