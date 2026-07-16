# 实验 02 结果

本目录保存 `ResNet18+CIFAR-100` 下全保护和随机保护的分类头与权重训练方式对比。八组均使用 500 条 hard label query，并在 10,000 条 `eval_ms` 上评估；`best` 按 surrogate accuracy 选择。所有配置均采用 `formal_victim_then_public_v1`、seed 42，并在每个独立配置前重置初始化与 query sampler。

全保护不复制 victim 权重，冻结组冻结公开 ImageNet 权重：

```text
配置              可训练参数  best epoch  accuracy  fidelity  posterior KL
replace_frozen       51300        55       0.1762    0.1825    2.976940
replace_finetune  11227812        75       0.1804    0.1932    3.711718
adapter_frozen      100100        53       0.1718    0.1777    5.208276
adapter_finetune  11789612        96       0.1627    0.1710    7.037834
```

随机保护固定保护 `61/122` 个 unit，分类头两个 unit 固定保护且不参与随机抽取；冻结组只冻结从 victim 复制出的暴露权重：

```text
配置              可训练参数  best epoch  accuracy  fidelity  posterior KL
replace_frozen     4444900         4       0.0164    0.0205     13.130089
replace_finetune  11227812        86       0.1812    0.1935      3.591751
adapter_frozen     5006700        14       0.0170    0.0164    137.881760
adapter_finetune  11789612        88       0.1636    0.1728      6.950258
```

## 结论

随机保护下 `replace_finetune` 的 accuracy/fidelity 为 `18.12%/19.35%`，仍是最强攻击配置。全保护下按 accuracy 和 fidelity 同样由 `replace_finetune` 最强；虽然 `adapter_finetune` 的 KL 更大，但其 accuracy/fidelity 更低，不能据此把 adapter 选为统一攻击器。

随机保护后冻结暴露权重的两组只有约 `1%-2%` accuracy，而允许全部权重共同微调后恢复到 `16.36%-18.12%`。这表明逐 tensor unit 拼接 victim 与公开权重会破坏相邻状态的协调关系，攻击者需要共同微调才能有效利用暴露权重。在当前协议中，替换分类头也优于适配分类头，因此后续普通 MS 使用 `replace+finetune`。

这里按 `eval_ms` 选择 best 仅用于 Lab 阶段探索攻击配置，不作为论文主实验中的独立评估结果。

```text
metrics.json  八组配置、统一随机轨迹、保护 unit 及完整 best/end 原始指标
history.tsv   八组配置共 800 条逐 epoch 训练和评估记录
```
