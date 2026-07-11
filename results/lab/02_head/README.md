# 实验 02 结果

本目录保存 `ResNet18+CIFAR-100` 下全保护和随机保护的分类头与权重训练方式对比。八组均使用 500 条 hard label query，并在 10,000 条 `eval_ms` 上评估；`best` 按 `surrogate_acc` 选择。

全保护不复制 victim 权重，冻结组冻结公开 ImageNet 权重：

```text
配置              可训练参数  best epoch  accuracy  fidelity  posterior KL
replace_frozen       51300        85       0.1455    0.1527    2.945590
replace_finetune  11227812        51       0.1800    0.1923    3.339833
adapter_frozen      100100        59       0.1753    0.1782    3.958877
adapter_finetune  11789612        99       0.1636    0.1715    5.085829
```

随机保护固定保护 `61/122` 个 unit，分类头两个 unit 固定保护且不参与随机抽取；冻结组只冻结从 victim 复制出的暴露权重：

```text
配置              可训练参数  best epoch  accuracy  fidelity  posterior KL
replace_frozen     4444900         2       0.0131    0.0141     7.650908
replace_finetune  11227812        69       0.1870    0.1998     3.566619
adapter_frozen     5006700         4       0.0162    0.0174    29.038946
adapter_finetune  11789612        72       0.1678    0.1824     6.780190
```

## 结论

全保护下最强配置是 `replace_finetune`。随机保护下同样是 `replace_finetune` 最强，其 accuracy 为 `18.70%`、fidelity 为 `19.98%`，均略高于全保护对应结果。

随机保护后冻结暴露权重的两组仅达到约 `1%-2%` accuracy，而允许全部权重共同微调后恢复到 `16.78%-18.70%`。这表明逐 tensor unit 拼接 victim 与公开权重会破坏相邻状态的协调关系，攻击者需要共同微调才能有效利用暴露权重。在本次设置中，替换分类头也优于适配分类头，因此后续随机保护 MS 应采用 `replace+finetune` 作为当前最强攻击配置。

这里按 `eval_ms` 选择 best 仅用于 Lab 阶段探索攻击配置，不作为论文主实验中的独立评估结果。

```text
metrics.json  八组配置、保护 unit 及完整 best/end 原始指标
history.tsv   八组配置共 800 条逐 epoch 训练和评估记录
```
