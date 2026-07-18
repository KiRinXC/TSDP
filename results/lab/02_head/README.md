# 实验 02 结果

本目录记录 `ResNet18+CIFAR-100` 的分类头结构与权重训练方式对比。八组均使用
500 条 soft-posterior query，其中 400 条训练、100 条按 soft cross-entropy 选择
最早的最优 checkpoint；checkpoint 固定后只在 10,000 条 `eval_ms` 上评估一次。

全保护不复制 victim 权重，冻结组只训练分类头或 adapter：

```text
配置              可训练参数  best epoch  accuracy  fidelity  posterior KL
replace_frozen       51300       100       0.1724    0.1820    2.733553
replace_finetune  11227812        45       0.1390    0.1463    3.039817
adapter_frozen      100100        39       0.1794    0.1866    3.132601
adapter_finetune  11789612        76       0.1292    0.1319    3.405513
```

随机保护固定保护 `61/122` 个 unit，分类头 weight 与 bias 固定保护；冻结组冻结从
victim 复制出的暴露权重：

```text
配置              可训练参数  best epoch  accuracy  fidelity  posterior KL
replace_frozen     4444900          1       0.0131    0.0143    5.790709
replace_finetune  11227812         53       0.1394    0.1501    2.914578
adapter_frozen     5006700          1       0.0121    0.0140    9.763113
adapter_finetune  11789612         96       0.1345    0.1433    3.193306
```

在两种保护范围的“全部参数共同微调”对照中，`replace` 的 accuracy、fidelity 和
posterior KL 均优于 `adapter`，因此后续普通 MS 继续使用直接替换分类头。冻结对照
不呈现统一的三指标支配关系；尤其随机拼接 victim/public 状态后，冻结暴露权重会使
攻击几乎失效，说明该对照不能替代允许全模型联合微调的攻击者。

```text
metrics.json  八组 validation-best 选择、单次 eval_ms 指标与保护计划
history.tsv   八组配置共 800 条 query-train/query-validation 日志
```
