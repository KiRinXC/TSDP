# MS 原始结果

本目录只保存 surrogate 在 `eval_ms` 上的原始评估结果，不保存 accuracy drop、fidelity drop、相对黑盒倍数等派生指标。

```text
results/MS/<model>/<dataset>/
  metrics.tsv             所有 run 的原始指标索引
  <run_id>/metrics.json   单次运行 best 与 end 的原始指标
```

每条记录同时保存样本总数、正确数、一致数、KL 总和及对应均值，后续统计应优先使用整数计数和总和重算。冻结与微调的运行分别保留，由后续汇总阶段选择攻击效果更强的配置。

## ResNet18+CIFAR-100 上下限

query budget 固定为训练集的 1%，即 500。以下均为 `best.pth` 在 10,000 条 `eval_ms` 上的原始结果。

```text
保护策略         标签   run_id        best epoch  accuracy  fidelity  posterior KL
no_protection    soft   0c9cba1e527c  0           0.6182    1.0000    0.0000000011
full_protection  hard   0162efd649b5  71          0.1681    0.1758    5.057231
```

全保护严格模拟只能查询输入和输出类别的黑盒，因此只允许 hard pseudo label，训练阶段只读取 `labels.tsv`，不读取 `posteriors.pt`。`no_protection` 是所有 victim 权重暴露的攻击上界；`full_protection` 是仅标签查询的黑盒下界。
