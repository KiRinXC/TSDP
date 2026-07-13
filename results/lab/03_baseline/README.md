# 实验 03 结果

本目录保存当前正式协议下 `ResNet18+CIFAR-100` 四种 MS baseline 的保护参数比例曲线。绘图读取 32 个策略扫描点和 2 个上下界参考点。

```text
accuracy.png       surrogate accuracy 曲线
fidelity.png       fidelity 曲线
posterior_kl.png   posterior KL 曲线
data.tsv           34 个输入点及其 artifact_id、run_id 和原始指标
manifest.json      输入协议、策略 artifact 与输出清单
```

## 结果观察

在约 `1%` 实际参数保护比例下，`large_01` 的 accuracy/fidelity 为 `0.4090/0.4764`，同等成本附近的 `shallow_04` 为 `0.5560/0.6992`，全局大权重标量保护表现出更强的 MS 抑制效果。

`large_weight` 的分类头部分暴露时按 mask 混合初始化，不额外丢弃可见 victim 标量。8 个扫描点的 accuracy 和 fidelity 随保护比例严格下降，posterior KL 严格上升，实际不可见参数量与横坐标一致。

深层保护的第一个点已经保护 `21.4790%` 参数，但 accuracy/fidelity 仍为 `0.4567/0.5429`。其曲线需要到很高参数比例才接近全保护参考线，按实际参数成本比较时不占优。

`large_08` 在约 `95%` 参数保护下达到 `0.1533` accuracy 和 `0.1600` fidelity，已经接近全保护的 `0.1545/0.1610`。partial 点略低于全保护参考线属于固定种子训练波动；上下界参考线不是逐点必须满足的数学约束。
