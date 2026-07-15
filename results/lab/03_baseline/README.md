# 实验 03 结果

本目录汇总当前正式协议下 `ResNet18+CIFAR-100` 的 MS 结果。绘图读取四种 baseline 的 32 个扫描点、`head_only` 与 TensorShield 两个普通 victim 单点、TEESlice standalone 单点，以及普通 victim 的 no/full 两个参考点，共 37 个输入点。

```text
metrics.png        accuracy、fidelity 与 posterior KL 三联总图
accuracy.png       surrogate accuracy 曲线
fidelity.png       fidelity 曲线
posterior_kl.png   posterior KL 曲线
data.tsv           37 个输入点及其比较范围、artifact_id、run_id 和原始指标
manifest.json      输入协议、策略 artifact 与输出清单
```

## 结果观察

在约 `1%` 实际参数保护比例下，`large_01` 的 accuracy/fidelity 为 `0.4090/0.4764`，同等成本附近的 `shallow_04` 为 `0.5560/0.6992`，全局大权重标量保护表现出更强的 MS 抑制效果。

`head_only` 只保护分类头的 `51,300` 个参数，实际保护比例为 `0.4569%`，其 accuracy/fidelity/posterior KL 为 `0.4404/0.5135/1.347578`。与保护比例接近的 `shallow_02`（`0.4144%`，`0.5651/0.7280/0.389144`）相比，只隐藏分类头对 MS 的抑制明显更强；但与普通 victim 的全保护结果 `0.1545/0.1610/2.835290` 仍有较大距离，因此分类头是重要控制变量，但并不足以单独达到当前全保护参考水平。

`large_weight` 的分类头部分暴露时按 mask 混合初始化，不额外丢弃可见 victim 标量。8 个扫描点的 accuracy 和 fidelity 随保护比例严格下降，posterior KL 严格上升，实际不可见参数量与横坐标一致。

深层保护的第一个点已经保护 `21.4790%` 参数，但 accuracy/fidelity 仍为 `0.4567/0.5429`。其曲线需要到很高参数比例才接近全保护参考线，按实际参数成本比较时不占优。

`large_08` 在约 `95%` 参数保护下达到 `0.1533` accuracy 和 `0.1600` fidelity，已经接近全保护的 `0.1545/0.1610`。partial 点略低于全保护参考线属于固定种子训练波动；上下界参考线不是逐点必须满足的数学约束。

图中的 no/full 水平线只使用普通先训练后分区 victim。TEESlice 因 victim 结构与训练过程不同，以独立星形单点和 `standalone_reproduction` 范围展示，不与普通 victim 策略连线，也不参与同条件排序。
