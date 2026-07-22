# 实验 03 结果

本目录汇总当前正式协议下 `ResNet18+CIFAR-100` 的 MS 结果。绘图读取四种 baseline 的 32 个扫描点、`head_only` 与 TensorShield 两个普通 victim 单点、TEESlice standalone 单点、no protection 白盒以及 soft/hard 两条正式黑盒参考线，共 38 个输入点。

主图横坐标统一使用普通 ResNet18+CIFAR-100 的 `11,227,812` 个可训练参数作为
分母。TEESlice 最终保护 `703,092` 个可训练参数；其自身架构下的原生 private
parameter ratio 为 `5.9223%`，统一分母后的绘图比例为 `6.2621%`。两者均保存在
`data.tsv`，避免把 TEESlice 增加的 proxy slice 同时放入分子和分母后与普通策略
直接比较。

```text
metrics.png        accuracy、fidelity 与 posterior KL 三联总图
accuracy.png       surrogate accuracy 曲线
fidelity.png       fidelity 曲线
posterior_kl.png   posterior KL 曲线
data.tsv           38 个输入点、双参数比例、协议、来源和原始指标
manifest.json      输入协议、统一分母、artifact、双黑盒定义与输出清单
```

## 实验结论

总图说明，在当前统一协议下，保护位置与状态语义比单纯保护参数量更能解释低成本区间
的效果。分类头保护以 `0.4569%` 参数明显强于成本接近的浅层保护；TensorShield 以
`8.9934%` 参数明显强于相近成本的完整层与大权重策略，是普通固定 victim 下当前最强
的低比例 baseline。相对地，大权重扫描需要约 80% 以上参数才接近 soft 黑盒，因此
不适合作为低成本关键路径方案。

图中的 soft 与 hard 黑盒是不同查询能力的正式边界，TEESlice 则改变了 victim 结构。
因此普通部分保护策略只能与 soft 黑盒做同接口判断，TEESlice 只能作为 standalone
单点展示；不能把三类点混为同条件成本排名。曲线局部反弹或偶然越过黑盒属于单 seed、
有限 query 和优化路径波动，不表示更多保护会泄露信息，也不表示部分保护在信息上强于
完整黑盒。

soft-posterior 黑盒按 validation soft cross-entropy 选择第 45 轮，三项指标为 `0.1390/0.1463/3.039817`；hard-label 黑盒按 validation hard cross-entropy 选择第 3 轮，三项指标为 `0.0890/0.0969/3.387234`。两条线均为正式参考：soft 是普通部分保护策略的同接口对照，hard 展示 label-only 查询能力边界。

`head_only` 只保护 `0.4569%` 参数，得到 `0.3985/0.4621/1.616573`。它比成本接近的 `shallow_02`（`0.4144%`，`0.5608/0.7215/0.412557`）更能抑制 MS，确认分类头不可见具有显著影响；但仍远未达到 soft 黑盒。

在约 `1%` 参数比例下，`large_01` 为 `0.3637/0.4204/1.486967`，明显强于 `shallow_04` 的 `0.5541/0.6918/0.482461`。不过大权重扫描需要约 80% 以上保护比例才接近 soft 黑盒，不能作为低成本关键路径方案。

TensorShield 保护 `8.9934%` 参数，结果为 `0.1728/0.1865/2.694492`，明显强于相近成本的 `shallow_10` 和 `large_02`，但三项仍未达到 soft 黑盒。它继续是当前普通固定 victim 下最强的低比例正式 baseline。

完整层扫描不是逐点严格单调：`middle_14` 相比 `middle_12`、`deep_16` 相比 `deep_14` 出现小幅反弹。`deep_14`、`deep_16` 和 `large_08` 个别指标越过 soft 黑盒，只能视为单 seed、有限 query 与选模带来的攻击训练波动；攻击者可以忽略暴露状态并回退到 soft 黑盒，不能把这些点解释为信息意义上强于黑盒。

TEESlice 因 victim 结构和训练过程不同，使用独立星形单点和 `standalone_reproduction` 标记。其 validation-best 黑盒结果为 `0.1580/0.1698/3.342776`，不与普通 victim 策略连线，也不参与同条件成本排序。
