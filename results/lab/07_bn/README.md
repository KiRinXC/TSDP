# 实验 07 结果：BN Gamma 分组闭包

## 实验结论

BN gamma 的保护效果来自跨层组合，而不是任意一个局部 gamma 组。十种子 drop 实验中，
删除 Block BN2 或 Downsample BN 都在 10/10 seed 使攻击三项同时反弹，删除 Stem 在
9/10 seed 反弹；但删除 Block BN1 没有造成反弹，反而略微增强保护。这说明当前固定
五个 `conv1.weight` 的直接拼接攻击依赖 Block BN2 和 Downsample 的条件闭包，不能把
20 个 gamma 当作等价成员。

seed-42 add 实验中，任何单一 gamma 组都不能达到 soft 黑盒三线，进一步说明稳定效果
来自跨层尺度状态的联合保护。Block BN1 的反向现象应解释为当前拼接与优化条件下的
交互，不应外推为主动公开 BN1 gamma 会提升安全性。

新增的 seed-42 Feature Conv 扩展中，联合保护三个 downsample Conv 与 Stem
`bn1.weight` 后，三项均明显优于 Feature Conv Top-5 基线，并接近 soft 黑盒。该结果
说明 stage 切换映射与 Stem 尺度状态是值得保留的条件补充候选，但联合 case 不能分离
四个新增 state 的单独贡献。

## 十种子 Drop 消融

固定五个 `conv1.weight` 和完整分类头后，全部 gamma 相对 No gamma 在 10/10 seed
三项同时改善。相对 All gamma 的配对结果为：

| 删除组 | Accuracy 变化 | Fidelity 变化 | KL 变化 | 三项同时反弹 |
|---|---:|---:|---:|---:|
| Stem | +0.01308 | +0.01435 | -0.11758 | 9/10 |
| Block BN1 | -0.00605 | -0.00666 | +0.03965 | 0/10 |
| Block BN2 | +0.02225 | +0.02312 | -0.20992 | 10/10 |
| Downsample BN | +0.02224 | +0.02374 | -0.20939 | 10/10 |

因此 Block BN2 与 Downsample 是最稳定的条件必要组，Stem 有中等贡献；Block BN1
在当前直接拼接攻击下不是必要保护项。

## Seed-42 Add 消融

单组加入相对 No gamma 的改善顺序为 Downsample BN > Block BN2 > Stem >>
Block BN1。最强的 Downsample 仍未同时达到 seed-42 soft 黑盒三线，说明单组不能
替代跨层组合。

## Feature Conv 与 Downsample/Stem 扩展

本组固定 PG03 Feature main Conv Top-5 与完整替换分类头，再共同加入三个
`downsample.0.weight` 和 Stem `bn1.weight`：

| 保护集合 | 参数比例 | best epoch | Accuracy | Fidelity | Posterior KL |
|---|---:|---:|---:|---:|---:|
| Feature Conv Top-5 | 5.7101% | 94 | 0.1798 | 0.1947 | 2.642482 |
| + 3 Downsample Conv + Stem BN1 | 7.2429% | 93 | 0.1285 | 0.1384 | 3.000704 |

新增组合相对基线改善：

```text
Accuracy      -0.0513
Fidelity      -0.0563
Posterior KL  +0.358222
```

soft-posterior 黑盒为 `0.1390/0.1463/3.039817`。新增组合的 accuracy 与 fidelity
分别越过黑盒 `0.0105` 和 `0.0079`，但 KL 仍低 `0.039113`，因此没有同时达到三条
黑盒参考线。该测试只使用 seed 42，且一次共同加入四个 state；不能写成多种子稳定
结论，也不能判断是哪个 downsample Conv 或 Stem gamma 单独主导。结合 Lab09，低于
黑盒的单项指标仍需保留直接拼接初始化陷阱的解释边界。

```text
drop.json / drop.tsv / drop_history.tsv / drop.png
drop_<case>_mask.pt
add.json / add.tsv / add_history.tsv / add.png
add_<case>_mask.pt
feature.json / feature.tsv / feature_history.tsv / feature.png
feature_mask.pt
```
