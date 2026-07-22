# PG02 结果：有效秩

PG02 从 PG01 的同一批四路输出计算交叉残差、自然残差、四路输出的逐图片谱有效秩，
不重新运行模型，也不做特征图或参数量归一化。

## 实验结论

`cross_rank × natural_rank` 与残差幅值给出了不同排序。all 中前列由
`layer1.1.bn2.weight`、`layer1.1.conv2.weight` 和 `layer1.1.bn1.weight` 等中前层
候选占据；main 中 `layer1.1.conv2.weight` 排名第一，而它并不是 PG01 原始残差乘积
的 main 第一项。这说明残差能量大小与残差分布在多少通道方向上是两种不同信号，
不能相互替代。

20 个 BN gamma 独立排序后，前三项为 `layer1.1.bn2.weight`、
`layer1.1.bn1.weight` 和 `layer2.0.bn1.weight`，秩乘积分数分别为 `245.875722`、
`143.481494` 和 `117.298162`。这说明 BN 内部的谱分布排序偏向中前层，与 PG03
偏向末端 BN、PG04 偏向 stem BN 的结果均不同。

layer4 的输出空间为 `1×1`，有效秩容量只有 1，因此对应候选的非零
`cross_rank × natural_rank` 都固定为 1，无法继续区分其通道结构。有效秩适合描述
空间—通道谱分布，但本身不表示保护成本或实际 MS 效果。

## 文件

```text
metrics.json / data.tsv / main.tsv / bn.tsv
all_<7种秩指标>.png
main_<7种秩指标>.png
bn_<7种秩指标>.png
```
