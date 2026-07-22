# PG06 结果：特征量与参数量联合归一化

PG06 对 PG01 的两项原始残差使用共同分母
`sqrt(feature_count×parameter_count)`，主分数等于 PG03 特征归一化乘积分数与 PG04
参数归一化乘积分数的几何平均。all、main 和 bn 均在各自候选集内独立排序。

## 实验结论

联合归一化仍明显偏向小参数 BN：all 前十项全部是 BN gamma，第一个 Conv weight
`layer1.1.conv1.weight` 只排在 all 第 17。all 与 BN 的前五项均为
`bn1.weight`、`layer1.1.bn2.weight`、`layer1.0.bn1.weight`、
`layer1.0.bn2.weight` 和 `layer1.1.bn1.weight`，与 PG04 Parameter BN Top-5
集合完全一致。

main 前五项依次为：

```text
layer1.1.conv1.weight
layer1.0.conv1.weight
layer2.0.conv1.weight
layer1.0.conv2.weight
layer1.1.conv2.weight
```

它与 PG04 Parameter main Top-5 的集合也完全一致，只改变了内部顺序；main Top-10
集合仍与 PG04 完全相同。因此按集合保护时，PG06 的 BN Top-5、main Top-5 及二者
联合 mask 都已经被 PG05 的 Parameter Top-5 case 覆盖，不需要重复训练 surrogate。

BN Top-10 相比 PG04 Parameter BN Top-10 只替换一项：PG06 加入 PG03 Feature BN
第一名 `layer4.1.bn2.weight`，移除 `layer2.1.bn1.weight`。这说明几何联合分数主要
保留参数归一化对早期小 BN 的偏好，同时允许残差信号特别强的末端 BN 进入前列。
该排序仍是数据侧代理，没有新增实际保护训练或多随机种子结论。

## 文件

```text
metrics.json / all.tsv / main.tsv / bn.tsv
all_<cross|natural|product>.png
main_<cross|natural|product>.png
bn_<cross|natural|product>.png
```
