# PG03 结果：特征图归一化

PG03 将 PG01 的交叉残差总量和自然残差总量分别除以输出元素数 `C×H×W`，主分数为
两项归一化残差的乘积。

## 实验结论

特征图归一化后，all 乘积前三项为 `layer2.0.conv1.weight`、
`layer1.1.conv1.weight` 和 `layer4.1.bn2.weight`，分数分别为 `1.061349`、
`0.982852` 和 `0.981602`；main 前三项为 `layer2.0.conv1.weight`、
`layer1.1.conv1.weight` 和 `layer1.0.conv1.weight`。

20 个 BN gamma 独立排序后，第一名为 `layer4.1.bn2.weight`，乘积分数为
`0.981602`；第二名 `layer4.0.downsample.1.weight` 为 `0.133594`。两者差距约
`7.35` 倍，说明按输出特征位置归一化时，末端 BN gamma 承载的平均残差信号最突出。
这只是 BN 候选内部的排序，不包括 stem 或 downsample Conv。

与 PG01 相比，早期大特征图不再仅凭输出元素多占据前列，说明该归一化回答的是
“每个输出特征位置平均承载多少交叉×自然残差信号”。但它没有考虑保护这些输出所需
隐藏的 weight 参数量，因此不能直接解释保护效率。

## 文件

```text
metrics.json / data.tsv / main.tsv / bn.tsv
all_<cross|natural|product>.png
main_<cross|natural|product>.png
bn_<cross|natural|product>.png
```
