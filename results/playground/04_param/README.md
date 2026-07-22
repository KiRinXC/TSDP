# PG04 结果：参数量归一化

PG04 将 PG01 的交叉残差总量和自然残差总量分别除以 `numel(weight)`，主分数仍为
两项归一化残差的乘积。

## 实验结论

参数量归一化彻底改变了 all 排序：前五项全部是只含 64 个参数的早期 BN gamma，
依次为 `bn1.weight`、`layer1.1.bn2.weight`、`layer1.0.bn1.weight`、
`layer1.0.bn2.weight` 和 `layer1.1.bn1.weight`。其中 `bn1.weight` 的乘积分数约为
`4357.91`，远高于大卷积，说明 BN gamma 以极小参数成本承载了很高的残差密度。
20 个 BN gamma 独立排序的前五项和该顺序一致；`bn1.weight` 相比第二名
`layer1.1.bn2.weight` 约高 `13.22` 倍，早期 BN 在按参数成本衡量时占据明显优势。

main Conv 中前三项为 `layer1.1.conv1.weight`、`layer1.0.conv1.weight` 和
`layer1.0.conv2.weight`，分数约为 `0.012134`、`0.011067` 和 `0.001292`。这与
PG03 的 main 排序不同，说明按特征位置和按参数成本衡量会选择不同的卷积。

PG05 的近似同成本 BN 对照进一步表明：Parameter BN Top-5 以 `0.4598%` 参数比例
得到 `0.3555/0.3998/1.831663`，三项均优于 Feature BN Top-5 的
`0.4024/0.4777/1.246723`。这支持参数归一化筛选少量 BN gamma，但该证据仍只有
seed 42，不能外推为稳定多种子结论。PG05 的两组交叉对照还显示，无论固定 Feature
Conv 还是 Parameter Conv，Parameter BN Top-5 都优于 Feature BN Top-5。

## 文件

```text
metrics.json / data.tsv / main.tsv / bn.tsv
all_<cross|natural|product>.png
main_<cross|natural|product>.png
bn_<cross|natural|product>.png
```
