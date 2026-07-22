# PG01 结果：四路原始 Weight 输出

PG01 对固定 500 张 query 保存了 20 个 Conv weight 与 20 个 BN gamma 的
`z_pp/z_pv/z_vp/z_vv`，并额外保存紧凑公式直接计算的交叉残差 `I`。全部为 float32，
共 40 个文件、约 0.944 GiB；所有 bias 和最终分类层均未进入候选。

## 实验结论

未归一化的主分数 `raw_cross_l1 × raw_natural_l1` 明显受输出特征总量影响。all 前三项
为 `bn1.weight`、`layer1.1.conv1.weight` 和 `layer1.0.conv1.weight`，分数分别约为
`1.7850e7`、`1.64895e7` 和 `1.50397e7`；main 前三项则是后两者及
`layer2.0.conv1.weight`。这说明原始总残差适合保存完整信号，但不能直接比较不同
空间尺寸或不同参数量候选的单位效率。

stem `conv1.weight` 的自然残差非零，但交叉残差和乘积分数严格为 0，因为 public 与
victim 在第一层接收完全相同的图片，输入差为 0。这也验证了紧凑交叉公式没有把四路
float32 相消误差误计为真实残差。

PG01 的结论仅是“原始残差总量如何分布”。PG02–PG04 必须读取本目录的同一份数据；
这里没有 surrogate 训练，不能据此声称某个 weight 已具有实际保护效果。

## 文件

```text
manifest.json                   模型、query、公式、正确性检查和输出索引
data.tsv / main.tsv             all 40 项与直接抽取的 main 16 项原始指标
activations/unit_<index>.pt     四路 z 与精确交叉残差 I
all_<cross|natural|product>.png
main_<cross|natural|product>.png
```
