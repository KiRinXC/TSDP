# PG05 结果：Top-5 保护诊断

本实验仅使用 seed 42。八组都固定保护替换分类头；主比较是在 PG03/PG04 各自的
归一化方式内，同时保护 BN Top-5 与 main Conv Top-5。四个拆分组作为联合方案的
消融，另增加两个跨归一化交叉组。正式 soft-posterior 黑盒参考为
`0.1390 / 0.1463 / 3.039817`。

## 实验结果

| 排名与候选 | 保护参数比例 | best epoch | accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|---:|
| Feature BN Top-5 | 0.4734% | 93 | 0.4024 | 0.4777 | 1.246723 |
| Feature Conv Top-5 | 5.7101% | 94 | 0.1798 | 0.1947 | 2.642482 |
| **Feature BN+Conv Top-5** | **5.7267%** | **78** | **0.1692** | **0.1818** | **2.688260** |
| Parameter BN Top-5 | 0.4598% | 93 | 0.3555 | 0.3998 | 1.831663 |
| Parameter Conv Top-5 | 2.4269% | 93 | 0.2911 | 0.3251 | 2.100138 |
| **Parameter BN+Conv Top-5** | **2.4297%** | **100** | **0.2534** | **0.2829** | **2.263676** |
| **Feature Conv + Parameter BN** | **5.7130%** | **90** | **0.1489** | **0.1601** | **2.823222** |
| Feature BN + Parameter Conv | 2.4434% | 94 | 0.2844 | 0.3181 | 1.926525 |

## 实验结论

Feature Conv + Parameter BN 交叉组在八组中最强，为
`0.1489/0.1601/2.823222`，与 soft 黑盒只相差 accuracy `+0.0099`、fidelity
`+0.0138`、KL `-0.216595`。它比 Feature Conv 拆分组只增加 320 个 gamma 参数，
却带来 accuracy `-0.0309`、fidelity `-0.0346`、KL `+0.180740`，说明 Parameter BN
Top-5 能够高效补充 Feature Conv Top-5。

在固定 Feature Conv 的对照中，Parameter BN 交叉组相比 Feature BN 同源联合组又
改善 `-0.0203 accuracy / -0.0217 fidelity / +0.134962 KL`，而且少保护 1,536 个
gamma 参数。在固定 Parameter Conv 的对照中，Parameter BN 同源联合组相比 Feature
BN 交叉组改善 `-0.0310/-0.0352/+0.337151`，同样少保护 1,536 个 gamma 参数。
两个受控方向都表明 Parameter BN Top-5 优于 Feature BN Top-5。

Parameter 联合组为 `0.2534/0.2829/2.263676`。它比 Parameter Conv 拆分组只多保护
320 个 BN gamma 参数，却带来 `-0.0377 accuracy / -0.0422 fidelity / +0.163538 KL`。
相对极小的新增成本取得三项一致改善，说明参数量归一化筛选出的早期 BN gamma 与
Conv Top-5 联合时具有更明显的边际保护效率。

两组 BN 的成本非常接近，Parameter BN 为 `0.4598%`，Feature BN 为 `0.4734%`。
Parameter BN 的 accuracy 和 fidelity 分别低 `0.0469`、`0.0779`，posterior KL 高
`0.584939`，三项一致优于 Feature BN。这个近似同成本对照支持参数量归一化更适合
筛选少量 BN gamma。

两套 Conv Top-5 本身的成本仍不同，因此不能仅凭 Feature Conv + Parameter BN 的
绝对最优结果认定 Feature 归一化整体更高效。八组均未达到 soft 黑盒，且这里只运行
一个随机种子，当前结论只能作为后续同成本、多随机种子诊断的依据。

## 文件

```text
metrics.json                  完整协议、来源哈希、八组结果与 soft 黑盒引用
data.tsv                      八组主结果及相对 soft 黑盒差值
history.tsv                   八组共 800 轮 query train/validation 日志
metrics.png                   三指标柱状图与 soft 黑盒参考线
<case>_mask.pt                八组紧凑保护 mask
```
