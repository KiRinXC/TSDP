# 实验 10 结果

本目录记录五个固定 BasicBlock 上两种局部卷积与 BN gamma 配对策略的 seed-42
完整 MS 结果。实验协议和运行方式见 `lab/10_pair/README.md`；本文件只记录实际
产物、指标与必要解读。

## 结果

| 策略 | 保护参数 | 保护比例 | Best epoch | Accuracy | Fidelity | Posterior KL |
|---|---:|---:|---:|---:|---:|---:|
| 五个 `conv1.weight` + 对应 `bn2.weight` + 分类头 | 641,764 | 5.7158% | 92 | **0.1532** | **0.1662** | **2.808921** |
| 五个 `conv2.weight` + 对应 `bn1.weight` + 分类头 | 1,010,404 | 8.9991% | 94 | 0.1562 | 0.1694 | 2.784013 |

seed-42 正式参考边界为：

| 边界 | Accuracy | Fidelity | Posterior KL |
|---|---:|---:|---:|
| No protection 白盒 | 0.6182 | 1.0000 | 0.000000001 |
| Soft 黑盒 | 0.1390 | 0.1463 | 3.039817 |
| Hard-label 黑盒 | 0.0890 | 0.0969 | 3.387234 |

在当前单 seed 下，`conv1+BN2 gamma` 三项都略优于 `conv2+BN1 gamma`：accuracy
低 0.0030，fidelity 低 0.0032，Posterior KL 高 0.024908，同时少保护 368,640
个参数，即少 3.2833 个百分点。因此当前结果不支持用更高成本的对应
`conv2+BN1 gamma` 替换 `conv1+BN2 gamma`。

两种局部配对都未达到 soft 黑盒三线，说明只保护五个局部 gamma 不能复现 Lab05
中 Stem、全部 Block BN2 和 Downsample BN 联合形成的跨层尺度闭包。该结论只来自
seed 42，数值差异较小，不能表述为跨 seed 稳定规律。

## 文件

```text
metrics.json             协议、来源哈希、两种 mask、选模与最终指标
data.tsv                 两种策略的绘图原始点
history.tsv              两种策略共 200 轮 train/validation 日志
metrics.png              三指标柱状图与 soft/hard 黑盒参考线
conv1_bn2_mask.pt        五个 conv1、对应 BN2 gamma 与分类头 mask
conv2_bn1_mask.pt        五个 conv2、对应 BN1 gamma 与分类头 mask
```
