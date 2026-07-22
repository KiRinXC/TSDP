# PG07 结果：固定结构依赖下的 Feature Conv Top-k

本实验仅使用 seed 42。所有 case 固定保护替换分类头、Stem `bn1.weight` 和三个
downsample Conv，再按 PG03 Feature main `product_rank` 依次加入 Conv。正式 soft
黑盒参考为 `0.1390 / 0.1463 / 3.039817`，hard 黑盒参考为
`0.0890 / 0.0969 / 3.387234`。

## 实验结果

| k | 本级新增 Conv | 保护比例 | best epoch | accuracy | fidelity | posterior KL |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 无 | 1.9897% | 93 | 0.3008 | 0.3389 | 2.146325 |
| 1 | `layer2.0.conv1.weight` | 2.6463% | 96 | 0.2282 | 0.2497 | 2.473071 |
| 2 | `layer1.1.conv1.weight` | 2.9746% | 96 | 0.1997 | 0.2152 | 2.590039 |
| 3 | `layer1.0.conv1.weight` | 3.3030% | 93 | 0.1864 | 0.2027 | 2.674944 |
| 4 | `layer3.0.conv1.weight` | 5.9296% | 96 | 0.1422 | 0.1540 | 2.907615 |
| 5 | `layer2.1.conv1.weight` | 7.2429% | 93 | 0.1285 | 0.1384 | 3.000704 |
| 6 | `layer4.0.conv1.weight` | 17.7494% | 96 | 0.1180 | 0.1279 | 3.058497 |
| 7 | `layer4.1.conv1.weight` | 38.7624% | 93 | 0.1185 | 0.1290 | 3.059604 |

## 实验结论

Top-0 只保护固定结构集合时为 `0.3008/0.3389/2.146325`，说明分类头、Stem BN1
和三个 downsample Conv 本身还不足以形成强保护。前三个 Feature Conv 只把成本提高
到 3.3030%，就把 accuracy/fidelity 分别降到 0.1864/0.2027，早期排名项具有明显
边际贡献。

Top-5 为 `0.1285/0.1384/3.000704`，mask 与 Lab07 完全相同，三个指标也逐值一致，
证明本次独立 PG07 重放复现了 Lab07。它已在 accuracy 和 fidelity 上越过 soft 黑盒，
但 KL 仍低 0.039113。Top-6 为首个三项都越过 soft 黑盒的点：相对 Top-5 又改善
`-0.0105 accuracy / -0.0105 fidelity / +0.057793 KL`，代价是保护比例从 7.2429%
增加到 17.7494%。

Top-7 把成本大幅增加到 38.7624%，accuracy 相对 Top-6 上升 0.0005、fidelity 上升
0.0011，尽管 KL 仍增加 0.001107，仍已满足“任一指标变差即反弹”的早停条件。因此
实验保留 Top-7 作为反弹证据并停止，Top-8–16 不进入本实验结果。Top-6 是反弹前的
最后一点，也是当前扫描中的推荐停止点。所有已运行点仍弱于 hard 黑盒；这里只运行
一个随机种子，反弹位置不能外推为跨 seed 稳定结论。

## 文件

```text
metrics.json                  完整协议、来源哈希、早停条件、参考线与 8 组结果
data.tsv                      逐级成本、指标、相对 Top-0/前一级和黑盒差值
history.tsv                   Top-0–7 共 800 轮 query train/validation 日志
metrics_by_k.png              三项指标随 Feature Conv Top-k 的变化
metrics_by_cost.png           三项指标随实际保护参数比例的变化
top_<k>_mask.pt               实际运行的 8 个完整 tensor mask
```
