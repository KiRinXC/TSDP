# MS 原始结果

本目录只保存 surrogate 在 `eval_ms` 上的原始评估结果，不保存 accuracy drop、fidelity drop、相对黑盒倍数等派生指标。

```text
results/MS/<model>/<dataset>/
  metrics.tsv                   普通固定 victim 的原始指标索引
  <artifact_id>/metrics.json    单次运行的选模信息与最终原始指标
```

当前正式协议先从 500 条 query 中按 seed 42 与固定 offset 100 拆出 400 条训练样本和 100 条 validation 样本。soft 攻击按 validation soft cross-entropy、hard 攻击按 validation hard cross-entropy 选择最早的最优 `best.pth`；`eval_ms` 不参与选模，只在 checkpoint 固定后完整评估一次。普通正式策略与 soft 黑盒使用 `soft_query_validation_best_v1`，label-only 黑盒使用 `hard_query_validation_best_v1`。正式 surrogate 不保存 `end.pth`。

## ResNet18+CIFAR-100 上下界

| 保护策略 | 查询输出 | artifact_id | best epoch | accuracy | fidelity | posterior KL |
|---|---|---|---:|---:|---:|---:|
| no protection | soft posterior | `no_protection` | 0 | 0.6182 | 1.0000 | 1.0591e-9 |
| full protection | soft posterior | `full_protection` | 45 | 0.1390 | 0.1463 | 3.039817 |
| full protection | hard label | `hard_blackbox` | 3 | 0.0890 | 0.0969 | 3.387234 |

soft `full_protection` 与 hard `hard_blackbox` 使用相同 victim、500 条 query、400/100 划分、完整保护 mask、公开初始化和训练超参数，只改变查询输出与 validation loss。两者都是正式黑盒参考：soft 用于和 posterior-visible 部分保护策略进行同接口比较，hard 用于展示 label-only 查询能力边界。

## ResNet18+CIFAR-100 仅分类头保护

`head_only` 只隐藏 `last_linear.weight` 和 `last_linear.bias`，完整复制普通 victim 的其余 backbone 状态。mask 保护 `2/122` 个 unit、`51,300/11,227,812` 个参数，比例为 `0.4569%`，分类头使用 `replace`。

| run_id | best epoch | accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|
| `d2d4a36a3208` | 93 | 0.3985 | 0.4621 | 1.616573 |

它比参数比例接近的 `shallow_02`（`0.4144%`，`0.5608/0.7215/0.412557`）更能抑制 MS，但仍未达到 soft 黑盒，因此分类头是重要控制变量，却不足以单独替代关键路径保护。

## ResNet18+CIFAR-100 TensorShield

TensorShield 只使用作者确认 rank 对应的 Figure 12(d) 固定集合，不重新计算 importance。mask 保护作者 eligible rank Top-10 weight 与固定分类头 bias，共 `11/122` 个 unit、`1,009,764/11,227,812` 个参数，比例为 `8.9934%`。

| run_id | best epoch | accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|
| `5057ffe55a3e` | 93 | 0.1728 | 0.1865 | 2.694492 |

该结果明显强于相近参数比例的浅层与大权重扫描点，但三项指标仍未达到 soft 黑盒 `0.1390/0.1463/3.039817`。

## ResNet18+CIFAR-100 TEESlice

TEESlice 改变了 victim 结构与训练过程，因此按 `standalone_reproduction` 独立保存，不写入普通固定 victim 的主 `metrics.tsv`。最终剪枝 victim 的 `eval_ms` accuracy 为 `0.7578`；已知拓扑的 soft-posterior 黑盒 surrogate 按 validation loss 选择第 92 轮 `best.pth`，得到 `0.1580/0.1698/3.342776`。完整状态白盒实际评估为 `0.7578/1.0000/3.5986e-10`。详细成本和四阶段结果见 `results/MS/resnet18/c100/teeslice/README.md`。

## ResNet18+CIFAR-100 完整层 baseline

| 策略 | 层数 | 官方层范围 | unit | 参数比例 | 分类头 | artifact_id | best epoch | accuracy | fidelity | posterior KL |
|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|
| 浅层 | 2 | 1-2 | 12 | 0.4144% | exposed | `shallow_02` | 83 | 0.5608 | 0.7215 | 0.412557 |
| 中间层 | 2 | 9-10 | 18 | 4.2432% | exposed | `middle_02` | 62 | 0.4289 | 0.4995 | 1.126300 |
| 深层 | 2 | 17-18 | 8 | 21.4790% | replace | `deep_02` | 87 | 0.4072 | 0.4845 | 1.208450 |
| 浅层 | 4 | 1-4 | 24 | 1.0733% | exposed | `shallow_04` | 91 | 0.5541 | 0.6918 | 0.482461 |
| 中间层 | 4 | 8-11 | 30 | 10.8166% | exposed | `middle_04` | 65 | 0.3332 | 0.3668 | 1.693744 |
| 深层 | 4 | 15-18 | 20 | 63.5232% | replace | `deep_04` | 98 | 0.3501 | 0.3918 | 1.596568 |
| 浅层 | 6 | 1-6 | 42 | 2.1370% | exposed | `shallow_06` | 78 | 0.5140 | 0.6158 | 0.761770 |
| 中间层 | 6 | 7-12 | 42 | 17.3900% | exposed | `middle_06` | 48 | 0.2941 | 0.3291 | 1.904762 |
| 深层 | 6 | 13-18 | 38 | 80.4731% | replace | `deep_06` | 93 | 0.2474 | 0.2713 | 2.215831 |
| 浅层 | 8 | 1-8 | 54 | 4.7682% | exposed | `shallow_08` | 83 | 0.4787 | 0.5474 | 0.986718 |
| 中间层 | 8 | 6-13 | 60 | 23.3819% | exposed | `middle_08` | 92 | 0.2262 | 0.2499 | 2.312532 |
| 深层 | 8 | 11-18 | 50 | 90.9887% | replace | `deep_08` | 88 | 0.1956 | 0.2114 | 2.583895 |
| 浅层 | 10 | 1-10 | 72 | 9.0113% | exposed | `shallow_10` | 55 | 0.3620 | 0.3960 | 1.617883 |
| 中间层 | 10 | 5-14 | 78 | 35.4035% | exposed | `middle_10` | 65 | 0.1889 | 0.2068 | 2.464583 |
| 深层 | 10 | 9-18 | 68 | 95.2318% | replace | `deep_10` | 93 | 0.1599 | 0.1704 | 2.852605 |
| 浅层 | 12 | 1-12 | 84 | 19.5269% | exposed | `shallow_12` | 60 | 0.2983 | 0.3196 | 1.971245 |
| 中间层 | 12 | 4-15 | 90 | 56.7551% | exposed | `middle_12` | 90 | 0.1798 | 0.1875 | 2.501596 |
| 深层 | 12 | 7-18 | 80 | 97.8630% | replace | `deep_12` | 91 | 0.1437 | 0.1569 | 2.969841 |
| 浅层 | 14 | 1-14 | 102 | 36.4768% | exposed | `shallow_14` | 50 | 0.2143 | 0.2289 | 2.310467 |
| 中间层 | 14 | 3-16 | 102 | 78.1066% | exposed | `middle_14` | 93 | 0.1895 | 0.1991 | 2.446559 |
| 深层 | 14 | 5-18 | 98 | 98.9267% | replace | `deep_14` | 39 | 0.1184 | 0.1300 | 3.158334 |
| 浅层 | 16 | 1-16 | 114 | 78.5210% | exposed | `shallow_16` | 93 | 0.2081 | 0.2160 | 2.346672 |
| 中间层 | 16 | 2-17 | 114 | 99.4582% | exposed | `middle_16` | 90 | 0.1462 | 0.1541 | 3.141881 |
| 深层 | 16 | 3-18 | 110 | 99.5856% | replace | `deep_16` | 58 | 0.1242 | 0.1313 | 3.124720 |

完整层曲线并非全都严格单调。`middle_14` 相比 `middle_12`、`deep_16` 相比 `deep_14` 出现小幅反弹，说明在固定单 seed 和有限 query 下，更多保护参数不保证训练出的攻击模型逐点更弱。`deep_14`、`deep_16` 低于 soft 黑盒也不能解释为强于理论黑盒；攻击者始终可以忽略暴露权重并回退到 soft 黑盒训练。

## ResNet18+CIFAR-100 全局大权重标量 baseline

| artifact_id | 来源比例 | protected scalars | unit | 参数比例 | 分类头 | best epoch | accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `large_01` | 0.01 | 112229 | 100 | 1.0411% | exposed | 53 | 0.3637 | 0.4204 | 1.486967 |
| `large_02` | 0.10 | 1122291 | 101 | 10.0382% | mixed | 87 | 0.2902 | 0.3147 | 1.940470 |
| `large_03` | 0.30 | 3366873 | 101 | 30.0296% | mixed | 96 | 0.2251 | 0.2419 | 2.474480 |
| `large_04` | 0.50 | 5611456 | 101 | 50.0209% | mixed | 93 | 0.1888 | 0.1976 | 2.716675 |
| `large_05` | 0.70 | 7856038 | 101 | 70.0121% | mixed | 93 | 0.1663 | 0.1725 | 2.899945 |
| `large_06` | 0.80 | 8978329 | 101 | 80.0078% | mixed | 64 | 0.1545 | 0.1637 | 2.974200 |
| `large_07` | 0.90 | 10100620 | 101 | 90.0034% | mixed | 92 | 0.1446 | 0.1525 | 3.019584 |
| `large_08` | 0.95 | 10661766 | 101 | 95.0012% | mixed | 44 | 0.1356 | 0.1424 | 3.095049 |

大权重扫描在当前单 seed 下保持 accuracy/fidelity 严格下降、posterior KL 严格上升。约 `1%` 参数比例的 `large_01` 已明显强于成本相近的 `shallow_04`，但接近 soft 黑盒需要保护约 80% 以上参数；`large_08` 略越过 soft 黑盒属于攻击训练与选模波动，不应解释为部分保护拥有比黑盒更低的信息量。
