# MS 原始结果

本目录只保存 surrogate 在 `eval_ms` 上的原始评估结果，不保存 accuracy drop、fidelity drop、相对黑盒倍数等派生指标。

```text
results/MS/<model>/<dataset>/
  metrics.tsv                   所有 run 的原始指标索引
  <artifact_id>/metrics.json    单次运行 best 与 end 的原始指标
```

`artifact_id` 是便于定位的语义名称；planned baseline 直接使用 `plan_id`，上下界使用策略名。完整配置哈希保存在 `run_id` 字段中。

正式主保护策略使用 `posterior_replace_finetune_v2`：soft posterior 与 surrogate query 均使用确定性 test transform，训练配置使用 `lr_step=60`，主结果固定读取 `end.pth`。另保留 `ResNet18+C100` 全保护下的 `hard_label_replace_finetune_v1` 输出能力对比；它只改变 victim 的可观测输出，不替换 soft-posterior 主黑盒下界。本 README 只记录当前有效结果。

## ResNet18+CIFAR-100 上下界

```text
保护策略         查询输出       训练方式  分类头   artifact_id       end epoch  accuracy  fidelity  posterior KL
no_protection    soft posterior  identity  exposed  no_protection    0          0.6182    1.0000    0.0000000011
full_protection  soft posterior  finetune  replace  full_protection  100        0.1545    0.1610    2.835290
full_protection  hard label      finetune  replace  hard_blackbox    100        0.1393    0.1443    3.427757
```

`hard_blackbox` 的 run ID 为 `c9a62938cfcf`，`comparison_scope` 为 `ordinary_fixed_victim_output_ablation`。它与 soft `full_protection` 使用相同 victim、500 条 query、完整保护 mask、初始化和训练超参数，只将 posterior 改为 argmax hard label；正式结果读取第 100 轮 `end.pth`，第 88 轮 `best.pth` 仅作诊断。

## ResNet18+CIFAR-100 仅分类头保护结果

`head_only` 只隐藏 `last_linear.weight` 和 `last_linear.bias`，完整复制普通 victim 的其余 backbone 状态。最终 mask 保护 `2/122` 个 unit、`51,300/11,227,812` 个参数，参数比例为 `0.4569%`，分类头使用 `replace`，随后按统一协议对全部参数共同 finetune。

| checkpoint | epoch | accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|
| `end.pth`（主结果） | 100 | 0.4404 | 0.5135 | 1.347578 |
| `best.pth`（训练诊断） | 77 | 0.4416 | 0.5131 | 1.369571 |

该控制组在仅保护 `0.4569%` 参数时，MS 指标已明显低于参数比例相近的 `shallow_02`（`0.4144%`，accuracy/fidelity 为 `0.5651/0.7280`），说明分类头不可见本身对当前攻击有显著影响；但它仍明显未达到普通 victim 的 `full_protection` 参考结果，因此不能用“只保护分类头”替代后续关键路径保护。正式比较只读取 `end.pth`，`best.pth` 仅作诊断。

## ResNet18+CIFAR-100 TEESlice 结果

TEESlice 改变了 victim 结构与训练过程，因此按 `standalone_reproduction` 独立保存，不写入固定普通 victim 的主 `metrics.tsv`。本次剪枝模型在 `eval_ms` 上的 accuracy 为 `0.7578`；已知最终剪枝拓扑的黑盒攻击使用 500 条 soft posterior query，在固定第 100 轮得到 accuracy `0.1619`、fidelity `0.1784`、posterior KL `3.251131`。完整状态白盒的实际评估为 accuracy `0.7578`、fidelity `1.0000`、posterior KL `4.814928395546758e-10`。四阶段效用、剪枝成本和详细指标见 `results/MS/resnet18/c100/teeslice/README.md`。

当前总图同时展示 `head_only`、TensorShield、TEESlice、浅层/中间层/深层/大权重四类策略，以及普通预训练模型的 no/full 主参考界和 hard-label 全保护辅助参考线。TEESlice 点必须保留 `standalone_reproduction` 标记，只用于呈现其自身攻击区间，不参与固定普通 victim 下的同条件策略排序；hard-label 参考线也不参与 soft 主保护策略排序。

## ResNet18+CIFAR-100 TensorShield 结果

TensorShield 使用作者确认 rank 对应的 Figure 12(d) 固定保护集合，不重新计算 importance。最终 mask 保护 `11/122` 个 unit、`1,009,764/11,227,812` 个参数，参数比例为 `8.9934%`。

| checkpoint | epoch | accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|
| `end.pth`（主结果） | 100 | 0.1913 | 0.2099 | 2.505831 |
| `best.pth`（训练诊断） | 90 | 0.1916 | 0.2101 | 2.510781 |

主结果固定使用 `end.pth`；`best.pth` 不参与正式策略比较。

## ResNet18+CIFAR-100 完整层 baseline

| 策略 | 层数 | 官方层范围 | unit | 参数比例 | 分类头 | artifact_id | accuracy | fidelity | posterior KL |
|---|---:|---|---:|---:|---|---|---:|---:|---:|
| 浅层 | 2 | 1-2 | 12 | 0.4144% | exposed | `shallow_02` | 0.5651 | 0.7280 | 0.389144 |
| 中间层 | 2 | 9-10 | 18 | 4.2432% | exposed | `middle_02` | 0.4429 | 0.5204 | 1.058431 |
| 深层 | 2 | 17-18 | 8 | 21.4790% | replace | `deep_02` | 0.4567 | 0.5429 | 1.000962 |
| 浅层 | 4 | 1-4 | 24 | 1.0733% | exposed | `shallow_04` | 0.5560 | 0.6992 | 0.471333 |
| 中间层 | 4 | 8-11 | 30 | 10.8166% | exposed | `middle_04` | 0.3446 | 0.3833 | 1.615993 |
| 深层 | 4 | 15-18 | 20 | 63.5232% | replace | `deep_04` | 0.3821 | 0.4325 | 1.444718 |
| 浅层 | 6 | 1-6 | 42 | 2.1370% | exposed | `shallow_06` | 0.5177 | 0.6220 | 0.748295 |
| 中间层 | 6 | 7-12 | 42 | 17.3900% | exposed | `middle_06` | 0.3068 | 0.3422 | 1.804017 |
| 深层 | 6 | 13-18 | 38 | 80.4731% | replace | `deep_06` | 0.2589 | 0.2870 | 2.070988 |
| 浅层 | 8 | 1-8 | 54 | 4.7682% | exposed | `shallow_08` | 0.4812 | 0.5519 | 0.966082 |
| 中间层 | 8 | 6-13 | 60 | 23.3819% | exposed | `middle_08` | 0.2379 | 0.2580 | 2.265572 |
| 深层 | 8 | 11-18 | 50 | 90.9887% | replace | `deep_08` | 0.2113 | 0.2298 | 2.398826 |
| 浅层 | 10 | 1-10 | 72 | 9.0113% | exposed | `shallow_10` | 0.3767 | 0.4152 | 1.557270 |
| 中间层 | 10 | 5-14 | 78 | 35.4035% | exposed | `middle_10` | 0.1901 | 0.2088 | 2.428808 |
| 深层 | 10 | 9-18 | 68 | 95.2318% | replace | `deep_10` | 0.1785 | 0.1908 | 2.658920 |
| 浅层 | 12 | 1-12 | 84 | 19.5269% | exposed | `shallow_12` | 0.3134 | 0.3364 | 1.869116 |
| 中间层 | 12 | 4-15 | 90 | 56.7551% | exposed | `middle_12` | 0.1925 | 0.2007 | 2.455368 |
| 深层 | 12 | 7-18 | 80 | 97.8630% | replace | `deep_12` | 0.1433 | 0.1582 | 2.843914 |
| 浅层 | 14 | 1-14 | 102 | 36.4768% | exposed | `shallow_14` | 0.2253 | 0.2388 | 2.260448 |
| 中间层 | 14 | 3-16 | 102 | 78.1066% | exposed | `middle_14` | 0.2009 | 0.2081 | 2.396788 |
| 深层 | 14 | 5-18 | 98 | 98.9267% | replace | `deep_14` | 0.1373 | 0.1452 | 2.954076 |
| 浅层 | 16 | 1-16 | 114 | 78.5210% | exposed | `shallow_16` | 0.2205 | 0.2297 | 2.271671 |
| 中间层 | 16 | 2-17 | 114 | 99.4582% | exposed | `middle_16` | 0.1690 | 0.1735 | 2.866977 |
| 深层 | 16 | 3-18 | 110 | 99.5856% | replace | `deep_16` | 0.1346 | 0.1439 | 2.970565 |

## ResNet18+CIFAR-100 全局大权重标量 baseline

| artifact_id | 来源比例 | protected scalars | unit | 参数比例 | 分类头 | accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `large_01` | 0.01 | 112229 | 100 | 1.0411% | exposed | 0.4090 | 0.4764 | 1.256363 |
| `large_02` | 0.10 | 1122291 | 101 | 10.0382% | mixed | 0.3190 | 0.3562 | 1.767944 |
| `large_03` | 0.30 | 3366873 | 101 | 30.0296% | mixed | 0.2413 | 0.2594 | 2.290452 |
| `large_04` | 0.50 | 5611456 | 101 | 50.0209% | mixed | 0.2003 | 0.2087 | 2.541162 |
| `large_05` | 0.70 | 7856038 | 101 | 70.0121% | mixed | 0.1806 | 0.1845 | 2.697672 |
| `large_06` | 0.80 | 8978329 | 101 | 80.0078% | mixed | 0.1672 | 0.1763 | 2.765886 |
| `large_07` | 0.90 | 10100620 | 101 | 90.0034% | mixed | 0.1591 | 0.1658 | 2.833833 |
| `large_08` | 0.95 | 10661766 | 101 | 95.0012% | mixed | 0.1533 | 0.1600 | 2.837758 |

`large_weight` 的分类头与 backbone 使用同一保护 mask：受保护标量保留随机或公开初始化，未受保护标量复制 victim。当前扫描中保护比例增加时 accuracy 和 fidelity 严格下降，posterior KL 严格上升；横坐标统计的不可见参数与实际初始化行为一致。
