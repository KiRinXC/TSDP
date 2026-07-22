# 实验 02 结果

本目录记录 `ResNet18+CIFAR-100` 的分类头结构与权重训练方式对比。八组均使用
500 条 soft-posterior query，其中 400 条训练、100 条按 soft cross-entropy 选择
最早的最优 checkpoint；checkpoint 固定后只在 10,000 条 `eval_ms` 上评估一次。

## 实验结论

原八组对比说明，在全参数共同微调这一主攻击设定下，直接替换为
`Linear(512,100)` 的 `replace` 在全保护和随机保护中都得到更高 accuracy、fidelity
和更低 posterior KL，即比保留 ImageNet-1000 头再追加 adapter 的攻击更强。因此
后续普通 MS 统一使用替换分类头，不把 adapter 引入的结构失配误当成保护效果。

冻结实验说明，public/victim 拼接后的训练权限会实质改变攻击结果。尤其在随机保护中，
冻结已复制的 victim 暴露状态会让攻击几乎失效；这反映的是不相容初始化无法被修正，
不能取代允许攻击者联合微调全部可用状态的主威胁模型。

TensorShield Top-10 的三组 seed-42 对照进一步确认了这一点：冻结 public 保护状态、
训练 victim 暴露状态时攻击为 `0.1835/0.1986/2.655572`；反向冻结 victim 暴露状态时
降到 `0.1203/0.1285/3.070843`；两侧共同训练则恢复正式 Top-10 的
`0.1728/0.1865/2.694492`。所以在相同保护集合下，“哪一侧可训练”与保护参数量同样
重要。该 trainability 排序目前只由 seed 42 支持，不能写成十随机种子结论。

全保护不复制 victim 权重，冻结组只训练分类头或 adapter：

```text
配置              可训练参数  best epoch  accuracy  fidelity  posterior KL
replace_frozen       51300       100       0.1724    0.1820    2.733553
replace_finetune  11227812        45       0.1390    0.1463    3.039817
adapter_frozen      100100        39       0.1794    0.1866    3.132601
adapter_finetune  11789612        76       0.1292    0.1319    3.405513
```

随机保护固定保护 `61/122` 个 unit，分类头 weight 与 bias 固定保护；冻结组冻结从
victim 复制出的暴露权重：

```text
配置              可训练参数  best epoch  accuracy  fidelity  posterior KL
replace_frozen     4444900          1       0.0131    0.0143    5.790709
replace_finetune  11227812         53       0.1394    0.1501    2.914578
adapter_frozen     5006700          1       0.0121    0.0140    9.763113
adapter_finetune  11789612         96       0.1345    0.1433    3.193306
```

在两种保护范围的“全部参数共同微调”对照中，`replace` 的 accuracy、fidelity 和
posterior KL 均优于 `adapter`，因此后续普通 MS 继续使用直接替换分类头。冻结对照
不呈现统一的三指标支配关系；尤其随机拼接 victim/public 状态后，冻结暴露权重会使
攻击几乎失效，说明该对照不能替代允许全模型联合微调的攻击者。

```text
metrics.json  八组 validation-best 选择、单次 eval_ms 指标与保护计划
history.tsv   八组配置共 800 条 query-train/query-validation 日志
```

## TensorShield Top-10 trainability 结果

该结果只包含 seed 42，不是十随机种子实验。三组使用相同的 Top-10 保护集合、相同的
初始模型哈希 `96db121132c0eb8519e327f3ee9f5682a22d8642e6f297a09aa2ae460b7ae071`
和始终可训练的替换分类头；差别仅为 public protected state 与 victim exposed state
是否参与训练。

```text
配置                         可训练 public  可训练 victim  可训练 head  best epoch  accuracy  fidelity  posterior KL
public_frozen_victim_train             0       10218048         51300          93    0.1835    0.1986    2.655572
public_train_victim_frozen        958464              0         51300          77    0.1203    0.1285    3.070843
joint_finetune                    958464       10218048         51300          93    0.1728    0.1865    2.694492
```

`joint_finetune` 与 Lab04 的正式 TensorShield Top-10 在 best epoch 和三个指标上逐值
一致，说明本实验没有改变 Top-10 的初始化与主训练协议。在本次 seed-42 消融中，冻结
victim exposed state、仅训练 public protected state 与替换头的攻击效果最弱；该结论
只描述当前单种子结果，不能外推为多随机种子统计结论。

```text
top10_trainability.json         三组 trainability、共同初始化、选模与单次 eval_ms 指标
top10_trainability.tsv          三组参数计数与最终指标
top10_trainability_history.tsv  三组共 300 条 query-train/query-validation 日志
top10_trainability.png          三指标柱状图及正式 no/full/hard 参考线
```
