# MS 数据协议

本目录保存 Model Stealing 的可复现数据划分、伪标签和 posterior。顶层仅保存四个数据集目录及本说明文件；query 不单独落盘，而由各数据集 `manifest.json` 的 `query.split` 指向 `splits.tsv` 的 `query_pool_ms`。

当前协议为 `reference_random_overlap`：受害者模型使用官方训练集全量训练，再从同一训练集均匀随机无放回抽取 1% 作为 query pool。query pool 与 victim 训练集完全重叠，不按类别配额抽取。

```text
dataset/MS/
  c10/                         CIFAR-10 协议和模型标签
  c100/                        CIFAR-100 协议和模型标签
  s10/                         STL10 协议和模型标签
  t200/                        Tiny-ImageNet-200 协议和模型标签
```

每个数据集目录的基础产物：

```text
manifest.json                  划分来源、样本数和预算
splits.tsv                     victim_train、query_pool_ms、eval_ms 的索引
```

模型查询后的标签产物位于 `dataset/MS/<dataset>/<model>/`：

```text
manifest.json                  victim best.pth 的可追溯信息
labels.tsv                     伪标签和置信度
posteriors.pt                  softmax posterior 与 hard pseudo label
```

普通 victim 使用模型名作为 `<model>`，例如 `resnet18`。独立 defended victim 使用稳定标识；当前 TEESlice 的 `ResNet18+C100` 查询产物固定写入 `dataset/MS/c100/teeslice_r18/`，不会覆盖普通 ResNet18 的标签。

`splits.tsv` 内的 `query_rank` 定义 query 的固定顺序。后续 surrogate 对任意预算都必须使用前 `budget` 条 query，且预算不得超过本数据集 `manifest.json` 的 `query.max_budget`。

每次 surrogate 运行再在当前预算前缀内部按实验 seed 做一次固定 80/20 划分：

```text
query train        80%，只用于梯度更新
query validation   20%，只用于选择 validation loss 最低的 checkpoint
eval_ms            不参与选模，checkpoint 固定后只做一次最终评估
```

划分使用 `fixed_seeded_random_partition_of_query_rank`，随机流为实验 seed 加固定
offset 100；训练、validation 的 rank、source index 与 SHA256 写入每次结果元数据。
该划分是从 `query_pool_ms` 动态派生的攻击协议状态，不新增 `query.tsv`，也不在
`splits.tsv` 中伪造新的数据来源。当前计划预算都能被 5 整除，因此 validation
比例严格为 20%。

模型 posterior 使用确定性的 test transform 在原始 query 图像上生成。soft posterior 训练必须复用同一 transform，保证每个 posterior 始终绑定到实际被送入 surrogate 的同一图像。正式 `hard_blackbox` 为隔离输出能力差异同样固定使用 test transform；随机训练增强只允许用于另行明确声明的 hard-label Lab 消融。

这里的“评估隔离”特指 `eval_ms` 不再参与 surrogate checkpoint 选择。当前
`reference_random_overlap` 仍按基准仓库协议允许 query pool 来自 victim 训练集；
不能把它描述成 victim 训练数据与 query 数据互斥。
