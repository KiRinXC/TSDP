# MS 划分与受害者查询

本目录实现 MS 的固定数据协议和伪标签生成。协议采用参考开源工作中使用的同源随机重叠方式：victim 在官方训练集全量训练，`query_pool_ms` 从同一训练集均匀随机、无放回抽取。因此 query pool 是 `victim_train` 的子集，不按类别分层，也不要求类别覆盖。

query pool 固定为训练集大小的 1%。`splits.tsv` 的 `query_rank` 固化了查询顺序；后续每个预算都使用该顺序的前缀，保证可嵌套比较。

## 数据划分

生成四个数据集的划分并验证：

```bash
python3 exp/MS/transfer/prepare_splits.py all
python3 verify/verify_ms_splits.py
```

各数据集的固定配置如下：

```text
数据集  victim_train  query_pool_ms  后续 surrogate 预算
c10     50,000        500            50, 100, 300, 500
c100    50,000        500            50, 100, 300, 500
s10     5,000         50             50
t200    100,000       1,000          50, 100, 300, 500, 1,000
```

`victim_train` 使用官方训练集全量；`eval_ms` 使用官方 test 或 val 全量。STL10 的 unlabeled split 不参与本协议。

输出统一位于：

```text
dataset/MS/<dataset>/
  manifest.json
  splits.tsv
```

其中 `manifest.json` 的 `query.split` 始终指向 `splits.tsv` 中的 `query_pool_ms`，不会生成独立的 query 文件。

## 生成伪标签

先完成对应 victim 训练。查询阶段只允许加载最佳验证模型 `best.pth`，不会使用最后一个 epoch 的 `end.pth`。

```bash
python3 exp/MS/transfer/get_label.py resnet18 c100
```

需要显式指定权重或覆盖已有标签时：

```bash
python3 exp/MS/transfer/get_label.py resnet18 c100 \
  --checkpoint weights/MS/victim/resnet18/c100/best.pth \
  --overwrite
```

输出位于模型层级，便于后续 surrogate 训练直接消费：

```text
dataset/MS/<dataset>/<model>/
  manifest.json
  labels.tsv
  posteriors.pt
```

`labels.tsv` 与 `query_pool_ms` 具有相同的 `query_rank`、`record_id` 和公开训练集索引。`posteriors.pt` 保存同一顺序下的 softmax posterior 与 hard pseudo label。
