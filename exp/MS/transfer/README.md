# MS 划分与受害者查询

本目录实现 MS 的固定数据协议和伪标签生成。协议采用参考开源工作中使用的同源随机重叠方式：victim 在官方训练集全量训练，`query_pool_ms` 从同一训练集均匀随机、无放回抽取。因此 query pool 是 `victim_train` 的子集，不按类别分层，也不要求类别覆盖。

query pool 固定为训练集大小的 1%。`splits.tsv` 的 `query_rank` 固化了查询顺序；后续每个预算都使用该顺序的前缀，保证可嵌套比较。

## 数据划分

生成四个数据集的划分并验证：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/transfer/prepare_splits.py all
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" verify/verify_ms_splits.py
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

先完成对应 victim 训练。普通 victim 不单独划分 validation split；训练入口沿用两个参考仓库的风格，在 `eval_ms` 上逐轮评估并保存 accuracy 最高的 `best.pth`。查询阶段只允许加载该 `best.pth`，不会使用最后一个 epoch 的 `end.pth`。这里的 victim 选模规则不适用于 surrogate：正式 surrogate 在 query 内部划分出的 validation subset 上按 loss 选择最早的最优 `best.pth`，再到 `eval_ms` 完整评估一次，不保存 surrogate `end.pth`。

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/transfer/get_label.py resnet18 c100
```

TEESlice defended victim 使用独立模型标识，训练完成后按相同 `query_pool_ms` 协议查询：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/transfer/get_label.py teeslice_r18 c100
```

需要显式指定权重或覆盖已有标签时：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/transfer/get_label.py resnet18 c100 \
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

`labels.tsv` 与 `query_pool_ms` 具有相同的 `query_rank`、`record_id` 和公开训练集索引。`posteriors.pt` 保存同一顺序下的 softmax posterior 与 hard pseudo label。victim 查询固定使用各数据集的确定性 test transform，并在模型级 `manifest.json` 和 `posteriors.pt` 中记录 `input_transform=test`；后续 soft posterior 训练必须使用同一变换，不得叠加随机裁剪或翻转。

`teeslice_r18` 默认读取 `weights/MS/victim/teeslice_r18/c100/best.pth`，输出写入 `dataset/MS/c100/teeslice_r18/`。该入口只扩展可查询模型，不改变基础 split 或 query 顺序；manifest 会将其标记为 `standalone_reproduction`，避免混入固定普通 victim 的 baseline 汇总。
