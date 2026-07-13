# TEESlice 受害者训练

本目录训练并剪枝 TEESlice defended victim。TEESlice 不是在普通 victim 的 `state_dict` 上隐藏部分权重，而是基于公开 ImageNet backbone 构造任务相关 private slice，再经过 teacher 蒸馏、full model 训练和迭代剪枝产生独立的可查询模型。

当前只固化 `ResNet18+CIFAR-100`。实现参考 `Demo/TEESlice-artifact` 的 commit `93505cb3337ec8b89556ee29ffc598d31513aa5e`；作者代码中的模型名 `NetTailor` 对应论文 TEESlice。

## 固定协议

```text
原始 victim         weights/MS/victim/resnet18/c100/best.pth
公开预训练权重      weights/pre_train/resnet18-5c106cde.pth
训练数据            victim_train
内部选择数据        从 victim_train 固定划出 10%
最终评估数据        eval_ms，只用于最终 defended victim 评估
公开部分            CIFAR ResNet18 universal backbone
私有部分            proxy slice、路径 alpha、C100 分类头和适配后的 BN buffer
随机种子            42
```

内部 90/10 只服务于 TEESlice 的模型选择和剪枝，不写入 `dataset/MS/c100/splits.tsv`，也不改变通用 MS 划分。`eval_ms` 不参与 teacher、full model 或剪枝终点选择。

训练分为三个阶段：

```text
teacher  20 epoch  将普通 victim 蒸馏到 CIFAR ResNet18 teacher；StepLR(20, 0.1)
full     40 epoch  冻结 universal backbone，训练 proxy、alpha 和分类头；StepLR(30, 0.1)
prune    20 epoch  按内部验证准确率容忍范围迭代删除低 alpha proxy；CosineAnnealingLR
```

固定方法参数为 `max_skip=3`、`complexity_coeff=0.3`、`teacher_coeff=10.0`、KD temperature `4`。full 阶段使用 batch size 64、SGD、学习率 `0.1`、momentum `0.9` 和 weight decay `4e-4`；prune 阶段学习率为 `0.01`，初始删除 alpha 不高于 `0.1` 的 proxy 且至少删除 50%，之后每 2 epoch 删除候选 proxy 中最低的 5%。

作者实现冻结 universal backbone 参数，但允许其 BN `running_mean`、`running_var` 和 `num_batches_tracked` 随任务数据更新。本实现保留该行为，并将这些状态单独计入 `private_bn_buffer_count`，不计入论文 task parameter 或 task FLOPs。

作者发布的 `ResNet18+C100` 拓扑对应 `711,524` 个 task parameter 和 `29,868,032` task FLOPs。本实现保存该拓扑作为核对项，但正式 defended victim 仍由内部验证剪枝规则产生，不使用 `eval_ms` 强制匹配作者拓扑。

## 运行方式

先完成普通 victim 训练，然后运行：

```bash
python3 exp/MS/train_victim/teeslice/train.py resnet18 c100
```

只验证输入、模型结构和输出协议：

```bash
python3 exp/MS/train_victim/teeslice/train.py resnet18 c100 --dry-run
```

覆盖当前同语义产物时显式增加 `--overwrite`。重新训练 defended victim 会清除旧的 TEESlice query、surrogate 和 MS 结果，避免下游继续消费失效 checkpoint。

## 输出

```text
weights/MS/victim/teeslice_r18/c100/
├── teacher/best.pth
├── teacher/end.pth
├── full/best.pth
├── full/end.pth
├── best.pth                  内部验证选择的最终 defended victim，供 query 使用
├── end.pth                   prune 阶段固定训练终点
├── params.json
└── train.log.tsv

results/MS/resnet18/c100/teeslice/victim.json
```

完成本阶段后，使用 `exp/MS/transfer/get_label.py teeslice_r18 c100` 生成查询 posterior，再由 `exp/MS/train_surrogate/teeslice/attack.py` 训练攻击 surrogate。
