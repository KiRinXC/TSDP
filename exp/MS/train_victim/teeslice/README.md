# TEESlice 独立复现

本目录独立复现 TEESlice defended victim。TEESlice 会改变输入 stem、模型容量和训练目标，因此不与固定普通 victim 上的 TensorShield、完整层保护或通道保护结果直接排序，也不写入主 baseline 汇总。数据来源、query 顺序、预算和最终评估仍遵循项目统一 MS 协议。

当前只固化 `ResNet18+CIFAR-100`。实现参考 `Demo/TEESlice-artifact` 的 commit `93505cb3337ec8b89556ee29ffc598d31513aa5e`；作者代码中的 `NetTailor` 对应论文 TEESlice。

## 固定协议

```text
公开初始化          weights/pre_train/resnet18-5c106cde.pth
训练数据            victim_train
内部选择数据        从 victim_train 固定划出 10%
最终评估数据        eval_ms，只在阶段选择完成后评估
source 结构         CIFAR ResNet18：3x3 stride-1 stem，无 maxpool
公开部分            冻结的 CIFAR ResNet18 universal backbone
私有部分            proxy slice、路径 alpha、C100 分类头和任务适配 BN buffer
随机种子            42
```

内部 90/10 只服务于 source、teacher、full model 和剪枝终点选择，不写入 `dataset/MS/c100/splits.tsv`，也不改变通用 MS 划分。`eval_ms` 不参与 checkpoint 或剪枝拓扑选择。

训练分为四个阶段：

```text
source   20 epoch  使用真实标签训练作者的 CIFAR-stem ResNet18；StepLR(10, 0.1)
teacher  20 epoch  将 source soft posterior 蒸馏到同结构 teacher；训练期间 lr 固定为 0.01
full     40 epoch  冻结 universal backbone，训练全部 proxy、alpha 和分类头；第 31 轮降至 0.01
prune    20 epoch  以 full/best 为准确率基准，固定 lr=0.01 迭代删除低 alpha proxy
```

source 使用 batch size 64、SGD、学习率 `0.1`、momentum `0.5`、weight decay `5e-4`。teacher 使用学习率 `0.01`、momentum `0.5`、weight decay `4e-4` 和 KD temperature `4`；full 使用学习率 `0.1`、momentum `0.9`、weight decay `4e-4`；prune 使用学习率 `0.01`、momentum `0.9`、weight decay `5e-4`。

固定方法参数为 `max_skip=3`、`complexity_coeff=0.3` 和 `teacher_coeff=10.0`。full 阶段同时优化输出 KD、中间特征 MSE 和复杂度项。作者实现冻结 universal backbone 参数，但允许其 BN buffer 随任务数据更新；这些状态作为私有状态保存，不计入论文 task parameter 或 task FLOPs。

动态剪枝先删除 alpha 不高于 `0.1` 的 proxy，并至少纳入按 `int(proxy_count * 0.5)` 向下取整得到的最低分 proxy；当前 21 个 proxy 对应首次至少删除 10 个。之后每 2 epoch 检查内部验证准确率；只有当前模型满足

```text
pruned_accuracy > full_accuracy * (1 - 0.01)
```

才保存为 `last_tolerable` 并继续按照作者的全局 alpha 阈值语义删除最低 5% 原始 proxy。最终在 `last_tolerable` 和训练终点中选择满足容忍条件且准确率更高的模型；若没有剪枝候选满足条件，则回退到 `full/best.pth`。

作者发布的 `ResNet18+C100` 拓扑对应 `711,524` 个 task parameter 和 `29,868,032` task FLOPs。本实现保存该拓扑作为结构核对项，但正式拓扑由内部验证剪枝产生，不使用 `eval_ms` 强制匹配。

## 运行方式

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_victim/teeslice/train.py resnet18 c100
```

只验证输入、模型结构和输出协议：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_victim/teeslice/train.py resnet18 c100 --dry-run
```

覆盖当前同语义产物时显式增加 `--overwrite`。重新训练会清除旧的 TEESlice query、surrogate 和结果，不影响普通 ResNet18 或其他 baseline。

## 输出

```text
weights/MS/victim/teeslice_r18/c100/
├── source/{best,end}.pth        CIFAR-stem source victim
├── teacher/{best,end}.pth       蒸馏 teacher；full 固定读取 end.pth
├── full/{best,end}.pth          未剪枝 TEESlice
├── best.pth                     满足容忍条件的最终 pruned victim，供 query 使用
├── end.pth                      prune 阶段训练终点
├── params.json
└── train.log.tsv

results/MS/resnet18/c100/teeslice/victim.json
```

`victim.json` 分别保存 source、teacher、full 和 pruned 的原始效用指标，以及 `full -> pruned` 的准确率变化、容忍判断和剪枝前后成本。完成本阶段后使用 `exp/MS/transfer/get_label.py teeslice_r18 c100` 生成 query posterior。
