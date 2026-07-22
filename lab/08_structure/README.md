# 实验 08：ResNet18 计算图结构核对

本实验展开当前项目 `models/imagenet.py` 中的 ImageNet-style ResNet18，并核对
CIFAR-100 的 `3×32×32` 输入经过每个实际算子后的特征尺寸。

结构表不仅列出 `layer1` 至 `layer4`，还分别记录 8 个 BasicBlock 内的两次卷积、
两次 BatchNorm、两次 ReLU、identity/downsample 分支和逐元素残差相加。表中省略
batch 维；分类头输出是 100 类 logits，模型内部没有 Softmax。

最左列使用正式 surrogate 防御入口的 122-unit 注册表，即按 `state_dict` 稳定顺序
编号 `0-121`。结果表按一个 unit 一行展开：卷积通常只有一个 weight 行；
BatchNorm 展开为 weight、bias、running mean、running variance 和
`num_batches_tracked` 五行；无模型状态的计算节点标记为 `—`。

`structure.tsv` 保存不重复计算节点的紧凑结构定义；`run.py` 根据当前模型的
`state_dict` 将其展开为结果目录中的逐 unit 表，并同时核对编号和输出尺寸。

结果表 H 列直接读取
`exp/MS/train_surrogate/selector/tensorshield.py` 中固化的
`AUTHOR_RESNET18_C100_ELIGIBLE_RANK`，把全部 17 个 weight 标为 `Top-1` 至
`Top-17`。固定保护的 `last_linear.bias` 不属于该 rank，因此不在 H 列编号。

同一次运行还会从完整结构表抽取这 17 行，按 `unit编号` 升序写入
`results/lab/08_structure/tensorshield_top17.tsv`，同时保留每个 weight 的原始
Top-k 标签，便于沿模型结构检查候选。

运行以下命令可用一个 `1×3×32×32` 的虚拟输入核对所有可挂钩模块的输出尺寸：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/08_structure/run.py
```

核对对象为：

```text
results/lab/08_structure/resnet18_c100.tsv
```

该过程不读取数据集、victim 权重或预训练权重，不执行训练，也不生成模型权重。

## 前中层 conv1 攻击依赖消融

`dependency.py` 固定 Lab06 十随机种子候选验证得到的 5.7529% 基础保护集合：eligible
rank-1/2/4/7/9 的五个前中层 `conv1.weight`、rank-3 分类头 weight、固定分类头
bias，以及全部 20 个 BN gamma。随后分别重新暴露五个 `conv1.weight`，其余保护项
保持不变，以 leave-one-out 检验每个成员的条件攻击依赖。

```text
基础集合          五个 conv1 + 全部 BN gamma + 分类头 weight/bias
expose_rank_01    暴露 layer1.1.conv1.weight
expose_rank_02    暴露 layer2.0.conv1.weight
expose_rank_04    暴露 layer1.0.conv1.weight
expose_rank_07    暴露 layer2.1.conv1.weight
expose_rank_09    暴露 layer3.0.conv1.weight
```

实验固定使用 seed 43–52。每个 leave-one-out case 均重新执行 canonical surrogate
初始化和 400/100 soft-query validation-best 训练：最多 100 epoch，SGD
`lr=0.01`、`momentum=0.5`、`weight_decay=5e-4`，
`StepLR(step_size=60, gamma=0.1)`。checkpoint 固定后只在完整 `eval_ms` 上评估
一次。基础集合和同 seed soft full-protection 黑盒直接读取 Lab06 相同协议候选结果，
不重复训练。

若重新暴露一个成员后，相对同 seed 基础集合同时出现 accuracy 上升、fidelity
上升和 posterior KL 下降，则记为一次攻击反弹。只有该配对方向跨 seed 稳定时，
才把对应成员解释为当前集合中的条件攻击依赖；本实验仍是读取既有 MS 反馈后的
后验机制验证，不是无偏先验选择。

先核对来源、query 划分、六个策略 mask、参数量和分类头模式：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/08_structure/dependency.py --dry-run
```

运行五组 leave-one-out × 十 seed 的完整实验：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/08_structure/dependency.py
```

若运行中断，使用以下命令复用已经完整完成的逐 case 进度：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/08_structure/dependency.py --resume
```

输出为：

```text
results/lab/08_structure/dependency.json
results/lab/08_structure/dependency.tsv
results/lab/08_structure/dependency_history.tsv
results/lab/08_structure/dependency.png
results/lab/08_structure/dependency_*_mask.pt
```

## `conv1` 与对应 `conv2` 的直接保护替换

`swap.py` 只做一个结构消融：固定分类头 weight/bias 和全部 20 个 BN gamma，
将当前基础集合中的五个 `conv1.weight` 一一换成同一 BasicBlock 的
`conv2.weight`：

```text
conv1 组    layer1.0/1.1、layer2.0/2.1、layer3.0 的 conv1.weight
conv2 组    上述五个 BasicBlock 中一一对应的 conv2.weight
```

conv1 组直接复用 Lab06/Lab08 的十种子结果，保护 645,924 个参数，占 5.7529%。
对应 conv2 tensor 更大，conv2 组保护 1,014,564 个参数，占 9.0362%。两组固定
相同五个 BasicBlock、相同五个卷积 tensor、相同分类头和 BN gamma，但不是同参数
成本比较。

只对 conv2 组新增 seed 43–52 的十组100%直接拼接攻击。训练使用相同的400/100
soft query、100 epoch、SGD与StepLR参数，并按query-validation soft
cross-entropy选择最早best；checkpoint固定后只在完整`eval_ms`上评估一次。
图中同时复用同seed conv1组和soft黑盒，不增加利用强度、表征指标或其他case。

该消融只检验“把保护位置从conv1换成对应conv2后，能否维持当前直接攻击下的保护
效果”。它不能单独证明conv1完成坐标转换；只有出现位置差异后，才另行设计表征空间
实验解释原因。

先核对两个保护集合、参数量、mask和十种子来源：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/08_structure/swap.py --dry-run
```

执行十组新增训练：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/08_structure/swap.py
```

若运行中断：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/08_structure/swap.py --resume
```

输出为：

```text
results/lab/08_structure/swap.json
results/lab/08_structure/swap.tsv
results/lab/08_structure/swap_history.tsv
results/lab/08_structure/swap.png
results/lab/08_structure/swap_conv2_mask.pt
```

## 卷积与局部 BN Gamma 配对

`pair.py` 在同一五个 BasicBlock 上比较五个 `conv1.weight+bn2.weight` 与五个
`conv2.weight+bn1.weight`，两组均固定保护完整分类头。它不加入全部 BN gamma、
downsample、BN bias 或运行统计量，也不是相同参数预算比较。

该实验只运行 seed 42，沿用 500 条 soft query 的 400/100 validation-best 协议，
最多训练 100 epoch，checkpoint 固定后各评估一次 `eval_ms`。

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/08_structure/pair.py --dry-run
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/08_structure/pair.py
```

```text
results/lab/08_structure/pair.json
results/lab/08_structure/pair.tsv
results/lab/08_structure/pair_history.tsv
results/lab/08_structure/pair.png
results/lab/08_structure/conv1_bn2_mask.pt
results/lab/08_structure/conv2_bn1_mask.pt
```
