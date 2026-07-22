# 实验 07：BN Gamma 分组闭包

本实验固定 Lab06 十种子候选中的五个前中层 `conv1.weight` 与完整分类头，只改变
20 个 BN gamma 的功能分组，分别回答各组在完整闭包中的必要性和单组加入时的
充分性。候选来源固定为 `results/lab/06_weight/candidate.json`。

四个互斥分组为：

```text
stem          bn1.weight，1 个 state、64 个参数
block_bn1     八个 BasicBlock 的 bn1.weight，8 个 state、1,920 个参数
block_bn2     八个 BasicBlock 的 bn2.weight，8 个 state、1,920 个参数
downsample    三个 downsample.1.weight，3 个 state、896 个参数
```

基础保护集合固定为五个 `conv1.weight` 与 `last_linear.weight/bias`。`drop.py`
使用 seed 43–52 比较 No gamma、All gamma 及四个 leave-one-group-out 配置；完全
相同的 All gamma 和 matched soft 黑盒从 Lab06 candidate 复用。`add.py` 使用
seed 42 从 No gamma 出发分别只加入一组 gamma，不声明跨 seed 稳定性。

两项训练均使用 500 条 soft query 的固定 400/100 train/validation 划分、
`formal_victim_then_public_v1` 初始化、最多 100 epoch、SGD 与 StepLR，并按
query-validation soft cross-entropy 选择最早 best；checkpoint 固定后只在完整
`eval_ms` 上评估一次。

## Feature Conv、Downsample Conv 与 Stem BN1

`feature.py` 追加一个 seed-42 单次诊断。PG03 Feature main Conv Top-5 恰好与本 Lab
固定的五个 `conv1.weight` 集合相同；在它们和完整替换分类头上，再保护三个 stage
切换位置的 downsample Conv weight 与一个 Stem BN gamma：

```text
layer2.0.downsample.0.weight
layer3.0.downsample.0.weight
layer4.0.downsample.0.weight
bn1.weight
```

这里“一个 BN1”明确指顶层 Stem `bn1.weight`，不是八个 BasicBlock BN1 中任选
一个。总保护集合为 11/122 个完整 tensor unit、813,220/11,227,812 个参数，比例
7.2429%。Feature Conv Top-5 基线只读复用 PG05 的同 seed、同 query、同初始化和同
训练协议结果；新增组合按相同 400/100 soft-query validation-best 协议训练，固定
checkpoint 后只遍历一次完整 `eval_ms`。图中同时展示正式 soft 与 hard 黑盒参考线。

## 运行

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/drop.py --dry-run
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/drop.py
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/drop.py --resume
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/add.py --dry-run
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/add.py
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/feature.py --dry-run
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/feature.py
```

## 输出

```text
results/lab/07_bn/drop.json
results/lab/07_bn/drop.tsv
results/lab/07_bn/drop_history.tsv
results/lab/07_bn/drop.png
results/lab/07_bn/drop_<case>_mask.pt
results/lab/07_bn/add.json
results/lab/07_bn/add.tsv
results/lab/07_bn/add_history.tsv
results/lab/07_bn/add.png
results/lab/07_bn/add_<case>_mask.pt
results/lab/07_bn/feature.json
results/lab/07_bn/feature.tsv
results/lab/07_bn/feature_history.tsv
results/lab/07_bn/feature.png
results/lab/07_bn/feature_mask.pt
```
