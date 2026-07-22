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

## 运行

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/drop.py --dry-run
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/drop.py
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/drop.py --resume
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/add.py --dry-run
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/07_bn/add.py
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
```
