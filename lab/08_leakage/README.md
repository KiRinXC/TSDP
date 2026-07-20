# 实验 08：泄露状态利用强度与攻击负迁移

本实验回答一个与保护位置选择不同的问题：攻击者已经获得未保护的真实 victim
状态后，为什么直接使用这些状态进行初始化，反而可能比完全忽略它们的 soft 黑盒
攻击泛化更差。

这里的自变量不是泄露 tensor 的数量，也不是信息论意义上的泄露比例，而是攻击者
对同一批已泄露状态的利用强度。系统保护集合始终固定为 Lab04 最终的 5.7529%
候选：eligible rank-1/2/3/4/7/9、固定分类头 bias 和全部 20 个 BN gamma。

## 固定协议

攻击者同时知道官方 public 状态和所有未保护的 victim 状态。对于未保护的浮点状态，
初始化沿 public 到 victim 的方向取五个预先固定的利用强度：

```text
0%      完全忽略泄露状态，等价于 matched soft 黑盒初始化
25%     使用四分之一 public→victim 状态差
50%     使用二分之一 public→victim 状态差
75%     使用四分之三 public→victim 状态差
100%    完整使用泄露状态，等价于当前 5.7529% 混合初始化
```

受保护状态在五个强度下始终保持同 seed 的 public/随机初始化。浮点 buffer 与浮点
parameter 使用同一插值；`num_batches_tracked` 等非浮点状态在中间强度固定使用
public 值，避免制造没有数值含义的整数插值。0% 和 100% 端点必须分别与 Lab04
matched soft 黑盒和 5.7529% 候选逐状态一致。

实验使用 seed 43–52。每个 seed 和每个强度都重放
`formal_victim_then_public_v1` canonical 初始化，使用同一组 500 条 soft query
及该 seed 固定的 400/100 query train/validation 划分。所有 surrogate 参数共同
finetune，最多 100 epoch，SGD `lr=0.01`、`momentum=0.5`、
`weight_decay=5e-4`，`StepLR(step_size=60, gamma=0.1)`。严格按 query-validation
soft cross-entropy 最低点选择最早 `best`，checkpoint 固定后只在完整 `eval_ms`
上评估一次。

0% 和 100% 的训练与最终指标直接复用 Lab04 同 seed、同协议结果；只新增
25%/50%/75% × 十 seed 的三十组训练。所有五个强度都重新构造未经训练的模型，
在 query train 和 query validation 上执行 epoch-0 只读探针，以区分初始化失配
与训练后的泛化负迁移。

`eval_ms` 不参与 epoch、利用强度或攻击超参数选择。另行报告一个适应性攻击者：
它只根据各自 query-validation 最低 soft cross-entropy，在五个强度中选择一个，
并读取该预先固定 checkpoint 的一次最终评估。数值并列时选择更低利用强度。

## 判定逻辑

实验首先比较同 seed 下利用强度相对 0% 黑盒的变化：

- 若 query-train 拟合相近，而 query-validation 与 `eval_ms` 随利用强度增加而
  变差，则支持“泄露状态诱导负迁移”；
- 若中间强度优于 0%，说明攻击者可以通过主动缩回泄露状态获得黑盒以上收益；
- 若只有 100% 端点变差，则需要继续检查端点附近的非线性或优化器敏感性；
- 若利用强度越高攻击越强，则当前“误导”解释不成立。

本实验只验证混合初始化是否造成负迁移，不回答为什么固定集合中的具体位置形成这种
不兼容；位置机制留给后续独立实验。

## 运行

先核对来源、十种子 query 划分、固定保护集合和五个强度的端点/插值状态：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/08_leakage/run.py --dry-run
```

运行三十组新增训练与全部 epoch-0 探针：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/08_leakage/run.py
```

若运行中断：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/08_leakage/run.py --resume
```

输出固定为：

```text
results/lab/08_leakage/metrics.json
results/lab/08_leakage/data.tsv
results/lab/08_leakage/history.tsv
results/lab/08_leakage/probe.tsv
results/lab/08_leakage/metrics.png
```
