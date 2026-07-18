# 实验 02：分类头与权重训练方式对比

本实验在 `ResNet18+CIFAR-100` 下对比全保护和随机保护两种设置，并比较分类头构造方式与可用权重训练方式：

```text
replace_frozen    用 Linear(512, 100) 替换分类头，并冻结指定权重
replace_finetune  用 Linear(512, 100) 替换分类头，全模型共同训练
adapter_frozen    保留 Linear(512, 1000)，追加 Linear(1000, 100)，并冻结指定权重
adapter_finetune  保留上述两层分类头，全模型共同训练
```

全保护不复制 victim 权重。其 `frozen` 变体冻结公开 ImageNet 参数、公开分类头和 BatchNorm 运行状态，只训练目标任务分类头。随机保护以 122 个 tensor unit 为基础：从骨干 `0-119` 中按 seed 42 固定随机保护 59 个 unit，分类头 `120-121` 固定保护且不参与抽取，总计保护 `61/122`；未保护 unit 从 victim 复制。随机保护的 `frozen` 变体只冻结这些已暴露状态，受保护状态的公开初始化与目标任务分类头仍可训练。

## 固定 MS 协议

```text
query budget       query_pool_ms 固定前 500 条
攻击者输出         victim soft posterior
query 划分         seed 42、offset 100 固定拆为 400 train / 100 validation
query transform    确定性的 test transform
surrogate 初始化   formal_victim_then_public_v1
训练               最多 100 epoch，batch size 64
优化器             SGD，lr=0.01，momentum=0.5，weight_decay=5e-4
调度               StepLR，step_size=60，gamma=0.1
选模               validation soft cross-entropy 最低的最早 epoch
最终评估           checkpoint 固定后仅在完整 eval_ms 上评估一次
```

八组配置都从相同 seed 和相同 canonical 构造前缀开始。`replace` 与 `adapter` 只在此前缀之后创建各自任务头，避免把调用顺序导致的随机初始化差异混入分类头比较。

## 运行方式

先核对输入、随机保护集合和八组模型构造：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/02_head/run.py --dry-run
```

完整重跑八组：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/02_head/run.py
```

## 输出

```text
results/lab/02_head/metrics.json  八组保护计划、选模信息与单次 eval_ms 原始指标
results/lab/02_head/history.tsv   八组逐 epoch 的 query train/validation 记录
```
