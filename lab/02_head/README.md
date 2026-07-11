# 实验 02：分类头与权重训练方式对比

本实验在 `ResNet18+CIFAR-100` 下对比全保护和随机保护两种设置。两种设置均使用 500 条 hard label query，并比较分类头构造方式和可用权重的训练方式。

四种配置如下：

```text
replace_frozen    用 Linear(512, 100) 替换分类头，并冻结指定权重
replace_finetune  用 Linear(512, 100) 替换分类头，全模型共同训练
adapter_frozen    保留 Linear(512, 1000)，追加 Linear(1000, 100)，并冻结指定权重
adapter_finetune  保留上述两层分类头，全模型共同训练
```

全保护不复制 victim 权重，`frozen` 表示冻结公开 ImageNet 参数及 BatchNorm 运行状态，只训练任务分类头。随机保护以 122 个 tensor unit 为基础：从骨干网络的 `0-119` 中固定随机保护 59 个 unit，分类头 `120-121` 不参与随机抽取并固定保护，总计保护 `61/122`。未保护的 unit 从 victim 复制；此时 `frozen` 只冻结这些复制出的暴露权重，受保护 unit 的公开初始化权重和任务分类头仍可训练。

query budget 固定为 CIFAR-100 训练集的 1%，即 500。八组配置使用相同的 query、hard label、数据增强、优化器和 `eval_ms`，以 `surrogate_acc` 最高的 epoch 作为各配置的 best 结果。

## 运行方式

```bash
python3 lab/02_head/run.py
```

默认重新运行全部八组。已有全保护结果时，只运行并合并随机保护四组可使用：

```bash
python3 lab/02_head/run.py --scope random
```

## 输出

```text
results/lab/02_head/metrics.json  八组配置的保护计划及 best/end 原始指标
results/lab/02_head/history.tsv   八组配置逐 epoch 的训练和评估记录
```
