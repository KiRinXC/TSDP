# 实验 05：State 类型保护对比

本实验在 `ResNet18+CIFAR-100` 上分别只保护一种 `state_dict` 条目类型，观察不同类型状态不可见时的 MS 效果。五种类型是相互独立的保护方案，不构成累计保护序列。

## 固定协议

```text
数据划分          dataset/MS/c100/manifest.json 中的 query_pool_ms 与 eval_ms
victim            weights/MS/victim/resnet18/c100/best.pth
surrogate 初始化  ImageNet-1K 官方预训练 ResNet18
攻击者可观测输出  victim soft posterior
query transform   确定性的 test transform
query budget      500，即 CIFAR-100 训练集的 1%
保护类型          weight、bias、running_mean、running_var、num_batches_tracked
保护语义          只保留所选类型的公开初始化，其余 victim 状态全部复制
分类头            严格按 mask 混合复制，不额外隐藏未保护类型
训练方式          全部 surrogate 参数共同微调；BN buffer 按正常 train 语义更新
训练轮数          100
优化器            SGD，lr=0.01，momentum=0.5，weight_decay=5e-4
学习率调度        StepLR，step_size=60，gamma=0.1
主要评估点        第 100 轮 end；best 只作为训练诊断
原始指标          surrogate accuracy、fidelity、posterior KL
随机种子          42
```

类型匹配使用 `state_dict` 名称最后一个字段的精确比较。例如 `weight` 同时包含卷积 weight、BN weight 和分类头 weight；不会把 `running_mean` 或 bias 纳入同一方案。

当只保护 `weight` 时，分类头 weight 使用公开模型替换 C100 分类头后的随机初始化，分类头 bias 从 victim 复制；只保护 `bias` 时则相反。保护三种 BN buffer 时，分类头 weight 和 bias 都从 victim 复制。

## 成本口径

`running_mean`、`running_var` 和 `num_batches_tracked` 不是可训练参数，因此不能使用现有 `protected_param_ratio` 作为唯一横坐标。本实验保存四种互补比例：

```text
protected_unit_ratio           受保护 state tensor 数量 / 122
protected_param_ratio          受保护可训练标量 / 全部可训练标量
protected_state_element_ratio  受保护 state 标量数 / 全部 state 标量数
protected_state_byte_ratio     受保护 tensor payload 字节数 / 全部 state payload 字节数
```

三张图统一使用 `protected_state_byte_ratio` 作为横坐标，并采用对数刻度。各类型只绘制独立散点，不把不同类型连接成曲线；无保护和全保护使用当前正式协议的 `end` 指标作为水平参考线。

## 运行方式

```bash
python3 lab/05_state/run.py
```

## 输出

```text
results/lab/05_state/metrics.json               五种类型的协议、保护统计和原始指标
results/lab/05_state/history.tsv                五组各 100 轮训练与评估记录
results/lab/05_state/data.tsv                   绘图使用的 end 原始点
results/lab/05_state/accuracy.png               保护存储比例与 accuracy
results/lab/05_state/fidelity.png               保护存储比例与 fidelity
results/lab/05_state/posterior_kl.png            保护存储比例与 posterior KL
results/lab/05_state/<type>_mask.pt              各类型的 122-unit 紧凑保护掩码
```
