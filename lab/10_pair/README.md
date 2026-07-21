# 实验 10：残差块卷积与局部 BN Gamma 配对保护

本实验固定 Lab04 后验候选覆盖的五个 BasicBlock，只比较两种局部配对保护策略，
不加入全部 20 个 BN gamma，也不加入 downsample、BN bias 或运行统计量。

五个固定 BasicBlock 为：

```text
layer1.0
layer1.1
layer2.0
layer2.1
layer3.0
```

两种策略定义为：

```text
conv1_bn2
  五个 <block>.conv1.weight
  五个 <block>.bn2.weight
  last_linear.weight
  last_linear.bias

conv2_bn1
  五个 <block>.conv2.weight
  五个 <block>.bn1.weight
  last_linear.weight
  last_linear.bias
```

`bn1.weight` 与 `bn2.weight` 均为对应 `BatchNorm2d` 的 gamma。两组都恰好保护
12 个完整 state tensor，但对应 `conv2.weight` 在阶段转换块中更大，因此本实验
不是相同参数预算比较；结果必须同时报告实际保护参数量与比例。

## 固定协议

```text
数据划分          dataset/MS/c100/manifest.json 中的 query_pool_ms 与 eval_ms
victim            weights/MS/victim/resnet18/c100/best.pth
surrogate 初始化  formal_victim_then_public_v1
攻击者可观测输出  victim soft posterior
query budget      500
query 划分        seed 42、offset 100，400 train / 100 validation
保护语义          选中 tensor 保持 public/随机初态，其余 victim state 直接暴露
分类头            weight 与 bias 均保护，head mode 为 replace
训练方式          全部 surrogate 参数共同微调
训练轮数          100
优化器            SGD，lr=0.01，momentum=0.5，weight_decay=5e-4
学习率调度        StepLR，step_size=60，gamma=0.1
选模              validation soft cross-entropy 最低的最早 epoch
主要评估点        checkpoint 固定后只评估一次完整 eval_ms
原始指标          surrogate accuracy、fidelity、posterior KL
随机种子          42
```

图中只绘制这两种保护策略的柱子，同时加入正式 soft 与 hard-label 黑盒参考线；
no-protection 白盒仍保存在 `metrics.json` 和结果表中，但不纳入图的纵轴，以免白盒
与局部保护结果跨度过大而压缩两根柱子的可读差异。参考边界不作为新增训练策略。

## 运行

先核对两个集合、完整分类头、12 个 unit 与实际参数成本：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/10_pair/run.py --dry-run
```

执行两组完整训练：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  lab/10_pair/run.py
```

## 输出

```text
results/lab/10_pair/metrics.json
results/lab/10_pair/data.tsv
results/lab/10_pair/history.tsv
results/lab/10_pair/metrics.png
results/lab/10_pair/conv1_bn2_mask.pt
results/lab/10_pair/conv2_bn1_mask.pt
```
