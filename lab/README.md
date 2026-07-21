# Lab 实验说明

本目录保存一些较小、可独立运行的验证实验。每个实验放在独立子目录中，尽量包含：

```text
README.md
run.py
```

实验输出默认写入 `results/lab/` 下对应目录，避免把运行产物混在代码目录里。

## 统一 MS 协议

训练普通 MS surrogate 的 Lab02、Lab04、Lab05、Lab06 和 Lab10 统一复用
`lab/protocol.py`，其数据划分、优化器、学习率调度、checkpoint 选择和正式边界读取
与 `exp/MS/train_surrogate/` 一致：

```text
query budget       先取 query_pool_ms 的固定 500 条前缀
query train        seed 42、offset 100 固定划分的 400 条
query validation   同一 budget 内互斥的 100 条
部分保护输出       victim soft posterior
训练               最多 100 epoch，batch size 64
优化器             SGD，lr=0.01，momentum=0.5，weight_decay=5e-4
调度               StepLR，step_size=60，gamma=0.1
选模               validation soft cross-entropy 最低的最早 epoch
最终评估           checkpoint 固定后只对完整 eval_ms 评估一次
参考线             no-protection 白盒、soft 黑盒与 hard-label 黑盒
```

这次调整解决了两个不可混用的问题：用全部 500 条 query 训练会缺少独立的
checkpoint 选择信号；使用 `eval_ms` 的逐 epoch 指标或固定训练终点会让主结果受测试集
选择或末轮波动影响。新协议把 query 内部选模与最终攻击评估分开，使保护策略之间、
Lab 与正式 `exp/` 之间使用相同随机初始化和相同观测能力时可以直接比较绝对指标。

旧协议生成的 Lab02/04/05/06 指标、图和历史不再是有效输入，必须由当前入口覆盖；
不得把历史数值与新结果拼接。`lab/03_baseline` 是只读正式结果汇总，不训练
surrogate。`lab/01_kmeans` 不属于 MS surrogate 训练，不受上述划分约束。

## 实验列表

```text
01_kmeans
  使用 ImageNet-1K 预训练 ResNet18 提取 CIFAR-100 特征，用 KMeans 检查 100 类样本的无监督可分性。

02_head
  在 ResNet18+CIFAR-100 的全保护与随机保护 MS 下，比较分类头替换/适配和可用权重冻结/微调。

03_baseline
  汇总四种正式 MS baseline、固定策略单点、TEESlice standalone 与 soft/hard 双黑盒参考线。

04_tensorshield
  在 ResNet18+CIFAR-100 上绘制作者 eligible rank 的 Top-1 至 Top-17 前缀曲线，对 Top-12 执行完整 leave-one-out 与联合删除消融，以十个独立 seed 对比 Top-10、BN gamma 闭包和删点候选，并比较排除分类头后的前 10、后 10 与分散 10 项。

05_state
  分别保护 ResNet18 的完整 state 类型和参数语义组，并在固定候选上执行四类 BN gamma 的十种子 drop 消融与 seed-42 单组加入实验。

06_weight
  在 TensorShield Top-10 至 Top-17 上补充 BN gamma、downsample Conv、二者组合与 Stem Conv，验证遗漏 weight 语义能否补齐 soft 黑盒差距。

10_pair
  在五个固定 BasicBlock 上比较 conv1+BN2 gamma 与对应 conv2+BN1 gamma 的局部配对保护。
```
