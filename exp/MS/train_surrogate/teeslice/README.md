# TEESlice 代理模型训练

本目录使用统一 MS 查询协议攻击 TEESlice defended victim。攻击者知道 TEESlice 方法和公开 ImageNet backbone，但看不到 private proxy、alpha、分类头和任务适配后的私有 BN buffer。

TEESlice 改变了 victim 的模型结构与参数边界，因此本入口独立于普通 `train_surrogate/defense/` 的 122-unit mask 插件；训练、评估和结果字段仍复用 `train_surrogate/core/`。

## 前置产物

```text
weights/MS/victim/teeslice_r18/c100/best.pth    defended victim
dataset/MS/c100/teeslice_r18/manifest.json      查询协议
dataset/MS/c100/teeslice_r18/posteriors.pt      500 条 soft posterior
weights/pre_train/resnet18-5c106cde.pth         公开 ImageNet 权重
```

查询产物由以下命令生成：

```bash
python3 exp/MS/transfer/get_label.py teeslice_r18 c100
```

## 固定协议

```text
victim              TEESlice defended victim best.pth
surrogate 初始化    公开 ImageNet CIFAR ResNet18 和随机 C100 分类头
query 来源          query_pool_ms 的固定预算前缀
query budget        500，即 victim_train 的 1%
攻击者可观测输出    soft posterior
query transform     确定性的 test transform
训练方式            所有 surrogate 参数共同 finetune
训练轮数            100
优化器              SGD，lr=0.01，momentum=0.5，weight_decay=5e-4
学习率调度          StepLR，step_size=60，gamma=0.1
主要评估点          第 100 轮 end.pth
原始指标            accuracy、fidelity、posterior KL 及其计数
随机种子            42
```

fidelity 和 posterior KL 始终相对于 TEESlice defended victim 计算，不能复用普通 ResNet18 victim 的 eval posterior。TEESlice 的成本使用 private proxy、分类头、BN buffer 和 FLOPs 描述，`protected_unit_count` 留空。

## 运行方式

```bash
python3 exp/MS/train_surrogate/teeslice/attack.py resnet18 c100 \
  --budget 500 --training-mode finetune --label-mode soft
```

只验证输入、初始化和结果协议：

```bash
python3 exp/MS/train_surrogate/teeslice/attack.py resnet18 c100 \
  --budget 500 --training-mode finetune --label-mode soft --dry-run
```

覆盖当前同语义结果时显式增加 `--overwrite`。

## 输出

```text
weights/MS/surrogate/resnet18/c100/teeslice/
├── best.pth                  surrogate accuracy 最高点，仅作诊断
├── end.pth                   第 100 轮固定终点，作为主 checkpoint
├── params.json
└── train.log.tsv

results/MS/resnet18/c100/teeslice/metrics.json
results/MS/resnet18/c100/metrics.tsv
```
