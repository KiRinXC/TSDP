# PG05：Top-5 保护诊断

本实验比较 PG03 特征图归一化与 PG04 参数量归一化得到的两类 Top-5，并以同一
归一化内的 BN+Conv 联合保护作为主比较：

```text
feature_bn_top5    PG03 bn.tsv 的 product_rank 1–5
feature_main_top5  PG03 main.tsv 的 product_rank 1–5
feature_joint_top5 上述两组的 10 项并集
param_bn_top5      PG04 bn.tsv 的 product_rank 1–5
param_main_top5    PG04 main.tsv 的 product_rank 1–5
param_joint_top5   上述两组的 10 项并集
cross_feature_conv_param_bn  PG03 main Top-5 + PG04 BN Top-5
cross_feature_bn_param_conv  PG03 BN Top-5 + PG04 main Top-5
```

`bn` 只包含 20 个 BN gamma；`main` 只包含 16 个 BasicBlock 主路径 Conv weight。
八组均额外固定保护 `last_linear.weight` 与 `last_linear.bias`，分类头不参与残差排名。
拆分组保护 5 个候选 tensor 和 2 个分类头 tensor，联合组保护 10 个候选 tensor 和
2 个分类头 tensor；两个交叉组同样保护 10 个候选 tensor 和 2 个分类头 tensor。
保护状态保留 canonical public 初始化，暴露状态复制 victim，随后全部参数共同
finetune。

## 固定协议

```text
模型与数据             ResNet18 + CIFAR-100
随机种子               42，仅一个 seed
surrogate 初始化       formal_victim_then_public_v1
攻击者输出             victim soft posterior
query budget           500
query train/validation 400/100，seed 42，offset 100
训练                   最多 100 epoch，SGD + StepLR 正式统一参数
选模                   validation soft cross-entropy 最低的最早 epoch
评估                   checkpoint 固定后，每组只遍历一次完整 eval_ms
参考线                 正式 full_protection soft-posterior 黑盒
```

运行：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  playground/05_diagnose/run.py
```

输出写入 `results/playground/05_diagnose/`，包括八个保护 mask、`history.tsv`、
`data.tsv`、`metrics.json` 与三指标总图。该实验只提供 seed-42 诊断结果，不能写成
多随机种子稳定结论。
