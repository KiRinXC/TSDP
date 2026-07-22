# PG07：固定结构依赖下的 Feature Conv Top-k

本实验固定保护 ResNet18 的替换分类头、Stem BN1 gamma 和三个残差捷径下采样
Conv，再按 PG03 Feature main Conv 的 `product_rank` 从高到低逐项加入保护集合，
形成最多 Top-0 到 Top-16 的嵌套 case。Top-0 用于测量固定结构集合自身的效果；
Top-5 与 Lab07 已运行的 Feature Conv Top-5 扩展 mask 相同，用于独立复现检查。

固定保护状态为：

```text
bn1.weight
layer2.0.downsample.0.weight
layer3.0.downsample.0.weight
layer4.0.downsample.0.weight
last_linear.weight
last_linear.bias
```

变量候选只读取 `results/playground/03_feature/main.tsv` 的 16 个 BasicBlock 主路径
Conv weight。分类头和固定结构状态不参与残差排名；每一级 Top-k 都包含前一级的完整
集合，并新增 `product_rank=k` 对应的一个 Conv weight。

各级按 k 递增顺序运行。若当前级相对前一级出现 MS accuracy 上升、Fidelity 上升或
Posterior KL 下降中的任意一种，则定义为反弹：保留当前反弹点作为证据，并停止后续
Top-k。若始终没有反弹，则运行至 Top-16。

## 固定协议

```text
模型与数据             ResNet18 + CIFAR-100
随机种子               42，仅一个 seed
surrogate 初始化       formal_victim_then_public_v1，每个 k 独立重放
攻击者可观测输出       victim soft posterior
query budget           500
query train/validation 400/100，seed 42，offset 100
训练                   每组最多 100 epoch，统一 SGD + StepLR 参数
选模                   validation soft cross-entropy 最低的最早 epoch
最终评估               checkpoint 固定后每组只评估一次完整 eval_ms
参考线                 正式 soft-posterior 与 hard-label 黑盒
```

运行：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  playground/07_topk/run.py --dry-run
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  playground/07_topk/run.py
```

输出写入 `results/playground/07_topk/`：实际执行 case 的 mask、`history.tsv`、`data.tsv`、
`metrics.json`、按 k 绘制的 `metrics_by_k.png` 和按保护参数比例绘制的
`metrics_by_cost.png`。本实验只提供 seed-42 诊断，不扩展为多随机种子结论。
