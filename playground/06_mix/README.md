# PG06：特征量与参数量联合归一化

本实验直接读取 PG01 的未归一化残差总量，不重新执行模型前向。对每个候选 weight，
定义输出特征元素数 `F=C×H×W`、参数量 `P=numel(weight)`，并使用对称联合分母：

```text
joint_normalizer = sqrt(F × P)
cross_residual   = raw_cross_l1 / joint_normalizer
natural_residual = raw_natural_l1 / joint_normalizer
product_score    = cross_residual × natural_residual
                 = raw_cross_l1 × raw_natural_l1 / (F × P)
```

该分数等价于 PG03 特征归一化乘积分数与 PG04 参数归一化乘积分数的几何平均。两个
残差使用同一分母，不预设交叉残差或自然残差应分别对应某一种归一化。

`all` 使用全部 40 个 Conv weight/BN gamma，`main` 使用 16 个 BasicBlock 主路径
Conv weight，`bn` 使用 20 个 BN gamma。三个 scope 分别重排 `product_rank`；继续
排除全部 bias、分类头、BN beta 和 BN buffer。

运行：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  playground/06_mix/run.py
```

输出写入 `results/playground/06_mix/`：`metrics.json`、`all.tsv`、`main.tsv`、
`bn.tsv`，以及 all/main/bn 各三张 cross、natural 和 product 图。本实验只派生排序，
不训练 surrogate，也不扩展随机种子。
