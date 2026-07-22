# PG03：特征图归一化残差乘积

本实验读取 PG01 的未归一化残差总量，以输出特征元素数
`F=C×H×W` 为分母：

```text
cross_residual   = raw_cross_l1 / F
natural_residual = raw_natural_l1 / F
product_score    = cross_residual * natural_residual
```

`product_score` 是主排序字段。`all` 使用全部 40 个 Conv weight/BN gamma，`main`
从同一结果抽取 16 个主分支 Conv，`bn` 从同一结果抽取 20 个 BN gamma；三个 scope
分别重排 `product_rank`。

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  playground/03_feature/run.py
```

输出 `metrics.json`、`data.tsv`、`main.tsv`、`bn.tsv`，以及 all/main/bn 各三张
残差图。
