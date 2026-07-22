# PG04：参数量归一化残差乘积

本实验读取 PG01 的未归一化残差总量，以当前 weight 的参数数目 `P=numel(weight)`
为分母。Conv 的 `P` 是卷积核参数数，BN gamma 的 `P` 是通道数：

```text
cross_residual   = raw_cross_l1 / P
natural_residual = raw_natural_l1 / P
product_score    = cross_residual * natural_residual
```

`product_score` 是主要保护效率代理分数：只有交叉残差与自然残差同时较大、同时参数量
较小时才会靠前。它仍是数据侧代理，不等同于实际 surrogate 保护效果；PG05 已使用
统一 seed-42 攻击协议诊断其 BN 与 main Conv Top-5。

`all` 使用全部 40 项，`main` 从同一结果抽取 16 个主分支 Conv，`bn` 从同一结果
抽取 20 个 BN gamma；三个 scope 分别重排 `product_rank`。

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  playground/04_param/run.py
```

输出 `metrics.json`、`data.tsv`、`main.tsv`、`bn.tsv`，以及 all/main/bn 各三张
残差图。
