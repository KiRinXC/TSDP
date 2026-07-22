# PG01：四路原始 Weight 输出

本实验固定 `ResNet18+CIFAR-100`、seed 42 和 `query_pool_ms` 前 500 张图片，只保存
public/victim 在共享 backbone weight 上的四路原始输出，不执行归一化、有效秩或
surrogate 训练。

候选固定为 20 个 Conv weight 与 20 个 BN gamma，共 40 项。所有 bias 和最终分类层
均排除。BN 输入分别使用 public/victim 自身的 running mean/variance 标准化，四路
输出只乘 gamma，不加 beta。

```text
z_pp = operator(h_public, weight_public)
z_pv = operator(h_public, weight_victim)
z_vp = operator(h_victim, weight_public)
z_vv = operator(h_victim, weight_victim)
I    = z_vv - z_vp - z_pv + z_pp
N    = z_vv - z_pp
```

每个 weight 除四路 `z` 外，还保存由紧凑公式直接计算的原始交叉残差 `I`，避免四个
float32 输出相消时把舍入误差累计为残差信号。自然残差不重复保存，可由
`z_vv-z_pp` 直接派生。

主分数为未归一化的残差乘积：

```text
raw_cross_l1   = mean_image(sum_output(abs(I)))
raw_natural_l1 = mean_image(sum_output(abs(N)))
product_score  = raw_cross_l1 * raw_natural_l1
```

`main` 16 项必须从同一次 `all` 40 项结果按模块名直接抽取，不重新前向。

运行：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  playground/01_raw/run.py --dry-run
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  playground/01_raw/run.py
```

输出：

```text
results/playground/01_raw/manifest.json
results/playground/01_raw/data.tsv
results/playground/01_raw/main.tsv
results/playground/01_raw/activations/unit_<index>.pt
results/playground/01_raw/<all|main>_<cross|natural|product>.png
```
