# PG02：四路输出有效秩

本实验只读取 PG01 保存的四路 float32 输出，不重新运行 public/victim 模型。对每张
图片的 `C×H×W` 有符号张量整理为 `(H×W)×C`，以奇异值平方归一化后的谱熵计算
有效秩；严格零张量的有效秩定义为 0。

有效秩对非零整体缩放不变，因此本实验不按特征图或参数量归一化。主组合分数为：

```text
rank_product = mean_image(r(I)) * mean_image(r(N))
```

同时保留四路输出秩差与 rank interaction。`all` 使用 PG01 的 40 项，`main` 从同一
结果抽取 16 个 BasicBlock 主分支 Conv，`bn` 抽取 20 个 BN gamma。三套候选分别按
`rank_product` 降序排序，同分时按 `state_name` 升序，并在各自范围内重新编号。

运行：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  playground/02_rank/run.py
```

输出 `metrics.json`、`data.tsv`、`main.tsv`、`bn.tsv`，以及 all/main/bn 各七张
秩指标图。
