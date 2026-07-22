# Playground 结果

本目录保存 PG01–PG05 的探索结果，不进入正式 `results/MS/` 或 `results/lab/` 索引。

```text
01_raw      40 个 weight 的四路原始输出、精确交叉残差和未归一化乘积
02_rank     all/main/bn 有效秩、秩差与秩乘积
03_feature  all/main/bn 特征图归一化残差与乘积
04_param    all/main/bn 参数量归一化残差与乘积
05_diagnose BN/Conv 同源联合与跨归一化交叉 Top-5 保护效果
```

## 结论索引

- PG01 是唯一模型前向来源；后续三个实验都读取相同四路张量，避免重复前向造成数据
  漂移。它不包含 bias 或最终分类层。
- PG02 研究残差在通道方向上的谱分布。有效秩对非零整体缩放不变，因此不需要特征图
  或参数量归一化；all、main、bn 的秩乘积排名均在各自候选集内重新编号。
- PG03 的乘积分数衡量每个输出特征位置平均承载的交叉残差与自然残差。
- PG04 的乘积分数以 weight 参数量为成本，强调小参数候选；它只是保护效率代理，
  需要结合 PG05 的实际攻击结果解释。
- PG02–PG04 都对 all 40 项、main 16 项和 BN gamma 20 项分别重排；PG02 的
  `rank_product_rank` 与 PG03/PG04 的 `product_rank` 均不沿用 all 排名。
- PG05 的主比较是每种归一化内部联合保护 BN Top-5 与 Conv Top-5。两个联合组均比
  对应 Conv 拆分组更强。交叉实验进一步发现 Feature Conv + Parameter BN 最强，为
  `0.1489/0.1601/2.823222`；在分别固定两套 Conv 的对照中，Parameter BN 都优于
  Feature BN。八组都未达到 soft 黑盒，且只运行 seed 42。
