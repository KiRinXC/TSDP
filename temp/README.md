# 交叉残差与因果残差临时验证

本目录只保留两条仍在验证的 filter 级保护思路：

1. `residual.py`：计算公开预训练模型与受害者模型的四路交叉前向残差；
2. `causal.py`：把局部交叉残差通过受害者模型的下游计算图投影到最终 posterior；
3. `attack.py`：在相同卷积参数预算下分别保护两种分数选出的 filter，并按统一
   MS 协议直接比较保护效果。

早期 ARC、REC 门优化、入口层、尾部层、Shapley、独立 XAI tensor/block 等探索已经
失效并删除。需要追溯时使用 Git 历史，不在 `temp/` 中保留兼容脚本或旧产物。

## 残差定义

对同一张图像和同一卷积位置，记公开权重/公开输入、受害者权重/受害者输入产生的
pre-BN 输出为 `PP`、`PV`、`VP`、`VV`。逐 filter 的局部 weight 残差为：

```text
R_weight = 0.5 * ((PV - PP) + (VV - VP))
```

`residual.py` 使用 `mean(abs(R_weight))` 作为交叉残差分数。`causal.py` 则把
`R_weight` 从当前卷积输出移除再逐步注回，并以受害者 posterior KL 为目标计算
residual conductance；绝对 conductance 的样本均值作为因果残差分数。

交叉残差的测量边界不包含 BN。因果投影的后续计算包含 BN、残差连接和分类头。

## 数据隔离

固定 500 条 `query_pool_ms` 先按 seed 42 拆成 400 条 query-train 与 100 条
query-validation：

- 两种 filter 分数只读取 400 条 query-train 图像，不读取标签或保存的 posterior；
- 100 条 query-validation 只用于 surrogate checkpoint 选择，不参与 filter 排名；
- 10,000 条 `eval_ms` 不参与选择，只在每个 case 的 checkpoint 固定后评估一次。

## 保护与 MS 协议

两种方法使用相同的 `239,616` 个卷积权重参数预算。filter 按分数降序遍历，只有
在剩余预算可容纳整个 filter 时才纳入。除此之外固定完整保护：

- 分类头 weight 与 bias，共 51,300 个参数；
- 全部 20 个 BN gamma，共 4,800 个参数。

surrogate 使用 `formal_victim_then_public_v1` 初始化轨迹。未保护状态由 victim
直接暴露，保护状态保持公开初始化；随后与正式部分保护攻击一致，对整个 surrogate
进行联合微调。MS 固定为：

```text
label                  soft posterior
query train/validation 400 / 100
最大 epoch             100
batch size             64
优化器                 SGD，lr=0.01，momentum=0.5，weight_decay=5e-4
调度器                 StepLR，step_size=60，gamma=0.1
checkpoint             validation soft cross-entropy 最低的最早 epoch
最终评估               checkpoint 固定后对 eval_ms 评估一次
```

图中同时给出正式 white-box、soft 黑盒、hard 黑盒和 TensorShield 参考；其中部分
保护策略与 soft 黑盒共享 soft-posterior 攻击训练，hard 黑盒仅作为 label-only
下界参考。

## 运行

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" temp/residual.py --overwrite
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" temp/causal.py --overwrite
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" temp/attack.py --overwrite
```

三个入口的 `--dry-run` 都不会训练或读取 `eval_ms`。输出统一写入
`temp/output/`；该目录不进入正式 `results/MS/`、`weights/MS/` 或主汇总表。

## 当前结果

400 张 query-train 图像重算后，两种方法的卷积预算几乎完全用满；加入完整分类头与
全部 BN gamma 后，总保护比例均为 `2.6336%`：

```text
方法             filter 数  卷积参数  best  accuracy  fidelity  posterior KL
交叉残差              300     239587    90   0.1846    0.2016       2.569087
因果残差              208     239594    90   0.1627    0.1803       2.720244
TensorShield            -    1009704    93   0.1728    0.1865       2.694492
soft 黑盒               -          -    45   0.1390    0.1463       3.039817
```

因果残差在相同 filter 预算下三项均优于直接交叉残差，并且以更低保护比例三项均优于
TensorShield，说明“沿下游计算图衡量局部残差对最终 posterior 的贡献”确实增加了
有用信息。不过它仍未达到 soft 黑盒，当前结论只是因果分数值得继续扩展为跨层连通
通道块路径，不是方法已经完成。
