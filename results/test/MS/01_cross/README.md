# 测试 01 结果：交叉残差、自然残差与有效秩

本目录保存 ResNet18+CIFAR-100 的两部分 Test01 结果。数据侧分析使用固定 500 张
query，比较全部 40 个 Conv weight/BN affine 参数组，并抽取 16 个 BasicBlock 主分支
卷积；诊断性前缀扫描则使用统一 MS 训练协议检验 `product_score` 排序。

## 结果文件

```text
metrics.json
    数据、模型哈希、公式、正确性检查、40/16 项指标和输出索引

all.tsv
    全部 40 个候选的九项指标

main.tsv
    16 个 BasicBlock 主分支卷积的九项指标

all_cross.png                 main_cross.png
all_natural.png               main_natural.png
all_cross_rank.png            main_cross_rank.png
all_rank_gap_vv_vp.png        main_rank_gap_vv_vp.png
all_rank_gap_pv_pp.png        main_rank_gap_pv_pp.png
all_rank_interaction.png      main_rank_interaction.png
all_natural_rank.png          main_natural_rank.png
all_rank_gap_uu_pp.png        main_rank_gap_uu_pp.png
all_product.png               main_product.png
    40 项和 16 项各九张单指标图

sweep.json
    Product 前缀、停止规则、最佳点、正式参考线、保护集合与三项 MS 原始指标

sweep.tsv
    Top-0 至第一次反弹点的保护比例、validation-best epoch 与最终三指标

sweep_history.tsv
    五个已运行前缀各 100 epoch 的 query train/validation 日志，共 500 行

sweep.png
    Product 前缀 accuracy 折线与 hard/soft 黑盒、TensorShield Top-10 参考线

main_sweep.json
    16 个主分支卷积的 Product 前缀、停止点、保护集合与三项 MS 原始指标

main_sweep.tsv
    Main Top-0 至第一次反弹点的保护比例、validation-best epoch 与最终三指标

main_sweep_history.tsv
    八个已运行 Main 前缀各 100 epoch 的训练/验证日志，共 800 行

main_sweep.png
    Main Product 前缀 accuracy 折线与三条正式参考线

main_affine_sweep.json / main_affine_sweep.tsv
    Main Conv weight 与对应 BN affine 绑定后的前缀、停止点和三项 MS 指标

main_affine_sweep_history.tsv
    八个 Conv+affine 前缀各 100 epoch 的训练/验证日志，共 800 行

main_affine_sweep.png
    Conv+对应 BN affine 前缀 accuracy 折线与三条正式参考线

```

所有图按对应指标的绝对值降序排列，但柱子和文字显示真实值。负的 rank gap 在图中
保留负号并向左延伸。

## 首位结果

| 指标 | 40 项绝对值首位 | 真实值 | 16 项绝对值首位 | 真实值 |
|---|---|---:|---|---:|
| 交叉残差 | `layer4.1.bn2` | 1.464529 | `layer2.0.conv1` | 0.905005 |
| 自然残差 | `conv1` | 1.311637 | `layer1.1.conv1` | 1.244755 |
| 交叉残差有效秩 | `layer1.1.conv2` | 15.273210 | `layer1.1.conv2` | 15.273210 |
| `r(z_vv)-r(z_vp)` | `bn1` | +5.954765 | `layer2.1.conv1` | +2.304092 |
| `r(z_pv)-r(z_pp)` | `layer1.1.conv2` | -5.936694 | `layer1.1.conv2` | -5.936694 |
| rank interaction | `layer1.1.conv2` | +4.793478 | `layer1.1.conv2` | +4.793478 |
| 自然残差有效秩 | `layer1.1.bn2` | 16.986243 | `layer1.1.conv2` | 12.628225 |
| `r(z_uu)-r(z_pp)` | `layer1.1.bn2` | -5.803418 | `layer1.1.conv2` | -4.488594 |
| 残差乘积 | `layer2.0.conv1` | 1.061349 | `layer2.0.conv1` | 1.061349 |

## 必要解读

交叉残差幅值和交叉残差秩给出不同排序。`layer4.1.bn2` 的交叉幅值最大，但其
输出空间为 `1×1`，有效秩容量只有 1；`layer1.1.conv2` 的交叉幅值在 40 项中
并不靠前，却拥有最大的交叉有效秩和 rank interaction。因此，幅值衡量交叉项有
多强，有效秩衡量该交叉项的能量分布在多少个通道方向上，二者不能互相替代。

stem `conv1` 的自然残差以 1.311637 排第一，但其紧凑交叉残差、交叉有效秩和
残差乘积均严格为零。原因是 public/victim 在 stem 前接收相同图片，输入差为零，
但不同 stem 权重仍会产生直接自然输出差。

在 16 个主分支卷积中，残差乘积前七项全部是 `conv1`；但秩差指标同时包含正值和
负值，不能只按有符号降序解读。例如 `layer1.1.conv2` 的
`r(z_pv)-r(z_pp)=-5.936694`，其绝对变化最大，负号表示 victim weight 在 public
输入上的输出有效秩低于完全 public 输出，而不是“变化较小”。

BN affine 的 beta 在四路交叉差分中严格抵消，因此交叉残差及其有效秩与原
gamma-only 公式一致；beta 仍进入四路输出和自然残差，所以自然残差、自然秩、秩差
及乘积分数已经重新计算。

这些结果首先是数据侧位置描述。下方 Product 前缀扫描只提供读取 `eval_ms` 的
post-hoc 诊断，仍不能仅凭当前排序声称对应位置就是可先验选择的攻击依赖位置。

## Product 前缀扫描结果

所有点固定保护 `last_linear.weight` 和 `last_linear.bias`，再按 `all_product.png`
对应顺序逐项增加完整候选组。五个点均独立重放 seed 42 canonical 初始化，使用同一
500 条 soft query 的 400/100 train/validation 划分，训练 100 epoch 并按
validation soft cross-entropy 选择最早 `best`。

| 点 | 本次新增 tensor | 保护比例 | Accuracy | Fidelity | Posterior KL |
|---|---|---:|---:|---:|---:|
| Top-0 | 仅完整分类头 | 0.4569% | 0.3985 | 0.4621 | 1.616573 |
| Top-1 | `layer2.0.conv1.weight` | 1.1136% | 0.3373 | 0.3803 | 1.900038 |
| Top-2 | `layer1.1.conv1.weight` | 1.4419% | 0.3094 | 0.3497 | 2.016931 |
| **Top-3** | `layer1.0.conv1.weight` | **1.7702%** | **0.2882** | **0.3239** | **2.096797** |
| Top-4 | `layer4.1.bn2.weight/bias` | 1.7793% | 0.2964 | 0.3321 | 1.800742 |

Top-4 accuracy 从 Top-3 的 0.2882 回升至 0.2964，故按预先固定的第一次严格反弹
规则停止，并选择紧邻反弹前的 **Top-3**。最佳点保护 198,756/11,227,812 个参数，
即 **1.770211%**；其中分类头占 51,300 个参数，三个卷积 weight 占 147,456 个
参数。Top-4 的 BN affine 候选同时保护 `layer4.1.bn2.weight` 与 `.bias`，因此增加
两个 state unit 和 1,024 个参数。

正式参考线为 hard-label 黑盒 0.0890、soft-posterior 黑盒 0.1390、TensorShield
Top-10 0.1728。Top-3 的 accuracy 0.2882 和 fidelity 0.3239 均明显高于三条保护
参考，Posterior KL 2.096797 也低于 TensorShield 的 2.694492 及两条黑盒。因此，
当前 Product 前缀虽从 Top-0 到 Top-3 连续削弱攻击，但在首次反弹前仍未达到黑盒
或 TensorShield Top-10 的保护效果。

该扫描按 `eval_ms` accuracy 判断停止，属于 post-hoc 排名诊断，不是满足正式
selector/eval 隔离的先验选点方法，不能把 Top-3 当作正式方法结果。

## Main Product 前缀扫描结果

该扫描排除 BN affine、stem 和 downsample，只读取 `main.tsv` 中 16 个
BasicBlock 主分支卷积的 `product_score` 排序。固定分类头、训练协议、随机种子、
参考线与停止规则均和 40 项扫描完全一致。

| 点 | 本次新增 tensor | 保护比例 | Accuracy | Fidelity | Posterior KL |
|---|---|---:|---:|---:|---:|
| Top-0 | 仅完整分类头 | 0.4569% | 0.3985 | 0.4621 | 1.616573 |
| Top-1 | `layer2.0.conv1.weight` | 1.1136% | 0.3373 | 0.3803 | 1.900038 |
| Top-2 | `layer1.1.conv1.weight` | 1.4419% | 0.3094 | 0.3497 | 2.016931 |
| Top-3 | `layer1.0.conv1.weight` | 1.7702% | 0.2882 | 0.3239 | 2.096797 |
| Top-4 | `layer3.0.conv1.weight` | 4.3968% | 0.1970 | 0.2164 | 2.549389 |
| Top-5 | `layer2.1.conv1.weight` | 5.7101% | 0.1798 | 0.1947 | 2.642482 |
| **Top-6** | `layer4.0.conv1.weight` | **16.2166%** | **0.1519** | **0.1626** | **2.794303** |
| Top-7 | `layer4.1.conv1.weight` | 37.2296% | 0.1565 | 0.1665 | 2.789883 |

Top-7 accuracy 从 Top-6 的 0.1519 回升至 0.1565，因此按规则选择 **Top-6**。
该点固定保护完整分类头，并保护以下六个卷积：

```text
layer2.0.conv1.weight
layer1.1.conv1.weight
layer1.0.conv1.weight
layer3.0.conv1.weight
layer2.1.conv1.weight
layer4.0.conv1.weight
```

最佳点保护 1,820,772/11,227,812 个参数，即 **16.216624%**。它的三项结果均优于
TensorShield Top-10 的 0.1728/0.1865/2.694492，并接近但尚未达到 soft 黑盒的
0.1390/0.1463/3.039817；距离 hard 黑盒 0.0890/0.0969/3.387234 更远。同时其
保护比例高于 TensorShield 的 8.9934%，因此当前 Main Product 前缀是保护效果更强、
但参数成本更高的诊断结果，尚未形成更优的安全-成本方案。

Main 排名前七项全部是 `conv1.weight`，且从 Top-0 到 Top-6 accuracy 单调下降，
说明排除 BN 后，`product_score` 对这组六个 conv1 给出了连续有效的前缀；但
`layer4.1.conv1.weight` 参数量很大，加入后成本从 16.2166% 跳到 37.2296%，保护
效果反而略微回退。这只是当前 seed 和 `eval_ms` 上的后验诊断，仍不能单独证明
“conv1 是先验攻击依赖规则”。

## Main Product 与对应 BN Affine 绑定

该变体在每个 Main Product 卷积后绑定对应 BN affine：`conv1.weight` 配对
`bn1.weight/bias`，`conv2.weight` 配对 `bn2.weight/bias`。它不保护 running
mean/variance、`num_batches_tracked` 或任何 downsample BN。分类头、排序、训练
协议和反弹规则与 Main Product 扫描相同。

| 点 | 新增的配对 BN affine | 保护比例 | Accuracy | Fidelity | Posterior KL |
|---|---|---:|---:|---:|---:|
| Top-0 | 无 | 0.4569% | 0.3985 | 0.4621 | 1.616573 |
| Top-1 | `layer2.0.bn1.weight/bias` | 1.1158% | 0.3327 | 0.3794 | 1.898859 |
| Top-2 | + `layer1.1.bn1.weight/bias` | 1.4453% | 0.3179 | 0.3574 | 1.989537 |
| Top-3 | + `layer1.0.bn1.weight/bias` | 1.7748% | 0.2977 | 0.3352 | 2.040224 |
| Top-4 | + `layer3.0.bn1.weight/bias` | 4.4060% | 0.2038 | 0.2209 | 2.515924 |
| Top-5 | + `layer2.1.bn1.weight/bias` | 5.7215% | 0.1823 | 0.1996 | 2.619094 |
| **Top-6** | + `layer4.0.bn1.weight/bias` | **16.2371%** | **0.1579** | **0.1664** | **2.781295** |
| Top-7 | + `layer4.1.bn1.weight/bias` | 37.2592% | 0.1595 | 0.1678 | 2.777217 |

Top-7 accuracy 从 0.1579 回升到 0.1595，因此选择 Top-6。该点保护六个卷积、
对应六组 BN affine 和完整分类头，共 1,823,076 个参数，比例为
**16.237144%**。

与不绑定 affine 的 Main Top-6 相比，affine 增加 2,304 个参数，即 0.020520 个
百分点；accuracy 从 0.1519 升到 0.1579，fidelity 从 0.1626 升到 0.1664，
Posterior KL 从 2.794303 降到 2.781295，seed 42 上三项都回退。因此当前单 seed
结果不支持“完整局部 BN affine 与卷积绑定后必然增强保护”。它也不能替代 Lab04
覆盖全部 20 个 BN gamma 的跨层尺度闭包。

该结果仍按 `eval_ms` accuracy 后验停止，只是单 seed 的诊断性比较，不是正式
选择器。当前不执行对应 BN affine 的十种子配对验证，也不保留多 seed 结果产物。
