# 测试 01：交叉残差、自然残差与有效秩

本测试在 ResNet18+CIFAR-100 上先比较 public model 与 victim model 的 40 个
算子参数组，并从中抽取 16 个 BasicBlock 主分支卷积；随后用一个独立的诊断性
前缀扫描检验 `product_score` 排序。数据侧排名只使用固定 500 张
`query_pool_ms` 图片，前缀扫描则按统一 MS 训练协议训练 surrogate。

## 输入

```text
public checkpoint
    weights/pre_train/resnet18-5c106cde.pth

victim checkpoint
    weights/MS/victim/resnet18/c100/best.pth

data
    dataset/MS/c100/manifest.json 指向的 query_pool_ms
    按 query_rank 取固定前 500 张
    使用 test transform，不进行数据增强
```

模型均设为 `eval()`，全程关闭梯度。随机种子固定为 42。

## 候选集合

40 个候选包括：

- 全部 20 个 `Conv2d.weight`；
- 全部 20 个 BatchNorm2d affine 参数组；每组同时包含 `.weight`（gamma）和
  `.bias`（beta）。

BN running state、分类头、ReLU、pooling 和残差加法不作为独立候选。
BN running mean/variance 只用于构造各自模型的标准化输入。

16 个主分支卷积固定为：

```text
layer1.0.conv1    layer1.0.conv2
layer1.1.conv1    layer1.1.conv2
layer2.0.conv1    layer2.0.conv2
layer2.1.conv1    layer2.1.conv2
layer3.0.conv1    layer3.0.conv2
layer3.1.conv1    layer3.1.conv2
layer4.0.conv1    layer4.0.conv2
layer4.1.conv1    layer4.1.conv2
```

16 项必须从同一次 40 项结果按 module 名称直接抽取，不进行第二次前向计算。

## 四路输出

对同一位置记 public/victim 输入为 `h_p/h_v`。Conv 的 public/victim weight
记为 `W_p/W_v`，四路输出为：

```text
z_pp = Conv(h_p, W_p)
z_pv = Conv(h_p, W_v)
z_vp = Conv(h_v, W_p)
z_vv = Conv(h_v, W_v)
```

BN affine 先分别使用各模型 running state 标准化输入，再以相同下标规则应用
public/victim `(gamma, beta)`：

```text
z_pp = gamma_p*h_hat_p + beta_p
z_pv = gamma_v*h_hat_p + beta_v
z_vp = gamma_p*h_hat_v + beta_p
z_vv = gamma_v*h_hat_v + beta_v
```

代码和结果字段沿用 `z_vv`；当用户侧/private 模型记为
`u` 时，`z_uu` 与这里的 `z_vv` 指同一个自然输出。

交叉残差使用紧凑形式：

```text
Conv: I = Conv(h_v-h_p, W_v-W_p)
BN:   I = (gamma_v-gamma_p)*(h_hat_v-h_hat_p)
```

beta 是与输入无关的平移项，因此在四项交叉差分中严格抵消；但它保留在四路输出和
自然残差 `z_vv-z_pp` 中。这就是把候选从 BN gamma 扩展到完整 BN affine 后，交叉项
公式不变、自然残差会重新计算的原因。

代码必须在首批同时核对该紧凑形式与四项展开
`z_vv-z_vp-z_pv+z_pp` 的数值一致性。

自然残差定义为：

```text
N = z_uu-z_pp = z_vv-z_pp
```

## 有效秩

对每张图片、每个 `C×H×W` 有符号张量，将通道作为列整理为：

```text
A: C×H×W -> (H×W)×C
```

不取绝对值、不中心化。若奇异值为 `sigma_i`，则：

```text
p_i = sigma_i^2 / sum_j(sigma_j^2)
r(A) = exp(-sum_i(p_i*log(p_i)))
```

严格零能量张量定义为 `r(A)=0`。否则有效秩不超过 `min(H×W,C)`。
所有秩和秩差都先按单张图片计算，再对固定 500 张 query 求平均。

## 九项指标

每个候选保存以下九项：

```text
cross_abs_mean
    mean_image(mean_chw(abs(I)))

natural_abs_mean
    mean_image(mean_chw(abs(z_uu-z_pp)))

cross_rank_mean
    mean_image(r(I))

rank_gap_vv_vp_mean
    mean_image(r(z_vv)-r(z_vp))

rank_gap_pv_pp_mean
    mean_image(r(z_pv)-r(z_pp))

rank_interaction_mean
    mean_image((r(z_vv)-r(z_vp))-(r(z_pv)-r(z_pp)))

natural_rank_mean
    mean_image(r(z_uu-z_pp))

rank_gap_uu_pp_mean
    mean_image(r(z_uu)-r(z_pp))

product_score
    cross_abs_mean * natural_abs_mean
```

`product_score` 是两个汇总标量相乘，不是逐元素乘积后再平均。

## 排序与显示

每张图只展示一个指标。候选统一按该指标的绝对值降序排列，数值并列时按 module
名称升序。柱子的方向和文字都显示真实值：负的 rank gap 必须保留负号，不能把
绝对值当作显示值。两个残差幅值、两个残差秩和乘积分数本身非负，但仍执行同一
排序规则。

图文件使用两个语义前缀：

```text
all_*     全部 40 个 Conv weight 与 BN affine 参数组
main_*    16 个 BasicBlock 主分支卷积
```

每个范围生成九张图：

```text
all_cross.png                 main_cross.png
all_natural.png               main_natural.png
all_cross_rank.png            main_cross_rank.png
all_rank_gap_vv_vp.png        main_rank_gap_vv_vp.png
all_rank_gap_pv_pp.png        main_rank_gap_pv_pp.png
all_rank_interaction.png      main_rank_interaction.png
all_natural_rank.png          main_natural_rank.png
all_rank_gap_uu_pp.png        main_rank_gap_uu_pp.png
all_product.png               main_product.png
```

## 正确性门槛

1. public/victim 必须捕获同名的 20 个 Conv 与 20 个 BN；
2. 40 项必须由 20 个 Conv weight 和 20 个 BN affine 参数组组成，每个 BN 组明确
   包含 weight 与 bias；
3. main 子集必须恰好等于固定的 16 个主分支卷积；
4. 每项必须恰好统计 500 张 query；
5. Conv/BN 的自然输出必须与 forward hook 一致；
6. 紧凑交叉项必须与四项展开在容差内一致；
7. 所有指标必须有限，有效秩不得超过 `min(H×W,C)`；
8. stem `conv1` 的紧凑交叉残差和交叉有效秩必须严格为零；
9. 每张图的候选顺序必须等于对应指标绝对值降序；
10. `product_score` 必须逐项等于两个残差幅值的乘积。

## Product 前缀 MS 扫描

`sweep.py` 支持两个互不混合的范围：`--scope all` 读取 `all.tsv` 的全部 40 个
候选，`--scope main` 读取 `main.tsv` 的 16 个 BasicBlock 主分支卷积。两者均按
`product_score` 绝对值降序排列，数值并列时按 `weight_state` 升序。入口不重新
计算排名，也不读取其他实验给出的 tensor 集合。

所有前缀均固定保护完整分类头：

```text
last_linear.weight
last_linear.bias
```

`Top-0` 表示只保护分类头；`Top-k` 在此基础上额外保护排名前 `k` 个完整候选组。
Conv 候选保护完整 weight；BN affine 候选同时保护完整 weight 与 bias，因此一个
BN 候选对应两个 state unit，但 Top-k 仍按一个候选组计数。

`--scope main --paired-bn-affine` 构造第三种受控变体。每加入一个主分支
`conv1/conv2.weight`，同时加入其紧随的同编号 `bn1/bn2` affine：

```text
conv1.weight  <->  bn1.weight + bn1.bias
conv2.weight  <->  bn2.weight + bn2.bias
```

例如 `layer2.1.conv1.weight` 配对 `layer2.1.bn1.weight` 与
`layer2.1.bn1.bias`。该绑定不保护 running mean、running variance、
`num_batches_tracked`，也不包含 downsample BN。它只检验 Conv weight 与紧随
BN affine 同时隐藏的 MS 效果，不代表包含运行状态的完整 BN 算子已经在真实 TEE
内执行。

每个点独立重放统一训练协议：seed 42 canonical 初始化、500 条 soft query 固定拆为
400 train/100 validation、最多训练 100 epoch、SGD `lr=0.01`、`momentum=0.5`、
`weight_decay=5e-4`、`StepLR(step_size=60, gamma=0.1)`，按 validation soft
cross-entropy 最低且最早的 epoch 选择 `best`，之后只在完整 `eval_ms` 上评估一次。

扫描从 `Top-0` 开始。若 `accuracy(Top-k) > accuracy(Top-(k-1))`，则把 `Top-k`
记为第一次严格反弹并立即停止，把 `Top-(k-1)` 记为本次扫描的最佳点；相等不算
反弹。若扫描完 40 项仍未反弹，则以 `Top-40` 为终点。

这条停止规则明确读取了每个前缀的 `eval_ms` accuracy，因此它是对排序的诊断性
oracle，不符合正式 MS 的 selector/eval 隔离，不能作为先验选点算法、正式方法结果
或超参数选择依据。每个单独前缀内部的 checkpoint 选择仍严格只使用 query
validation。

当前对应 BN affine 的受控比较只运行 seed 42 前缀扫描，不执行十种子配对验证，
也不保留多 seed 入口、日志或结果产物。

## 运行

完整运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  "$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  test/MS/01_cross/weights.py
```

运行 Product 前缀 MS 扫描：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  "$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  test/MS/01_cross/sweep.py
```

运行 16 项 `main_product` 前缀扫描：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  "$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  test/MS/01_cross/sweep.py --scope main
```

运行 `main_product + paired BN affine` 前缀扫描：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  "$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  test/MS/01_cross/sweep.py --scope main --paired-bn-affine
```

单批检查，不写结果：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  "$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  test/MS/01_cross/weights.py --dry-run
```

只核对 Product 排序、40 个前缀 mask、正式参考线和输入协议：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  "$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  test/MS/01_cross/sweep.py --dry-run
```

## 输出

数据侧排名产物：

```text
results/test/MS/01_cross/README.md
results/test/MS/01_cross/metrics.json
results/test/MS/01_cross/all.tsv
results/test/MS/01_cross/main.tsv
results/test/MS/01_cross/all_*.png       九张
results/test/MS/01_cross/main_*.png      九张
```

诊断性前缀扫描额外输出：

```text
results/test/MS/01_cross/sweep.json
results/test/MS/01_cross/sweep.tsv
results/test/MS/01_cross/sweep_history.tsv
results/test/MS/01_cross/sweep.png
results/test/MS/01_cross/main_sweep.json
results/test/MS/01_cross/main_sweep.tsv
results/test/MS/01_cross/main_sweep_history.tsv
results/test/MS/01_cross/main_sweep.png
results/test/MS/01_cross/main_affine_sweep.json
results/test/MS/01_cross/main_affine_sweep.tsv
results/test/MS/01_cross/main_affine_sweep_history.tsv
results/test/MS/01_cross/main_affine_sweep.png
```
