# 测试 02：任务特定表征传输

## 目的

本测试验证一个不使用 MS 反馈的结构猜想：public model 到 victim model 的
微调会让少数算子承担任务特定的通道坐标、尺度和相关结构迁移。如果一个
算子将相同输入映射到明显不同的 public/victim 表征几何，它就是后续
victim 权重复用时的候选坐标接口。

当前只计算与 Test01 完全相同的 40 个 backbone 候选：

```text
20 个 Conv2d weight
20 个 BatchNorm2d affine 参数组，每组同时包含 bn.weight（gamma）与 bn.bias（beta）
```

分类头 `last_linear.weight/bias` 是固定私有任务边界：它在后续保护预算中必须
计入成本，但 ImageNet-1K public head 与 CIFAR-100 victim head 的维度和类别
语义均不对应，因此不参与本次表征传输排名。

## 固定输入

```text
模型                  ResNet18
数据集                CIFAR-100
图片                  query_pool_ms 按 query_rank 的固定前 500 张
public 权重           weights/pre_train/resnet18-5c106cde.pth
victim 权重           weights/MS/victim/resnet18/c100/best.pth
模型模式              eval
输入变换              确定性 test transform
随机种子              42
batch size            64
```

本测试与当前 Test01 一致，只使用固定 500 张 query，不再维护 50,000 张训练图片
的重复计算入口或结果。

## 相同输入的算子干预

对算子 `l` 先从 victim 自然前向中取得输入 `h_v`，然后把完全相同的输入
分别交给 public/victim 版本的该算子。这样上游表征已经固定，输出差异只
由当前参数产生。

Conv weight：

```text
z_p = Conv(h_v, W_p)
z_v = Conv(h_v, W_v)
```

BN affine：

```text
h_hat_v = (h_v - mean_v) / sqrt(var_v + eps)
z_p = gamma_p * h_hat_v + beta_p
z_v = gamma_v * h_hat_v + beta_v
```

BN affine 评分同时替换 gamma 与 beta。`running_mean/var` 固定使用 victim 状态，
因此本测试衡量的是相同 victim 标准化输入下，完整可学习仿射参数带来的尺度和平移
表征迁移；running state 仍不进入候选。

## 表征传输分数

对每个输出张量，将 `N×C×H×W` 整理为 `(N×H×W)×C`，把每个空间位置
视为一个通道坐标样本。用 population mean/covariance 建立 public/victim
高斯二阶表征：

```text
(mu_p, Sigma_p)
(mu_v, Sigma_v)
```

未归一化的二阶 Wasserstein 传输能量为：

```text
W2^2 = ||mu_v - mu_p||_2^2
       + Tr(Sigma_v + Sigma_p
            - 2 * sqrt(sqrt(Sigma_p) * Sigma_v * sqrt(Sigma_p)))
```

唯一排名分数使用对称二阶能量归一化：

```text
RT = W2^2 /
     (0.5 * (E[||z_p||_2^2] + E[||z_v||_2^2]) + epsilon)
```

`RT` 表示把 public 算子输出的通道几何传输成 victim 输出几何所需的
相对能量。均值项和协方差项只是同一 `W2^2` 的精确分解，不是额外排名指标。

协方差全部使用 `float64` 累积。理论上的半正定矩阵仅将浮点误差产生的负特征值
截断为零，不加经验 shrinkage 或人工平滑系数。

## 与 Test01 对照

Test02 必须读取 `results/test/MS/01_cross/all.tsv`，按
`abs(cross_abs_mean)` 降序重建 Test01 rank，并对同名的 40 个候选记录：

```text
Test02 RT rank / score
Test01 cross rank / absolute mean
rank delta
Spearman rank correlation
Kendall rank correlation
Top-5 / Top-10 / Top-20 overlap
```

对照只用于判断两个数据选择器是否表达相同结构，不参与 RT 分数或排名计算。

## 科学判定

Test02 不以“复现已知后验候选”作为优化目标。当前只检验：

1. `layer4.1.bn2` 的尺度迁移是否在 RT 中仍然明显；
2. 阶段转换或前中层卷积是否产生较大的算子内部表征迁移；
3. Test01 中的高分是来自算子自身迁移，还是输入残差与权重残差的交互；
4. RT 是否给出具有清晰均值/协方差几何含义的独立排名。

本阶段不生成保护 mask，不训练 surrogate，不读取 `eval_ms`。必须在公开
排名后再单独固定保护预算和 MS 验证协议，不能根据 MS 结果回改本公式。

## 运行

只处理第一个 batch 并核对算子输出，不写结果：

```bash
PYTHONDONTWRITEBYTECODE=1 \
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  test/MS/02/transport.py --dry-run
```

完整计算 500 张图片的 40 项排名与 Test01 对照：

```bash
PYTHONDONTWRITEBYTECODE=1 \
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  test/MS/02/transport.py
```

## 输出

```text
results/test/MS/02/metrics.json
results/test/MS/02/weights.tsv
results/test/MS/02/weights_conv.tsv
results/test/MS/02/weights_bn.tsv
results/test/MS/02/weights.png
    40 个 backbone 候选的 RT 协议、排名、分类表和统一图

results/test/MS/02/tensors.tsv
results/test/MS/02/tensors.png
    从 40 项排名直接抽取的 16 个 BasicBlock 主分支卷积

results/test/MS/02/comparison.tsv
results/test/MS/02/comparison.png
    Test02 RT 与 Test01 500-query 交叉残差的逐项排名对照
```
