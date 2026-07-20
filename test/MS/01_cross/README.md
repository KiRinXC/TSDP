# 测试 01：交叉残差绝对均值

## 目的

本测试比较 ResNet18 public model 与 victim model 相同 weight 算子位置的交叉残差，
在完全不读取 MS 指标的条件下生成先验排名。当前只计算 40 个乘性 weight tensor：

```text
20 个 Conv2d weight
20 个 BatchNorm2d gamma，即 bn.weight
```

分类头、BN beta、BN running state、池化、ReLU 和残差相加不参与排名。BN
`running_mean` 与 `running_var` 只用于分别构造 public/victim 的自然标准化输入。

## 固定输入

```text
模型                  ResNet18
数据集                CIFAR-100
public 权重           weights/pre_train/resnet18-5c106cde.pth
victim 权重           weights/MS/victim/resnet18/c100/best.pth
模型模式              eval
输入变换              确定性 test transform
随机种子              42
batch size            64
```

同一个公式分别应用于两个数据范围：

```text
query                  query_pool_ms 按 query_rank 排序的固定前 500 张
full                   official_train 的全部 50,000 张，source_index 升序
```

这里不再单独使用 50,000 张图片复算 16 个 BasicBlock 主分支卷积的排名稳定性。
50,000 张范围只用于计算完整 40 项排名。

## Conv weight 交叉残差

对同一张图片，在卷积位置 `l` 获得 public/victim 自然传播输入 `h_p`、`h_v`，再将
两个输入分别送入 public/victim 卷积权重：

```text
z_pp = Conv(h_p, W_p)
z_pv = Conv(h_p, W_v)
z_vp = Conv(h_v, W_p)
z_vv = Conv(h_v, W_v)
```

交叉残差为：

```text
I = z_vv - z_vp - z_pv + z_pp
  = Conv(h_v-h_p, W_v-W_p)
```

目标特征图是 Conv2d 的直接输出，尚未进入紧随其后的 BN。

## BN gamma 交叉残差

BN 在 eval 模式下先分别使用两模型自己的 running state 标准化输入：

```text
h_hat_p = (h_p - mean_p) / sqrt(var_p + eps)
h_hat_v = (h_v - mean_v) / sqrt(var_v + eps)
```

只把 gamma 乘法视为被评分算子：

```text
I_gamma = gamma_v*h_hat_v - gamma_p*h_hat_v
          - gamma_v*h_hat_p + gamma_p*h_hat_p
        = (gamma_v-gamma_p)*(h_hat_v-h_hat_p)
```

beta 是位于 gamma 之后的加性项，在四路交叉差分中抵消，因此不进入候选。

## 唯一评分

Conv 与 BN gamma 完全使用同一个评分：

```text
score = mean_image(mean_{c,h,w}(|I[c,h,w]|))
```

即对每张图片：

1. 对交叉残差逐元素取绝对值；
2. 对 `C×H×W` 的全部元素取平均；
3. 最后对全部输入图片取平均。

这里没有符号抵消，不使用平方、RMS、自然输出参考量、参数量归一化或其他组合指标。
代码实现必须等价于：

```python
interaction.abs().mean(dim=(1, 2, 3))
```

## 16 个主分支卷积图

500-query 的 40 项结果固定后，从中直接提取八个 BasicBlock 主分支的 `conv1` 和
`conv2`，共 16 个 Conv weight：

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

该子集不单独执行第二次前向计算，表格中的分数必须与 500-query 的 40 项总表逐值
一致。`tensors.png` 只展示这 16 项。

## 正确性门槛

每个数据范围都必须满足：

1. public/victim 恰好捕获同名的 20 个 Conv2d 和 20 个 BatchNorm2d；
2. 手工 Conv/BN 计算与 hook 捕获的自然输出一致；
3. 四项展开与紧凑交叉式一致；
4. stem Conv 因两模型输入相同，交叉残差接近零；
5. query/full 分别恰好处理 500/50,000 张图片；
6. 40 个分数全部有限，并按绝对均值降序保存。

## 运行

分别用一个 batch 核对两个范围，不写结果：

```bash
PYTHONDONTWRITEBYTECODE=1 \
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  test/MS/01_cross/weights.py --scope both --dry-run
```

完整计算 500-query 与 50,000-image 两套结果：

```bash
PYTHONDONTWRITEBYTECODE=1 \
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" \
  test/MS/01_cross/weights.py --scope both
```

需要单独重算某一范围时，可将 `--scope both` 改为 `--scope query` 或
`--scope full`。允许调整的参数只有设备与 DataLoader worker 数；模型、数据范围、
评分公式和候选集合均已固定。

## 输出

```text
results/test/MS/01_cross/weights.json
results/test/MS/01_cross/weights.tsv
results/test/MS/01_cross/weights_conv.tsv
results/test/MS/01_cross/weights_bn.tsv
results/test/MS/01_cross/weights.png
    500-query 的 40 项协议、总排名、分类别排名和图

results/test/MS/01_cross/weights_full.json
results/test/MS/01_cross/weights_full.tsv
results/test/MS/01_cross/weights_full_conv.tsv
results/test/MS/01_cross/weights_full_bn.tsv
results/test/MS/01_cross/weights_full.png
    50,000-image 的 40 项协议、总排名、分类别排名和图

results/test/MS/01_cross/tensors.tsv
results/test/MS/01_cross/tensors.png
    从 500-query 总排名直接提取的 16 个主分支卷积及单独图
```
