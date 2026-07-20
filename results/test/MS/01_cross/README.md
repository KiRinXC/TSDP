# 测试 01 结果：交叉残差绝对均值

本目录保存 ResNet18 public model 与 CIFAR-100 victim model 在相同 weight
算子位置的交叉残差排名。当前结果只报告残差统计，不包含保护 mask、surrogate
训练或 MS 指标。

## 评分口径

对每张图片、每个候选算子得到交叉残差张量 \(I\) 后，先逐元素取绝对值，再对
输出通道和空间维度 `C×H×W` 取平均，最后对所有输入图片取平均：

```text
score = mean_image(mean_chw(abs(I)))
```

因此，不同输出分辨率不会因为空间位置更多而天然获得更高分。这里没有保留符号，
也没有使用平方、范数、空间求和或只除通道数的旧定义。

候选共 40 个：

- 20 个 `Conv2d.weight`，包括 stem、BasicBlock 主分支和 downsample 卷积；
- 20 个 `BatchNorm2d.weight`，即 BN gamma。

分类头不参与本次交叉残差排名。

## 结果文件

```text
weights.json / weights.tsv
    固定 500 张 query 图片的 40 项统一排名、协议和正确性检查
weights_conv.tsv / weights_bn.tsv
    上述结果按 Conv weight 与 BN gamma 分别抽取的类别内排名
weights.png
    500 张 query 的 40 项统一排名图
weights_full.json / weights_full.tsv
    CIFAR-100 全部 50,000 张训练图片的 40 项统一排名
weights_full_conv.tsv / weights_full_bn.tsv
    全训练集结果的两类候选排名
weights_full.png
    全训练集的 40 项统一排名图
tensors.tsv / tensors.png
    从 500 张 query 的 40 项结果直接抽取的 16 个 BasicBlock 主分支卷积
```

## 40 项排名

500 张 query 与全部 50,000 张训练图片的前 10 名顺序完全一致：

| 排名 | 类型 | 模块 | 500 张分数 | 50,000 张分数 |
|---:|---|---|---:|---:|
| 1 | BN gamma | `layer4.1.bn2` | 1.464529 | 1.464361 |
| 2 | Conv weight | `layer2.0.conv1` | 0.905005 | 0.902746 |
| 3 | Conv weight | `layer1.0.conv1` | 0.816918 | 0.811313 |
| 4 | Conv weight | `layer1.1.conv1` | 0.789595 | 0.787009 |
| 5 | Conv weight | `layer3.0.conv1` | 0.683209 | 0.683028 |
| 6 | Conv weight | `layer4.0.conv1` | 0.512609 | 0.509288 |
| 7 | Conv weight | `layer2.1.conv1` | 0.492335 | 0.492292 |
| 8 | Conv weight | `layer4.1.conv1` | 0.480868 | 0.478597 |
| 9 | Conv weight | `layer3.0.conv2` | 0.477448 | 0.476088 |
| 10 | Conv weight | `layer2.0.downsample.0` | 0.408543 | 0.406461 |

两种输入规模的前 23 名顺序也完全一致，差异只出现在后部若干相近分数之间。
固定 500 张 query 已足以稳定呈现这套排名的主要结构；全训练集结果仅用于本次
对照，不再为原 16 个卷积单独重复一套全数据计算。

stem `conv1` 在两种输入规模下都接近零。这是交叉项定义的结构结果：两模型在第一
个卷积前接收完全相同的图片，输入差为零；不能把该分数直接解释成 stem 不重要。

## 16 个主分支卷积

下表直接从 500 张的 40 项统一表中抽取，不执行第二次前向计算：

| 子集排名 | 40 项排名 | 模块 | 交叉残差绝对均值 |
|---:|---:|---|---:|
| 1 | 2 | `layer2.0.conv1` | 0.905005 |
| 2 | 3 | `layer1.0.conv1` | 0.816918 |
| 3 | 4 | `layer1.1.conv1` | 0.789595 |
| 4 | 5 | `layer3.0.conv1` | 0.683209 |
| 5 | 6 | `layer4.0.conv1` | 0.512609 |
| 6 | 7 | `layer2.1.conv1` | 0.492335 |
| 7 | 8 | `layer4.1.conv1` | 0.480868 |
| 8 | 9 | `layer3.0.conv2` | 0.477448 |
| 9 | 11 | `layer2.0.conv2` | 0.345800 |
| 10 | 12 | `layer3.1.conv1` | 0.327012 |
| 11 | 15 | `layer1.0.conv2` | 0.290695 |
| 12 | 16 | `layer2.1.conv2` | 0.280245 |
| 13 | 19 | `layer1.1.conv2` | 0.257663 |
| 14 | 22 | `layer4.0.conv2` | 0.194684 |
| 15 | 30 | `layer3.1.conv2` | 0.134839 |
| 16 | 32 | `layer4.1.conv2` | 0.131814 |

按当前定义，七个 `conv1` 位于 16 个主分支卷积的前七名；
`layer3.0.conv2` 排在第八。该表只描述交叉残差大小，不把排名直接解释成 MS
攻击依赖或保护效果。
