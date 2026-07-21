# 测试 02 结果：任务特定表征传输

本目录保存相同 victim 输入下 20 个 Conv weight 与 20 个 BN affine 参数组的
高斯二阶表征传输排名。分类头固定为私有边界，不参与排名。本测试没有
生成保护 mask，没有训练 surrogate，也没有读取 `eval_ms`。

## 结果文件

```text
metrics.json
    固定协议、输入/权重哈希、数值检查、40 项排名与 Test01 聚合对照
weights.tsv / weights.png
    40 个候选的统一 RT 排名和柱状图
weights_conv.tsv / weights_bn.tsv
    Conv weight 与 BN affine 的类别内排名
tensors.tsv / tensors.png
    从统一表直接抽取的 16 个 BasicBlock 主分支卷积
comparison.tsv / comparison.png
    Test02 RT 与 Test01 500-query 交叉残差的逐项对照
```

## RT 排名

前 10 项为：

| 排名 | 类型 | 模块 | RT | 均值传输 | 协方差传输 |
|---:|---|---|---:|---:|---:|
| 1 | Conv | `layer3.1.conv1` | 1.958194 | 22.860168 | 12.183278 |
| 2 | BN affine | `layer3.1.bn1` | 1.743032 | 15.743798 | 15.787958 |
| 3 | BN affine | `layer3.1.bn2` | 1.735015 | 11.014056 | 13.483849 |
| 4 | Conv | `layer3.1.conv2` | 1.567019 | 0.027062 | 0.065960 |
| 5 | BN affine | `layer4.1.bn2` | 1.545050 | 39.701159 | 1659.451512 |
| 6 | Conv | `layer4.1.conv2` | 1.376031 | 0.243167 | 0.687352 |
| 7 | Conv | `layer4.0.downsample.0` | 1.240311 | 13.056310 | 20.196567 |
| 8 | Conv | `layer3.0.downsample.0` | 1.210310 | 6.226456 | 12.399534 |
| 9 | Conv | `layer4.0.conv2` | 1.172746 | 1.806090 | 4.451191 |
| 10 | Conv | `layer4.0.conv1` | 1.126654 | 47.712405 | 76.380834 |

`layer4.1.bn2` 在 Test01 排第 1，在 Test02 排第 5，说明它的强变化不只是
输入残差与 affine 残差的交互；将输入固定为 victim 坐标后，public/victim
affine 仍会产生显著的通道均值与协方差迁移。相对原 gamma-only 定义，beta 只改变
均值而不改变协方差，因此 BN 项的 covariance transport 保持不变，但 mean
transport 与最终 RT 已按完整 affine 重新计算。

## 与 Test01 的对照

```text
Spearman rank correlation    0.596998
Kendall rank correlation     0.458974
Top-5 overlap                1/5
Top-10 overlap               2/10
Top-20 overlap               15/20
```

Top-5 唯一重合项是 `layer4.1.bn2`。Top-10 另外重合
`layer4.0.conv1`。两种方法在中等范围上存在共同结构，但对最高优先级算子
的判断明显不同：Test01 强调输入差与权重差的乘性交互，Test02 强调相同
victim 输入下算子自身产生的相对表征几何迁移。

## 与已有机制证据的核对

已有 5.7529% 后验候选中的五个 `conv1` 在 RT 统一表中为：

| 模块 | RT 排名 | RT |
|---|---:|---:|
| `layer2.0.conv1` | 13 | 0.957223 |
| `layer1.1.conv1` | 14 | 0.902759 |
| `layer3.0.conv1` | 16 | 0.841787 |
| `layer2.1.conv1` | 21 | 0.690377 |
| `layer1.0.conv1` | 25 | 0.441461 |

当前 RT 没有把这五个位置聚集到排名头部，也没有将全部 BN affine 识别为一个
联合闭包。这与 Lab09 的机制证据一致：其中 BN gamma 的保护效果主要来自跨层交互，
即使单算子评分补入 beta，也不能由 20 个独立 affine 分数直接表达。

`layer3.1` 的两个卷积和两个 BN 占据 RT 前四名。其中
`layer3.1.conv2` 的未归一化传输能量只有 `0.093022`，但该算子的
public/victim 对称二阶能量也只有 `0.059362`，归一化后 RT 为
`1.567019`。因此当前 RT 衡量的是“相对几何迁移”，会把被 victim 显著压低
的低能量算子排到高位。而 Lab09 在 coherent victim 中替换
`layer3.1.conv1/conv2` 的 KL 损伤只有 `0.00209/0.00419`，说明相对迁移大不等于
该算子对最终任务或攻击具有大影响。

## 当前结论

Test02 成功将“算子自身的表征迁移”与 Test01 的“输入差×权重差交互”
分开，并确认 `layer4.1.bn2` 的强尺度迁移在两种定义下都存在。

但当前“单算子相对 RT”还不能作为最终保护位置选择器：

1. 它会高估被显著压低的低能量算子；
2. 它不表达 BN affine 的跨层联合闭包；
3. 它没有将已有五个后验 `conv1` 独立聚集到头部。

该结论只否定“相对单算子 RT 直接等于攻击依赖”，不否定表征坐标和尺度迁移
本身。后续若继续这条主线，需要在不读取 MS 结果的前提下，区分“局部相对迁移”和
“能够沿后续计算图保持的表征迁移”。
