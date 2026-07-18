# 实验 06：TensorShield Weight 语义闭包

本实验在 `ResNet18+CIFAR-100` 上检查 TensorShield eligible Top-k 距离当前 soft-posterior 黑盒边界时，缺失的 weight 语义来自哪里。实验固定扫描 `k=10,...,17`，将 Lab04 的原始 Top-k 前缀与 BN gamma、downsample Conv、二者组合、Stem Conv 及三者并集进行受控比较。

本实验不重新计算 TensorShield rank，也不改变攻击者的查询输出。作者 eligible rank、Top-k 顺序和固定分类头 bias 均直接继承 Lab04；新增语义只来自 Lab05 已验证的参数分类。

## 六种保护组合

对每个 `k=10,...,17`，构造以下六组：

```text
top_k                 Lab04 eligible Top-k，并固定保护 last_linear.bias
top_k_bn_gamma        top_k + 全部 20 个 BN gamma
top_k_downsample_conv top_k + 3 个 downsample Conv weight
top_k_bn_gamma_downsample
                      top_k + BN gamma + downsample Conv
top_k_stem_conv       top_k + Stem Conv weight
top_k_all_extras      top_k + BN gamma + downsample Conv + Stem Conv
```

Top-10 至 Top-17 均已包含 rank-3 `last_linear.weight`，再加固定的 `last_linear.bias` 后，六组的分类头模式全部为 `replace`。三类额外参数与 17 个 eligible weight 不重叠：

```text
额外语义          state tensor  参数数量   模型参数比例
BN gamma                  20       4,800       0.0428%
downsample Conv            3     172,032       1.5322%
BN gamma + downsample     23     176,832       1.5749%
Stem Conv                  1       9,408       0.0838%
三类并集                  24     186,240       1.6587%
```

因此 `Top-17 + all_extras` 等于全部 Conv weight、全部 BN gamma、分类头 weight 与分类头 bias，即 Lab05 `weight` 组再增加 100 个分类头 bias。该终点用于检查两项独立 Lab 的结果是否一致；不能用 Lab05 数值替代本实验的受控训练。

## 固定协议

```text
数据划分          dataset/MS/c100/manifest.json 中的 query_pool_ms 与 eval_ms
victim            weights/MS/victim/resnet18/c100/best.pth
surrogate 初始化  formal_victim_then_public_v1：ImageNet-1K backbone + 固定随机分类头
攻击者可观测输出  victim soft posterior
query transform   确定性的 test transform
query budget      500，即 CIFAR-100 训练集的 1%
query 划分        seed 42、offset 100 固定拆为 400 train / 100 validation
Top-k             TensorShield 作者 eligible rank 的 k=10,...,17
固定分类头状态    每组均保护 last_linear.weight 与 last_linear.bias
额外保护          BN gamma、downsample Conv、二者组合、Stem Conv 或三者并集
暴露状态          从 victim 复制；保护状态保留公开预训练/随机初始化值
训练方式          所有 surrogate 参数共同微调，不冻结暴露权重
训练轮数          100
优化器            SGD，lr=0.01，momentum=0.5，weight_decay=5e-4
学习率调度        StepLR，step_size=60，gamma=0.1
选模               validation soft cross-entropy 最低的最早 epoch
主要评估点        checkpoint 固定后只评估一次完整 eval_ms
原始指标          surrogate accuracy、fidelity、posterior KL
随机种子          每个组合均把 seed 42 传给共享 canonical 初始化器
```

每个组合独立重放与正式 MS 入口、Lab04 和 Lab05 相同的 canonical 构造轨迹。以后扩展多随机种子时只替换实验 seed，不改变 victim→public→任务头的构造顺序。

`top_k` 的 8 个点直接读取 `results/lab/04_tensorshield/metrics.json`，并核对源文件 SHA256、mask、保护参数量和协议，不重复训练。其余五条曲线各训练 8 组，共新增 40 组、4,000 轮 query train/validation 记录。Lab05 只提供 `weight` 终点交叉参考，不参与结果复用。

soft-posterior `full_protection` 是与部分保护策略同接口的主黑盒边界；
hard-label `hard_blackbox` 作为第二条正式参考线同时绘制，但不用于选择组合。

## 运行方式

先验证 Lab04/Lab05 输入哈希、48 个 mask、参数量、分类头模式和终点闭包：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/06_weight/run.py --dry-run
```

运行完整实验：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/06_weight/run.py
```

完整运行只复用同协议 Lab04 的八个 Top-k 点，其余四十组均独立训练并覆盖同一语义
入口。

## 输出

```text
results/lab/06_weight/metrics.json       48 个组合、输入哈希、选模信息与单次 eval_ms 指标
results/lab/06_weight/history.tsv        40 个新增组合共 4,000 轮 query train/validation 记录
results/lab/06_weight/data.tsv           六条曲线的原始指标、保护成本与相对 Top-k 差值
results/lab/06_weight/metrics.png        accuracy、fidelity、posterior KL 三联曲线
results/lab/06_weight/<case>_mask.pt     40 个新增组合的紧凑保护掩码
```

横轴使用 `Top-k`，便于在相同作者前缀下比较新增语义；每个点的实际保护参数数量与
比例保存在 `data.tsv` 和 `metrics.json`。主图同时绘制 soft 与 hard-label 两条
full-protection 黑盒水平线；不绘制 no-protection 白盒线，以免压缩部分保护结果的
关键区间。三条边界的原始数值均保存在 `metrics.json`。

## 结果判据

```text
单个新增语义达到黑盒    该参数家族是主要遗漏来源
BN gamma 与 downsample
组合优于两个单项        两类语义具有互补作用
单项均不足、并集达到    三类 weight 语义具有互补作用
较小 k 的并集达到       后部大卷积可被低成本语义闭包替代
Top-17 并集仍未达到     后续再检查 BN beta/affine 或其他执行状态
```

本实验只报告原始三项 MS 指标并直接对照当前 soft 黑盒线，不在运行过程中定义或优化黑盒等效阈值。
