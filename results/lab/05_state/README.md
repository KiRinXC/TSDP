# 实验 05 结果

本目录记录 `ResNet18+CIFAR-100` 的五种完整 `state_dict` 类型与十三种参数语义组。
十八组均按 400/100 soft-posterior validation-best 协议训练，checkpoint 固定后各对
`eval_ms` 评估一次。

```text
保护组                    参数占比  best  accuracy  fidelity  posterior KL
weight                    99.9564%    99   0.1435    0.1490    2.902834
bias                       0.0436%    92   0.5755    0.7558    0.309087
running_mean               0.0000%    67   0.6074    0.8715    0.088244
running_var                0.0000%    67   0.6074    0.8715    0.088244
num_batches_tracked        0.0000%     1   0.6119    0.9111    0.038390
main_conv                 97.8416%    79   0.2210    0.2468    2.206965
downsample_conv            1.5322%    71   0.5400    0.6730    0.525170
bn_gamma                   0.0428%    96   0.4571    0.5439    1.060194
bn_beta                    0.0428%    92   0.5762    0.7567    0.308892
bn_affine                  0.0855%    52   0.4349    0.5061    1.146158
head_weight                0.4560%    93   0.3984    0.4622    1.616744
head_bias                  0.0009%     1   0.6122    0.9113    0.038495
downsample_branch          1.5482%    71   0.5381    0.6758    0.573691
stem_branch                0.0849%    76   0.5427    0.6915    0.487547
stem_conv                  0.0838%    92   0.5546    0.7114    0.429770
stem_bn_affine             0.0011%    65   0.5957    0.8083    0.184036
downsample_bn_affine       0.0160%    60   0.5859    0.7931    0.251477
head                       0.4569%    93   0.3985    0.4621    1.616573
```

主要观察：

- 只保护主路径 Conv 即使占全部参数的 `97.84%`，仍明显弱于保护全部 weight；
  Stem、downsample、BN affine 和分类头 weight 不能从后续建模中直接排除。
- 相同参数量下，BN gamma 的保护信号远强于 BN beta；全部 BN affine 又强于单独
  gamma，说明二者存在补充，但 gamma 是更优先的候选。
- 分类头的作用几乎全部来自 weight，单独保护 bias 接近无效。
- `running_mean` 与 `running_var` 的结果逐值相同，BN buffer 更适合作为执行闭包
  状态，而不是独立的长期保护来源。

## BN Gamma 分组消融

本消融固定 Lab04 最终候选中的五个 `conv1.weight` 和完整分类头，仅改变 Stem、
BasicBlock BN1、BasicBlock BN2 与 Downsample BN 四类 gamma。所有新 case 使用
seed 43–52 完整训练；`All 20 gamma` 与 matched soft 黑盒复用 Lab04 中 mask、
query 划分和训练协议完全相同的十种子结果。

十种子均值与样本标准差如下。Accuracy 与 Fidelity 越低越好，Posterior KL 越高
越好；“黑盒三线”表示同 seed 下三项同时达到或越过 matched soft 黑盒的次数。

| Gamma 配置 | Gamma 数 | 参数占比 | Accuracy | Fidelity | Posterior KL | 黑盒三线 |
|---|---:|---:|---:|---:|---:|---:|
| No gamma | 0 | 5.7101% | 0.18249±0.00599 | 0.19780±0.00652 | 2.61784±0.02845 | 0/10 |
| All 20 gamma | 20 | 5.7529% | 0.11259±0.00466 | 0.12141±0.00495 | 3.17270±0.05087 | 10/10 |
| All - Stem | 19 | 5.7523% | 0.12567±0.00710 | 0.13576±0.00850 | 3.05511±0.05683 | 9/10 |
| All - Block BN1 | 12 | 5.7358% | **0.10654±0.00443** | **0.11475±0.00540** | **3.21235±0.04102** | **10/10** |
| All - Block BN2 | 12 | 5.7358% | 0.13484±0.00520 | 0.14453±0.00603 | 2.96278±0.02271 | 4/10 |
| All - Downsample | 17 | 5.7449% | 0.13483±0.00522 | 0.14515±0.00470 | 2.96331±0.03633 | 6/10 |

相对 `All 20 gamma` 的同 seed 配对变化为：

| 删除组 | Accuracy 变化 | Fidelity 变化 | KL 变化 | 三项同时反弹 |
|---|---:|---:|---:|---:|
| Stem | +0.01308 | +0.01435 | -0.11758 | 9/10 |
| Block BN1 | -0.00605 | -0.00666 | +0.03965 | 0/10 |
| Block BN2 | +0.02225 | +0.02312 | -0.20992 | 10/10 |
| Downsample BN | +0.02224 | +0.02374 | -0.20939 | 10/10 |

因此，当前固定候选中的关键 gamma 不是全部 20 个平均起作用：

- Block BN2 与 Downsample BN 是最稳定的条件必要组。任意删除一组，10/10 seed 的
  三项都向攻击者有利方向反弹；二者分别控制残差分支相加前尺度与阶段转换 identity
  分支尺度，说明这两条分支的联合尺度闭包对当前攻击最关键。
- Stem gamma 有中等但稳定的贡献，删除后 9/10 seed 三项同时反弹。
- Block BN1 在当前固定直接拼接攻击下不是必要保护项。删除它后平均三项反而进一步
  改善，7/10 seed 三项同时改善，并保持 10/10 黑盒三线。这很可能来自受保护 public
  `conv1` 与暴露 victim `bn1` 的接口失配，不能解释为“主动泄露 BN1 会提高安全性”；
  能选择 public BN1 或执行状态收缩的适应性攻击仍需另行验证。
- 全部 gamma 相对 No gamma 在 10/10 seed 三项同时改善，证明总体 gamma 闭包的
  收益稳定；但最小经验闭包应优先保留 Stem、Block BN2 和 Downsample，而不是机械
  保护全部 20 个 gamma。

## BN Gamma 单组加入实验

该实验只使用 seed 42，固定同一组五个 `conv1.weight` 和完整分类头，以 No gamma
为基线，分别只加入一类 gamma。它与上一节的十种子 drop 消融互为补充：drop 回答
一类 gamma 在完整闭包中是否必要，add 回答该类单独加入时是否具有保护充分性。

| 配置 | Gamma 数 | 参数占比 | Best epoch | Accuracy | Fidelity | Posterior KL |
|---|---:|---:|---:|---:|---:|---:|
| No gamma | 0 | 5.7101% | 94 | 0.1798 | 0.1947 | 2.642482 |
| + Stem | 1 | 5.7107% | 98 | 0.1546 | 0.1699 | 2.782362 |
| + Block BN1 | 8 | 5.7272% | 93 | 0.1785 | 0.1907 | 2.645210 |
| + Block BN2 | 8 | 5.7272% | 93 | 0.1481 | 0.1633 | 2.803683 |
| + Downsample BN | 3 | 5.7181% | 89 | **0.1376** | **0.1509** | **2.939023** |

相对 No gamma 的变化如下；accuracy/fidelity 为负、KL 为正代表保护改善：

| 单独加入组 | Accuracy 变化 | Fidelity 变化 | KL 变化 |
|---|---:|---:|---:|
| Stem | -0.0252 | -0.0248 | +0.139880 |
| Block BN1 | -0.0013 | -0.0040 | +0.002728 |
| Block BN2 | -0.0317 | -0.0314 | +0.161201 |
| Downsample BN | **-0.0422** | **-0.0438** | **+0.296541** |

单独加入的排序是 Downsample BN > Block BN2 > Stem >> Block BN1，与 drop 消融中
Downsample、Block BN2 和 Stem 的必要性方向一致。Downsample 只用 3 个 gamma、
896 个参数就产生最大改善，说明阶段转换 identity 分支的尺度是当前直接拼接攻击的
高杠杆接口。Block BN1 几乎没有独立收益，也与 drop 实验中删除它未造成反弹一致。

不过最强的 Downsample 仍未同时达到 seed-42 soft 黑盒三线
`0.1390/0.1463/3.039817`：accuracy 已略低于黑盒，但 fidelity 仍高 0.0046，KL
仍低 0.100794。因此四类中没有任何单独一组足以替代跨层组合；本节也只有一个
seed，不能据此声明随机种子稳定性。

```text
metrics.json                    十八组保护统计、validation-best 与最终原始指标
history.tsv                     十八组各 100 轮 train/validation 日志
data.tsv                        三张图使用的十八个单次 eval_ms 原始点
accuracy/fidelity/posterior_kl.png
<group>_mask.pt                 十八组紧凑保护掩码
gamma.json                      六种 gamma 配置的十种子结果、配对效应与来源哈希
gamma.tsv                       六种配置共 60 个 MS 原始点
gamma_history.tsv               六种配置共 6,000 轮 train/validation 日志
gamma.png                       Accuracy、Fidelity 与 Posterior KL 三联消融图
gamma_<case>_mask.pt            六种配置的紧凑保护掩码
gamma_add.json                  No gamma 与四类单独加入的 seed-42 结果
gamma_add.tsv                   五种 add 配置及相对 No gamma 的原始差值
gamma_add_history.tsv           五种配置共 500 轮 train/validation 日志
gamma_add.png                   单组加入的三指标柱状图
gamma_add_<case>_mask.pt        五种 add 配置的紧凑保护掩码
```
