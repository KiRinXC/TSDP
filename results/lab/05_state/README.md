# 实验 05 结果

本目录保存 `ResNet18+CIFAR-100` 下五种完整 `state_dict` 类型与十三种参数语义组的独立保护结果。主要结果统一读取第 100 轮 `end`。所有组均采用 `formal_victim_then_public_v1`、seed 42，并在每组开始前重置初始化与 query sampler。

```text
保护组                    参数比例   state 存储比例  accuracy  fidelity  posterior KL
weight                  99.956358%      99.870611%    0.1540    0.1642       2.767021
bias                     0.043642%       0.043604%    0.5773    0.7629       0.288108
running_mean             0.000000%       0.042714%    0.6109    0.8801       0.076602
running_var              0.000000%       0.042714%    0.6109    0.8801       0.076602
num_batches_tracked      0.000000%       0.000356%    0.6109    0.8801       0.076602
main_conv               97.841610%      97.757677%    0.2390    0.2601       2.133624
downsample_conv          1.532195%       1.530881%    0.5508    0.6876       0.482493
bn_gamma                 0.042751%       0.042714%    0.4948    0.5899       0.843716
bn_beta                  0.042751%       0.042714%    0.5773    0.7625       0.288342
bn_affine                0.085502%       0.085429%    0.4715    0.5603       0.945171
head_weight              0.456010%       0.455619%    0.4408    0.5139       1.347826
head_bias                0.000891%       0.000890%    0.6107    0.8800       0.076540
downsample_branch        1.548156%       1.562828%    0.5475    0.6891       0.531090
stem_branch              0.084932%       0.086016%    0.5503    0.7022       0.451142
stem_conv                0.083792%       0.083720%    0.5611    0.7203       0.403427
stem_bn_affine           0.001140%       0.001139%    0.5967    0.8224       0.158504
downsample_bn_affine     0.015960%       0.015946%    0.5894    0.7985       0.233450
head                     0.456901%       0.456509%    0.4404    0.5135       1.347578
```

当前正式参考点为：

```text
方案          accuracy  fidelity  posterior KL
无保护          0.6182    1.0000       0.000000
全保护          0.1545    0.1610       2.835290
```

所有 Lab05 保护组统一额外微调 100 轮，因此分析语义组的边际影响时，使用 `num_batches_tracked` 的 `0.6109/0.8801/0.076602` 作为同训练轨迹的近似空操作对照，不把未经额外微调的正式无保护点直接当作差值基准。

## 结论

`main_conv` 占全部参数的 `97.8416%`，但攻击 accuracy 为 `23.90%`，比 `weight` 组高 `8.50` 个百分点。Stem Conv、downsample Conv、BN gamma 和分类头 weight 合计体量很小，却共同填补了主路径 Conv 与近黑盒结果之间的明显缺口。后续不能只在 16 个主路径 Conv 中寻找通道块。

BN gamma 只有 `4,800` 个参数，单独保护已把攻击 accuracy 从同训练对照的 `61.09%` 降到 `49.48%`；BN beta 单独保护降到 `57.73%`。二者合并为 `bn_affine` 后进一步降到 `47.15%`，说明 gamma 是主要贡献，beta 具有较小但不可忽略的组合效应。通道块应携带对应 gamma 和 beta，不能把 BN affine 排除在节点状态之外。

`bias` 与 `bn_beta` 的三项指标几乎完全一致，而 `head_bias` 与近似空操作对照几乎一致，说明原 `bias` 组的可测作用来自 BN beta，不是分类头的 100 个 bias。`head_weight` 的 accuracy/fidelity 为 `0.4408/0.5139`，同时保护 weight 与 bias 的 `head` 为 `0.4404/0.5135`，差异只有 4 个 accuracy 和 4 个 fidelity 样本。分类头 bias 可因成本极低而随 weight 固定保护，但当前没有证据把它计为独立安全贡献。

本次 `head` 的三个 end 指标与正式 `head_only` 逐值相同，说明统一随机轨迹已经消除了此前 Lab 与 exp 的随机分类头混杂，也给后续跨入口比较提供了直接回归检查。

单独保护 downsample Conv 后攻击 accuracy 为 `55.08%`；只保护 downsample BN affine 为 `58.94%`；保护完整 downsample 分支为 `54.75%`。完整分支相对 Conv-only 的 accuracy 仅再降低 `0.33` 个百分点，且 fidelity 没有同步降低。当前没有证据表明三个 downsample 分支的全部 BN 状态必须整体保护，但 downsample Conv 和对应 BN affine 都是可测的攻击信息来源，不能在候选图中删除。

Stem Conv 单独保护时 accuracy/fidelity 为 `0.5611/0.7203`，首个 BN affine 单独保护为 `0.5967/0.8224`，完整 Stem 为 `0.5503/0.7022`。Stem 的主要贡献来自 Conv，首个 BN affine 作用较弱但与 Conv 组合后仍有增益。完整 Stem 只占 `0.0849%` 参数，足以否定“Stem 可以未经验证直接排除”的假设。

三种 BN buffer 单独保护的 end 指标完全一致。100 轮微调后，公开初始化的 running mean/variance 已被相同 query 流程覆盖；buffer 应作为通道块执行状态记录，但不能被解释为独立的长期安全来源。

```text
metrics.json                    十八组协议、统一随机轨迹、保护统计、best/end 和正式参考指标
history.tsv                     十八组共 1,800 轮训练与评估记录
data.tsv                        三张图使用的十八个 end 原始点
accuracy.png                    保护存储比例与 accuracy
fidelity.png                    保护存储比例与 fidelity
posterior_kl.png                保护存储比例与 posterior KL
<group>_mask.pt                 十八组各自的紧凑保护掩码
```
