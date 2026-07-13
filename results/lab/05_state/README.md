# 实验 05 结果

本目录保存 `ResNet18+CIFAR-100` 下五种 `state_dict` 条目类型的独立保护结果。主要结果使用第 100 轮 `end` 指标，横坐标比例按原始 tensor payload 字节数计算。

```text
保护类型                  state 存储比例  参数比例    accuracy  fidelity  posterior KL
weight                       99.870611%   99.956358%    0.1582    0.1670       2.785335
bias                          0.043604%    0.043642%    0.5771    0.7636       0.288129
running_mean                  0.042714%    0.000000%    0.6107    0.8802       0.076649
running_var                   0.042714%    0.000000%    0.6107    0.8802       0.076649
num_batches_tracked           0.000356%    0.000000%    0.6107    0.8802       0.076649
```

当前正式参考点为：

```text
方案          accuracy  fidelity  posterior KL
无保护          0.6182    1.0000       0.000000
全保护          0.1545    0.1610       2.835290
```

## 结论

只保护 `weight` 时，攻击 accuracy 为 `15.82%`，仅比全保护高 `0.37` 个百分点；fidelity 仅高 `0.60` 个百分点。即使 bias 和全部 BN buffer 仍然暴露，攻击效果也已经接近纯黑盒。这说明当前 ResNet18 的模型能力泄露主要来自 weight。

只保护 `bias` 时，攻击 accuracy 仍有 `57.71%`。与暴露全部状态并执行相同训练过程的 `num_batches_tracked` 对照相比，accuracy 只降低 `3.36` 个百分点，但 fidelity 降低 `11.66` 个百分点。bias 对类别能力的保护有限，对输出行为一致性的影响更明显。

`running_mean`、`running_var` 和 `num_batches_tracked` 三组的 end 指标完全一致。当前 BatchNorm 使用固定 momentum，batch counter 不影响推理或 running statistics 的更新比例，因此 `num_batches_tracked` 组可作为统一 finetune 的近似空操作对照。100 轮训练后，公开初始化的 running mean/variance 已被相同 query 流程覆盖；单独隐藏这些 BN buffer 没有留下可测的额外保护效果。

无保护正式参考点不执行额外训练，而五个类型实验统一执行 100 轮 finetune。因此 BN buffer 三组相对无保护出现的 `0.75` 个百分点 accuracy 差异主要是训练漂移，不能解释为 buffer 本身提供了保护。

作为补充，新实验 04 在每个 k 使用相同初始化轨迹：TensorShield Top-10 保护 `8.9934%` 参数时，攻击 accuracy 为 `25.69%`；Top-12 保护 `24.7531%` 参数时进一步降至 `19.26%`。这说明关键 weight 前缀能够持续抑制攻击，但 Top-10 尚未达到当前全保护下界。

```text
metrics.json                    五种类型的协议、保护统计、best/end 和正式参考指标
history.tsv                    五组共 500 轮训练和评估记录
data.tsv                       三张图使用的 end 原始点
accuracy.png                   保护存储比例与 accuracy
fidelity.png                   保护存储比例与 fidelity
posterior_kl.png               保护存储比例与 posterior KL
<type>_mask.pt                 五种类型各自的紧凑保护掩码
```
