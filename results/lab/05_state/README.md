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

## 实验结论

本实验说明，`state_dict` 的保护价值不能由张量类型或参数量单独判断：

- 只保护主路径 Conv 即使占全部参数的 `97.84%`，仍明显弱于保护全部 weight；
  Stem、downsample、BN affine 和分类头 weight 不能从后续建模中直接排除。
- 相同参数量下，BN gamma 的保护信号远强于 BN beta；全部 BN affine 又强于单独
  gamma，说明二者存在补充，但 gamma 是更优先的候选。
- 分类头的作用几乎全部来自 weight，单独保护 bias 接近无效。
- `running_mean` 与 `running_var` 的结果逐值相同，BN buffer 更适合作为执行闭包
  状态，而不是独立的长期保护来源。

## 文件

```text
metrics.json                    十八组保护统计、validation-best 与最终原始指标
history.tsv                     十八组各 100 轮 train/validation 日志
data.tsv                        三张图使用的十八个单次 eval_ms 原始点
accuracy/fidelity/posterior_kl.png
<group>_mask.pt                 十八组紧凑保护掩码
```
