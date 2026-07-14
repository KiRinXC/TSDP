# TEESlice 查询攻击

本目录使用统一 MS query 协议攻击独立复现的 TEESlice pruned victim。TEESlice 会改变 victim 结构和训练方式，因此结果只写入其独立结果目录，不加入固定普通 victim 的主 baseline `metrics.tsv`。

## 攻击者能力

正式实验同时记录黑盒和白盒两种能力。黑盒采用 `blackbox_known_pruned_topology`：攻击者知道最终剪枝模型的完整连接关系和保护策略，可从 `keep_flags` 重建同构的 pruned TEESlice topology，并可获得公开 ImageNet backbone 权重，但看不到任何训练后的任务相关状态。

```text
surrogate 结构      与最终 victim 相同的 pruned TEESlice topology
公开连接信息        只复制 keep_flags，不复制其对应参数值
公开主干初始化      官方 ImageNet ResNet18；conv1 中心裁剪为 3x3
私有状态初始化      fresh 默认初始化：proxy 与分类头随机，alpha 使用公开默认初值
任务 BN 状态        不复制 victim buffer，从公开权重或模块默认值开始更新
可观测输出          500 条 soft posterior
不可观测            source、teacher、训练后的 proxy、alpha、分类头和任务 BN 状态
训练方式            surrogate 全部可执行路径参数共同 finetune
```

蒸馏后的 Teacher checkpoint 只属于模型所有者的离线构造过程，攻击者不得读取。拓扑只描述哪些 main/proxy 路径存在，不包含这些路径训练后的参数值。fresh surrogate 构造完成后只设置相同 `keep_flags`，并显式解除公开 backbone 原有的冻结标记；未激活路径不会参与前向计算，其余参数全部由 query 数据微调。

白盒采用 `whitebox_full_state`：重新构造同一模型并加载最终 defended victim 的完整 `state_dict` 与 `keep_flags`，随后实际执行一次 `eval_ms` 评估。白盒不进行 surrogate 训练；其 accuracy、fidelity 和 posterior KL 由评估代码计算，而不是手工填入解析上界。

## 前置产物

```text
weights/MS/victim/teeslice_r18/c100/best.pth
dataset/MS/c100/teeslice_r18/manifest.json
dataset/MS/c100/teeslice_r18/posteriors.pt
weights/pre_train/resnet18-5c106cde.pth
```

查询产物由以下命令生成：

```bash
python3 exp/MS/transfer/get_label.py teeslice_r18 c100
```

## 固定协议

```text
victim              TEESlice pruned victim best.pth
black-box visibility blackbox_known_pruned_topology
white-box visibility whitebox_full_state
surrogate topology  从 victim keep_flags 复制连接关系
surrogate 参数       公开 backbone + fresh 私有状态，全部 finetune
query 来源          query_pool_ms 的固定预算前缀
query budget        500，即 victim_train 的 1%
标签模式            soft posterior
query transform     确定性的 test transform
训练轮数            100
优化器              SGD，lr=0.01，momentum=0.5，weight_decay=5e-4
学习率调度          StepLR，step_size=60，gamma=0.1
主要评估点          第 100 轮 end.pth
原始指标            accuracy、fidelity、posterior KL 及其计数
随机种子            42
```

fidelity 和 posterior KL 始终相对于当前 TEESlice pruned victim 计算。`eval_ms` 不参与 surrogate checkpoint 选择；`best.pth` 只作为逐轮诊断，正式结果读取固定训练终点 `end.pth`。

## 运行方式

```bash
python3 exp/MS/train_surrogate/teeslice/attack.py resnet18 c100 \
  --budget 500 --training-mode finetune --label-mode soft
```

只验证输入、初始化和结果协议：

```bash
python3 exp/MS/train_surrogate/teeslice/attack.py resnet18 c100 \
  --budget 500 --training-mode finetune --label-mode soft --dry-run
```

覆盖当前同语义结果时显式增加 `--overwrite`。

## 输出

```text
weights/MS/surrogate/resnet18/c100/teeslice/
├── best.pth
├── end.pth
├── params.json
├── topology.json
└── train.log.tsv

results/MS/resnet18/c100/teeslice/metrics.json
```

`topology.json` 保存黑盒攻击使用的公开 `keep_flags`、活跃 proxy 数和拓扑摘要。`metrics.json` 同时保存 `blackbox_known_pruned_topology` 的训练结果与 `whitebox_full_state` 的实际评估结果，但不写入主 baseline 汇总索引。
