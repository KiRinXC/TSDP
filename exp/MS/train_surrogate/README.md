# surrogate 训练入口

本目录实现 MS baseline 的 surrogate 初始化、伪标签训练和 `eval_ms` 评估。所有运行都读取 victim 的 `best.pth`，并使用 `query_pool_ms` 固定顺序的预算前缀。

内部代码分为两部分：`core/` 保存数据读取、训练、评估和产物写入等公共流程；`defense/` 保存保护策略插件、掩码协议、模型初始化与冻结机制。`train.py` 是普通参数隐藏策略的单组正式入口，`sweep.py` 只负责这些策略的并行调度和续跑；TEESlice 使用 `teeslice/attack.py` 作为独立正式入口。

根目录的统一入口处理普通 victim 参数空间中的权重隐藏策略。TEESlice 会改变 victim 的训练结构和参数边界，因此攻击入口独立放在 `teeslice/`，复用 `core/` 的查询、训练、评估和产物逻辑，但不注册为 122-unit mask 策略。

```text
train_surrogate/
├── train.py              单组正式入口
├── sweep.py              可续跑的并行扫描入口
├── plan.py               固定保护计划与 mask
├── baseline.json         固定配置及保护统计
├── run.sh                命令行封装
├── core/                 公共训练、评估和产物管理
├── defense/              策略注册、mask、unit 映射和初始化
├── selector/             TensorShield 作者确认的固定 rank 列表
└── teeslice/             TEESlice defended victim 的独立 surrogate 攻击
```

## 保护策略

```text
no_protection    所有 victim 权重暴露，作为攻击上界
full_protection  所有 victim 权重不可见，作为 posterior-visible 黑盒下界
shallow          从官方第 1 层开始保护连续完整层
middle           保护不接触首尾的连续完整官方层
deep             连续保护到官方最后一层
custom           直接指定任意连续或离散 unit
large_weight     按公开预训练权重绝对值保护全局大权重标量
tensorshield     按作者确认 rank 构造 Figure 12 固定 tensor mask
```

ResNet18 以 `state_dict` 的稳定顺序定义 122 个基础 tensor unit，索引范围为 `0` 至 `121`。每个 unit 对应一个参数或缓冲区条目；后续所有完整层和细粒度策略均由这些 unit 聚合，不再另外定义底层保护单位。

下表给出 ResNet18 官方 18 层与 unit 的归属关系，用于在需要完整层边界时选择 unit：`conv1` 为第 1 层，8 个 BasicBlock 中的 16 个主分支卷积为第 2 至 17 层，`last_linear` 为第 18 层。BatchNorm 状态归入前置卷积；三个 downsample 分支归入对应 stage 首个 BasicBlock 的 `conv1` 层；`maxpool` 关联第 1 层，`avgpool` 关联第 18 层。池化层没有 `state_dict` 条目，因此不会增加 unit 数量。训练入口只把明确指定的官方层映射为 unit，不会把比例隐式转换成 unit。

```text
官方层  主模块                   unit 索引
1       conv1                    0-5
2       layer1.0.conv1           6-11
3       layer1.0.conv2           12-17
4       layer1.1.conv1           18-23
5       layer1.1.conv2           24-29
6       layer2.0.conv1           30-35, 42-47
7       layer2.0.conv2           36-41
8       layer2.1.conv1           48-53
9       layer2.1.conv2           54-59
10      layer3.0.conv1           60-65, 72-77
11      layer3.0.conv2           66-71
12      layer3.1.conv1           78-83
13      layer3.1.conv2           84-89
14      layer4.0.conv1           90-95, 102-107
15      layer4.0.conv2           96-101
16      layer4.1.conv1           108-113
17      layer4.1.conv2           114-119
18      last_linear              120-121
```

保护方案不接收比例。`shallow`、`middle` 和 `deep` 使用 `--protected-layers` 指定 1-based 官方完整层，例如 `1-3`、`8-11`、`16-18`。入口会把完整层映射为 unit；downsample 随所属官方层一起选择，因此最终 unit 不要求连续。也可以通过 `--protected-units` 提供映射后的精确 unit，但必须恰好组成符合策略方向的完整官方层。任意细粒度连续或离散 unit 使用 `custom`，例如 `3,6,9`。`no_protection` 和 `full_protection` 自动生成空集合与全集。对完整 unit 策略，分类头的 `last_linear.weight` 和 `last_linear.bias` 必须同时保护或同时暴露。

每种策略最终统一生成一个 `protection_mask.pt`。文件按模型 `state_dict` 顺序包含一个 unit mask：`True` 表示对应状态不可见，`False` 表示可被攻击者复制。完整 unit 使用紧凑的 `all` 或 `none` 标记；只保护部分标量时保存压缩位图。该掩码是保护方案的唯一权威表示，JSON 只记录掩码路径、SHA256 和统计量，不重复保存层列表或 unit 索引列表。

TensorShield 论文 Figure 12 将同一批索引 `0` 至 `121` 画成 L1 至 L7 七段，用于展示 Serdab 和 DarkneTZ 的四层选择。L1 至 L7 是该图对网络位置的展示分段，不是额外的 tensor unit，也不是 ResNet18 官方的深度定义；本实验不会用这七段替代官方 18 层。

`large_weight` 采用逐标量选择：通过 `--protected-scalars` 指定绝对标量数量，对 Conv、BatchNorm 和 Linear 的公开预训练 `weight` 绝对值进行全局排序，排名靠前的标量保持公开或随机初始化，其余标量复制 victim。BatchNorm 的 bias 和运行统计量沿用对应 weight 的掩码。分类头同样严格服从逐标量 mask：受保护 weight 保留目标类别 `Linear` 的随机初始化，未受保护 weight 复制 victim；Linear bias 不参与排序，因此完整复制 victim。不会因为分类头存在一个受保护标量而额外丢弃其他可见标量。

TEESlice 原实现先取阈值 `sorted_abs_weight[int(total * ratio)]`，再用严格小于阈值的位置保留 victim、严格大于阈值的位置换成公开预训练权重。等于阈值的位置会同时落在两个 mask 之外，因此原代码在有并列值时不能保证准确保护 `int(total * ratio)` 个标量，端点也可能损失一个阈值位置。当前实现保留相同的全局绝对值排序含义，但使用精确 top-k；整数预算因此严格等于受保护的 eligible weight 标量数。Linear bias 不参与排序，BatchNorm bias 随 weight 联动，所以最终 `protected_param_count` 可以大于 `protected_scalars`。

新增保护策略时，在 `defense/` 中实现 mask builder，并注册到 `DEFENSE_REGISTRY`。builder 只负责返回 victim 状态空间中的布尔掩码、分类头模式和策略统计量；公共初始化器负责组合公开权重与暴露的 victim 权重，训练器不包含具体策略分支。

## TensorShield 固定 Rank

当前 TensorShield baseline 不再运行 importance、Grad-CAM 或公式排序代码。`selector/tensorshield.py` 只保存作者确认的 `ResNet18+CIFAR-100` 41-weight rank、按论文规则得到的 17-weight eligible rank，以及 Figure 12(d) 发布的最终 10-weight 集合。

固定保护集合是作者 eligible rank 的 Top-10；Figure 12 按网络位置展示同一集合，不表示 importance 顺序。由于集合包含 `last_linear.weight`，mask 同步保护 `last_linear.bias`，最终保护 `11/122` 个完整 unit、`1,009,764/11,227,812` 个参数，参数比例为 `8.9934%`，分类头模式为 `replace`。mask 逻辑 SHA256 固定为 `1e3aa38124f084dd39eab42a4d3f1ddf1ca86807812796c66a8318c05e7aa2cb`。

统一训练入口直接根据固定列表构造 mask，不读取评分文件或外部源 mask。运行后仍会在该次 surrogate 权重目录中保存 `protection_mask.pt`，作为本次训练实际保护方案的权威产物：

```bash
bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense tensorshield \
  --budget 500 --training-mode finetune --label-mode soft
```

目前只有 `ResNet18+CIFAR-100` 注册了作者确认 rank。增加其他模型或数据集前，必须先获得并固化对应作者 rank 与最终保护集合；不得复用当前列表，也不得在正式入口中临时恢复公式估计排名。

## 正式攻击协议

当前正式协议为 `posterior_replace_finetune_v2`：按 TEESlice 与 TensorShield 的 adversary 训练入口使用 `lr_step=60`；soft posterior 训练使用与 victim 查询一致的确定性 test transform，不把原图 posterior 绑定到随机裁剪或翻转后的图像。完整协议如下：

```text
query 来源          query_pool_ms 的固定预算前缀
query budget        各数据集 victim 训练集的 1%
攻击者可观测输出    victim 完整 softmax posterior
标签模式            soft posterior
query 输入变换       与 posterior 生成一致的确定性 test transform
暴露权重            从 victim best.pth 复制
受保护 backbone     保留官方 ImageNet 初始化
分类头完整暴露      复制 victim 分类头，head_mode=exposed
分类头部分保护      按 mask 混合随机初始化与 victim 标量，head_mode=mixed
分类头完整保护      使用目标类别随机初始化 Linear，head_mode=replace
训练方式            除无保护恒等上界外，所有参数共同 finetune
正式 checkpoint     固定训练轮数后的 end.pth
诊断 checkpoint     best.pth，仅记录 eval_ms accuracy 最高点，不进入主结果汇总
```

权重保护策略不改变查询接口。`no_protection`、partial protection 和 `full_protection` 均读取 `posteriors.pt`，不得因为保护位置不同而切换 hard/soft。hard label 只用于 `lab/` 中明确标注的输出能力消融，不进入当前正式 baseline。

`no_protection` 是特殊恒等控制组：完整 victim 状态已经暴露，surrogate 在 epoch 0 直接评估并同时写出 `best.pth` 与 `end.pth`，不使用 query 更新参数。其余策略固定训练 100 epoch，使用 batch size 64、SGD、学习率 0.01、momentum 0.5、weight decay `5e-4`，并在第 60 个 epoch 后把学习率乘以 0.1；即 epoch 1-60 使用 `0.01`，epoch 61-100 使用 `0.001`。正式汇总统一读取 `end` 指标，不能从 `eval_ms` 选择 epoch 作为主结果。

## 初始化与训练

未受保护的状态直接从 victim `best.pth` 复制。受保护的 backbone 状态保留官方 ImageNet 初始化。

分类头完整暴露时直接复制 victim 分类头，完整保护时使用直接映射到当前数据集类别数的随机初始化 `Linear`。逐标量策略允许分类头部分暴露，此时严格按 mask 复制所有可见 victim 标量，只对不可见位置保留随机初始化。Lab 02 的替换头结论适用于分类头整体不可见的情况，不能用于丢弃部分暴露的分类头标量。正式实验不再运行 adapter 或 frozen 分支，所有初始化方式仍统一全模型 finetune。

正式入口只接受 `--training-mode finetune` 和 `--label-mode soft`。冻结机制仍作为底层验证代码保留，但不属于当前正式协议，也不得产生正式结果。

所有策略使用相同的 `posteriors.pt`、确定性 query 输入和 soft cross-entropy。soft 模式不使用 RandomCrop、RandomHorizontalFlip 等随机增强；`full_protection` 仅表示所有 victim 权重不可见，不额外叠加 label-only 输出限制。

## 运行方式

运行一个 `ResNet18+C100` 配置：

```bash
bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense shallow \
  --protected-layers 1-8 \
  --plan-id shallow_08 \
  --budget 500 \
  --training-mode finetune \
  --label-mode soft
```

其他 unit 选择方式：

```bash
bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense deep --protected-layers 11-18 --plan-id deep_08 \
  --budget 500 --training-mode finetune --label-mode soft

bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense middle --protected-layers 8-11 --plan-id middle_04 \
  --budget 500 --training-mode finetune --label-mode soft

bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense custom --protected-units 3,6,9 \
  --budget 500 --training-mode finetune --label-mode soft

bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense large_weight --protected-scalars 112229 --plan-id large_01 \
  --budget 500 --training-mode finetune --label-mode soft
```

重新确定 `ResNet18+C100` 上下界使用：

```bash
bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense no_protection --budget 500 \
  --training-mode finetune --label-mode soft

bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense full_protection --budget 500 \
  --training-mode finetune --label-mode soft
```

正式 baseline 使用各数据集 manifest 中训练集 1% 对应的 `max_budget`：`c10=500`、`c100=500`、`s10=50`、`t200=1000`。较小预算仅用于后续 Lab 中的 query budget 敏感性实验。

planned baseline 使用 `plan_id` 作为可读的 `artifact_id` 目录名，例如 `shallow_06` 和 `large_03`；上下界分别使用 `no_protection` 和 `full_protection`。完整配置仍生成稳定的短 `run_id` 并保存在 JSON、checkpoint 和结果索引中，用于核对协议一致性。非计划内的自定义配置继续使用 `run_id` 作为目录名。

## ResNet18+C100 baseline 保护计划

正式保护计划固定使用以下 8 个保护规模：

```text
保护层数  shallow  middle/SOTER  deep
2         1-2      9-10          17-18
4         1-4      8-11          15-18
6         1-6      7-12          13-18
8         1-8      6-13          11-18
10        1-10     5-14           9-18
12        1-12     4-15           7-18
14        1-14     3-16           5-18
16        1-16     2-17           3-18
```

`middle/SOTER` 始终围绕第 9、10 层构成的网络中心对称扩展，且不接触首尾层。三种完整层策略在同一行保护相同数量的官方层，但实际 unit 数量和参数量由层结构决定并单独记录。

`large_weight` 参考 TEESlice `weight_pruner.py` 的正式扫描点，使用 `0.01、0.1、0.3、0.5、0.7、0.8、0.9、0.95` 作为预算来源。计划生成时先统计当前公开 ResNet18 中 Conv、BatchNorm 和 Linear 的可排序 `weight` 总数，再按上述锚点向下取整为 8 个绝对 `protected_scalars`。正式训练命令和 mask 只保存整数预算；比例仅作为复现来源元数据，不作为训练参数。

当前 ResNet18+C100 的可排序标量总数为 `11,222,912`，固定预算如下。`protected_param_count` 还包含随 BatchNorm weight 同步保护的 bias，因此可能略大于 `protected_scalars`：

```text
配置      来源锚点  protected_scalars  protected_param_count  head_mode
large_01  0.01          112229              116896             exposed
large_02  0.1          1122291             1127072             mixed
large_03  0.3          3366873             3371664             mixed
large_04  0.5          5611456             5616247             mixed
large_05  0.7          7856038             7860829             mixed
large_06  0.8          8978329             8983121             mixed
large_07  0.9         10100620            10105412             mixed
large_08  0.95        10661766            10666558             mixed
```

计划入口会为 24 个完整层配置和 8 个大权重配置计算保护 unit 数、保护参数量、分类头模式及 mask SHA256，并写出结构化清单和紧凑 mask 包：

```bash
python3 exp/MS/train_surrogate/plan.py
```

```text
exp/MS/train_surrogate/baseline.json                32 组固定配置及保护统计
weights/MS/surrogate/resnet18/c100/baseline.pt     32 组紧凑保护 mask
```

后续训练必须按 `baseline.json` 的参数运行，并验证实际 `protection_mask_sha256` 与清单一致。不得临时修改层范围、标量预算或跳过 mask 校验。

浅层、中间层和深层的 24 组正式训练使用可续跑批量入口：

```bash
python3 exp/MS/train_surrogate/sweep.py layers --jobs 4
```

全局大权重标量保护的 8 组正式训练使用：

```bash
python3 exp/MS/train_surrogate/sweep.py large_weight --jobs 4
```

该入口只把清单中的整数 `protected_scalars` 传给 runner，并将 `source_ratio` 作为结果元数据保存。`large_01` 的分类头完整暴露；`large_02` 至 `large_08` 的分类头部分暴露，使用 `mixed` 模式按 mask 组合随机初始化标量与 victim 标量。

`--jobs` 只表示并行训练进程数，不改变任何单组实验参数；当前硬件使用 4。每个 worker 只领取一个 `plan_id` 并写入独立 run 目录，中央 `metrics.tsv` 使用进程锁和原子替换更新。正式 runner 会在创建结果目录前校验策略、层范围、保护 unit 数、保护参数量、分类头模式和 mask SHA256。已经存在完整 `metrics.json` 的 `plan_id` 会被跳过，因此中断后可以使用同一命令继续。

每个运行都会在保护摘要和 `results/MS/resnet18/c100/metrics.tsv` 中保存 `protected_unit_count`、`protected_param_count` 与 `protected_param_ratio`。层数用于表示原策略的保护位置和范围，参数保护比例用于后续横向观察不同策略的实际保护成本。

## 输出

```text
weights/MS/surrogate/<model>/<dataset>/<artifact_id>/
  best.pth                surrogate_acc 最高的 checkpoint
  end.pth                 最后一个 epoch 的 checkpoint
  protection_mask.pt      当前策略的 unit 保护掩码
  params.json             输入、训练参数和掩码摘要
  train.log.tsv           每个 epoch 的训练观测与 eval_ms 原始指标

results/MS/<model>/<dataset>/
  metrics.tsv             同一模型和数据集下所有 run 的原始指标索引
  <artifact_id>/metrics.json
                          best 与 end 的完整原始指标
```

`best.pth` 按逐 epoch 的 `surrogate_acc` 保存，仅用于诊断训练过程。`metrics.json` 同时保留 best 和 end，正式表格、绘图和策略比较只读取固定训练终点 `end`。无保护恒等上界的 best 和 end 均为 epoch 0。

正式保存的 MS 指标为：

```text
eval_count
victim_correct
surrogate_correct
agreement_count
victim_acc
surrogate_acc
fidelity
posterior_kl_sum
posterior_kl
```

准确率下降、fidelity 下降、相对黑盒倍数和归一化保护效果均不写入正式结果，由后续绘图脚本从这些原始值计算。
