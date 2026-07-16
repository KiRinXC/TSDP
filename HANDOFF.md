# TSDP 项目交接

本文面向一个完全没有历史上下文的新会话。开始工作前必须先阅读本文件、`AGENTS.md`、`README.md`、`STRUCTURE.md` 和 `FLOW.md`。如本文与实际代码或正式实验最近一层的 `README.md` 不一致，以代码和最近一层已经固化的正式协议为准，并先查明差异，不能直接运行实验。

## 1. 我们在做什么

TSDP 是一个围绕 TEE 场景下模型参数保护的研究项目。当前只推进 Model Stealing（MS）部分，Membership Inference Attack（MIA）和真实 TEE 部署暂不展开。

研究目标不是发明新的模型窃取算法，而是在统一、较强的攻击协议下比较不同参数保护粒度对 MS 的抑制效果。最终创新方向是把保护粒度从已有工作的完整层、完整 tensor 或离散标量进一步细化到通道块，并提取适合保护的关键路径。

当前阶段的具体任务是：

1. 固定统一 MS 数据、查询、初始化、训练和评估协议。
2. 在 `ResNet18+CIFAR-100` 上完成已有保护策略 baseline。
3. 以该组合为验证基准，下一步实现本项目的通道块保护策略。
4. 确认通道块策略有效后，再扩展到四个数据集和四个模型。

## 2. 当前有效的统一 MS 协议

### 2.1 数据划分

当前采用与公开工作一致的随机重叠思路，不再使用早期讨论过的 `strict_disjoint`：

```text
victim_train   官方训练集全量，用于训练 victim
query_pool_ms  从同一官方训练集随机无放回抽取 1%
eval_ms        官方 test 或 validation split；普通 victim 逐轮评估，surrogate 用于攻击评估
```

query 因而可能与 victim 训练数据重叠，这是当前论文 baseline 协议的有意选择，不是数据泄漏 bug。普通 victim 不另划 validation split，而是与 TensorShield、TEESlice 两个参考仓库保持一致，在 `eval_ms` 上逐轮评估并保存 accuracy 最高的 `best.pth`；该 checkpoint 用于生成 query 输出。这个约定只适用于 victim 固定，不能用于选择 surrogate 正式结果。随机种子统一为 `42`。

四个数据集的正式 query budget：

```text
c10    500
c100   500
s10     50
t200  1000
```

数据协议只允许通过 `dataset/MS/<dataset>/manifest.json` 的 `query.split=query_pool_ms` 引用 `splits.tsv`。不得重新建立 `query.tsv`，不得恢复 `dataset/query/` 或 `dataset/auxiliary/`。

### 2.2 victim 与查询输出

- 普通 victim 使用 `weights/MS/victim/<model>/<dataset>/best.pth` 生成查询输出。
- 权重文件统一命名为 `best.pth` 和 `end.pth`，不要恢复误导性的 `target.pth`。
- 正式攻击者可以观察完整 softmax posterior。
- 正式标签模式统一为 `soft`；hard label 只属于明确标注的 Lab 输出能力消融。
- posterior 生成和 surrogate 读取 query 时都必须使用确定性的 test transform。
- soft posterior 绝对不能与 `RandomCrop`、随机翻转等训练增强后的图像绑定，否则标签与输入错配。

### 2.3 surrogate 初始化与训练

普通固定 victim baseline 的规则：

```text
暴露状态          从 victim best.pth 精确复制
受保护 backbone   保留官方 ImageNet 初始化
分类头完整暴露    复制 victim 分类头，head_mode=exposed
分类头部分保护    按标量 mask 混合复制，head_mode=mixed
分类头完整保护    随机替换为目标类别 Linear，head_mode=replace
训练方式          除无保护恒等组外，全部参数共同 finetune
epochs            100
batch size         64
optimizer          SGD(lr=0.01, momentum=0.5, weight_decay=5e-4)
scheduler          StepLR(step_size=60, gamma=0.1)
```

100 轮是当前统一的强攻击协议。TensorShield 原仓库正式扫描本身使用 100 轮；TEESlice 部分旧脚本虽然只训练 10 轮，但现有日志表明 10/20 轮会明显低估攻击者。不要只把 epoch 改成 50：当前 `lr_step=60` 与 100 轮配套，单独截断会使学习率从不衰减，并令现有全部正式结果失效。

### 2.4 checkpoint 与评估

- victim query 固定读取按上述规则产生的 `best.pth`；普通 victim 的 `end.pth` 只保存训练终点。
- surrogate 的 `weights/MS/surrogate/.../best.pth` 和 `end.pth` 都是代理模型权重，不是 victim。
- 正式主结果必须读取固定训练终点 `end.pth`。
- `best.pth` 按逐轮 `eval_ms` accuracy 保存，仅用于诊断，不能用于论文主结果或策略选择。
- 使用 `best.pth` 作为正式结果会利用 test/eval 数据选模，产生 test leakage。
- `no_protection` 是 epoch 0 恒等控制组，其 `best.pth` 与 `end.pth` 相同。
- 正式保存原始 `accuracy`、`fidelity`、`posterior KL` 及计数。下降值、倍数和归一化指标只在绘图时派生。

## 3. 基础保护单位

`ResNet18` 以 `state_dict` 稳定顺序定义 122 个基础 tensor unit，索引为 `0..121`。每个 unit 对应一个参数或 buffer 条目。所有完整层、TensorShield 和未来通道块方案都应能回溯到这一权威状态空间，不能再发明相互冲突的底层编号。

完整层 baseline 使用官方 18 层：stem conv 为第 1 层，8 个 BasicBlock 的 16 个主卷积为第 2 至 17 层，分类头为第 18 层。BN、downsample 和池化归属详见 `exp/MS/train_surrogate/README.md`。

TensorShield 图中的 L1-L7 只是论文展示分段，不是新的网络深度，也不是第三级映射。不要用 L1-L7 替换 122 unit 或官方 18 层。

## 4. 已经完成的工作

### 4.1 数据、victim 与查询产物

- 四个数据集的 MS manifest 和 split 已建立在 `dataset/MS/`。
- 普通 victim 训练、`best/end` checkpoint 和查询标签流程已经实现。
- `get_label.py` 支持普通 victim 和 `teeslice_r18`。
- posterior、labels 和 victim checkpoint SHA256 在 manifest 中绑定。

### 4.2 普通 ResNet18+C100 上下界

正式主结果：

| 策略 | epoch | surrogate accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|
| no protection | 0 | 0.6182 | 1.0000 | 1.1305e-9 |
| full protection | 100 | 0.1545 | 0.1610 | 2.835290 |

当前 `full_protection` 表示所有 victim 参数不可见，但查询接口仍暴露 posterior。不要把它悄悄改回 label-only 黑盒；hard-label 全保护只在 Lab 中出现过。

### 4.3 三种完整层 baseline

以下策略均已完成 8 个保护规模，共 24 组正式结果：

- `shallow_02..16`：从输入侧连续保护 2、4、...、16 个官方层。
- `middle_02..16`：围绕网络中部连续、对称扩展，对应 SOTER 思路。
- `deep_02..16`：连续保护到官方第 18 层。

三者按相同层数扫描，但实际参数比例差异很大。这是原有完整层策略的真实属性，不要为了“看起来公平”篡改保护集合。绘图和与本方法比较时必须同时使用实际 `protected_param_ratio` 或 FLOPs 成本。

### 4.4 全局大权重标量 baseline

`large_01..08` 已按 `0.01、0.1、0.3、0.5、0.7、0.8、0.9、0.95` 来源锚点转换为绝对整数预算并全部运行。

该策略按所有 eligible weight 标量的绝对值全局排序，不是比较整个卷积 tensor 的绝对值和。mask 可以在同一 tensor 内逐标量离散保护。

分类头出现部分标量保护时必须使用 `mixed`：未保护标量复制 victim，保护标量保留随机初始化。绝对不能因为分类头只保护了一个标量，就丢弃整个分类头的其他暴露参数。这个错误曾导致保护成本与实际不可见参数不一致，已经修复并重跑。

### 4.5 仅分类头保护控制组

`head_only` 已在普通 `ResNet18+C100` victim 上完成。该策略只隐藏 `last_linear.weight` 和 `last_linear.bias`，完整复制其余 backbone 状态；保护 `2/122` 个 unit、`51,300/11,227,812` 个参数，参数比例为 `0.4569%`，分类头使用 `replace`，随后对全部参数共同 finetune。

正式 `end.pth` 结果：

```text
run id                  a669fb8f80e8
epoch                   100
accuracy                0.4404
fidelity                0.5135
posterior KL            1.347578
```

它比参数比例相近的 `shallow_02` 更能抑制 MS，但仍明显没有达到普通 victim 的 `full_protection` 参考结果。该实验用于量化分类头不可见的独立贡献，不重复 Lab 02 的 replace/adapter 选择实验。

### 4.6 TensorShield baseline

当前实现只使用论文作者直接提供的 `ResNet18+CIFAR-100` 最终 rank，不再保留 importance、Grad-CAM 或公式估计代码。

权威文件：

```text
exp/MS/train_surrogate/selector/tensorshield.py
exp/MS/train_surrogate/defense/tensorshield.py
```

正式 mask 是作者 eligible rank 的 Top-10 weight 集合，并联动保护 `last_linear.bias`：

```text
protected unit          11/122
protected params        1,009,764 / 11,227,812
protected ratio         8.9934%
mask SHA256             1e3aa38124f084dd39eab42a4d3f1ddf1ca86807812796c66a8318c05e7aa2cb
head mode               replace
```

正式 `end.pth` 结果：

```text
run id                  81fce87f83b4
epoch                   100
accuracy                0.1913
fidelity                0.2099
posterior KL            2.505831
```

只有 `ResNet18+C100` 获得了作者 rank。不得把该 rank 复用到其他数据集或模型，也不得重新启用公式估计冒充作者结果。没有对应作者 rank 的组合应标记为不适用。

### 4.7 TEESlice standalone 复现

TEESlice 改变 victim 结构和训练过程，不能写入普通 ResNet18 固定 victim 的主 `metrics.tsv`，也不能与普通 mask 策略假装是完全同条件点。

当前 `ResNet18+C100` 已实现四阶段 defended victim：

```text
source   在私有训练数据上监督训练 CIFAR-stem ResNet18
teacher  使用 source posterior 蒸馏同结构 teacher
full     冻结公开 backbone，训练 private proxy、alpha、分类头并适配 BN
prune    只在 victim_train 内部验证集动态删除低 alpha proxy
```

最终 victim：

```text
checkpoint              weights/MS/victim/teeslice_r18/c100/best.pth
eval_ms accuracy         0.7578
active proxy             8
private params           703,092
private param ratio      5.9223%
private FLOPs            27,756,644
```

剪枝判断只能使用 victim_train 内部验证集，不能用 `eval_ms` 调节拓扑。

TEESlice 黑盒攻击者能力已经固定为：知道最终 pruned topology 和保护策略，只复制 `keep_flags` 连接关系与官方 ImageNet backbone；private proxy、alpha、分类头和任务 BN 状态 fresh 初始化，然后全部可执行路径参数 finetune。完整状态白盒重新加载 victim 全部状态并实际评估，不再手工填解析上界。

正式结果：

| 能力 | 主评估点 | accuracy | fidelity | posterior KL |
|---|---|---:|---:|---:|
| blackbox known topology | surrogate end epoch 100 | 0.1619 | 0.1784 | 3.251131 |
| whitebox full state | victim 实际评估 | 0.7578 | 1.0000 | 4.8149e-10 |

TEESlice 结果只在 `results/MS/resnet18/c100/teeslice/` 保存，并标记为 `standalone_reproduction`。不要覆盖普通预训练模型的 `no_protection` 或 `full_protection`。

### 4.8 Lab 结论

有效 Lab 必须保留：

- `lab/01_kmeans`：预训练特征聚类验证。
- `lab/02_head`：分类头 replace/adapter 与 frozen/finetune 消融；结论支持参数不可见时使用 replace，并统一对 surrogate 执行 finetune。暴露分类头仍按正式可见性规则复制。
- `lab/03_baseline`：四种通用 baseline 曲线、分类头与 TensorShield 单点、TEESlice standalone 单点及普通 victim no/full 参考线的三指标总图。
- `lab/04_tensorshield`：作者 rank 的前缀、冗余和窗口消融。
- `lab/05_state`：不同 state 类型保护的 MS 对比。

Lab 结果不能混入正式主实验索引，但也不能以“清理历史”为由删除仍承担独立结论的 Lab。

## 5. 当前卡在哪里

当前没有代码运行阻塞，`ResNet18+C100` 的 baseline 组合已经完成。真正未决的是本项目方法的设计与扩展边界：

1. 尚未实现通道块保护策略及其关键路径选择算法。
2. 尚未确定通道块大小、跨层成本归一化、保护预算扫描点和相同成本比较方式。
3. `ResNet50`、`VGG16-BN`、`MobileNetV2` 尚未定义各自的基础 unit 与官方层映射。
4. TensorShield 缺少其他模型/数据集的作者 rank，不能自动扩展。
5. TEESlice 当前只实现 `ResNet18+C100`，其他组合需要单独适配 topology、训练和剪枝。

这些是研究和实现工作，不是通过复制当前 ResNet18 mask 就能解决的问题。

## 6. 下一步计划

### 第一阶段：在 ResNet18+C100 实现本项目方法

1. 先定义通道块保护的唯一权威 mask 表示，必须能够回溯到 122 个基础 state unit。
2. 明确定义卷积、BN、downsample 和分类头在通道块级别的联动规则。
3. 明确保护成本：至少记录受保护参数数、参数比例，并为后续 TEE 阶段预留 FLOPs/通信成本。
4. 固化若干绝对保护预算，不把比例作为训练入口的隐式参数。
5. 在正式运行前先更新最近一层中文 `README.md`，核对默认值、命令和协议完全一致。
6. 使用当前 500-query、soft posterior、100-epoch、end checkpoint 协议运行。
7. 与浅层、中间层、深层、大权重、仅分类头和 TensorShield 在相近实际成本处比较。
8. 随机通道块、排序规则和块大小变化属于本方法消融，放在 Lab 或独立消融中，不要混成公开 baseline。

### 第二阶段：判断是否值得扩展

只有当通道块策略在 `ResNet18+C100` 上同时表现出更低 surrogate accuracy/fidelity、更高 KL，并且 victim utility 与保护成本可接受时，才进入 4x4 扩展。

### 第三阶段：扩展数据集和模型

1. 同一 ResNet18 扩展到 `c10/s10/t200` 时可复用数据与训练协议，但必须重新训练 victim、生成 posterior 和运行结果。
2. 新模型必须先建立自己的 state unit 与完整层映射，不能复用 ResNet18 的 122-unit mask。
3. `no/full/large_weight` 可优先扩展；完整层 baseline 在完成模型映射后扩展。
4. TensorShield 只在取得对应作者 rank 的组合展示。
5. TEESlice 只在完成结构忠实适配和独立上下界后展示，并始终标为 standalone。

### 最终绘图

当前三指标总图已经展示：

```text
普通预训练模型 no/full bounds
shallow / middle / deep / large_weight
head_only
TensorShield
TEESlice standalone point
```

后续把本项目通道块方法加入同一图。TEESlice 必须继续使用不同标记并注明 victim 结构不同，不能与普通 victim 曲线连成同一条成本曲线。

## 7. 绝对不要再踩的坑

1. **不要使用 `eval_ms` 选择正式 surrogate checkpoint。** 主结果只能是固定 `end.pth`；`best.pth` 仅诊断。
2. **不要把 surrogate 和 victim 的 best/end 混淆。** query 用 victim `best.pth`，正式攻击结果用 surrogate `end.pth`。
3. **不要恢复 `target.pth`。** 权重统一只有 `best.pth` 和 `end.pth`。
4. **不要让 soft posterior 对应随机增强后的图像。** soft 模式只能使用确定性 test transform。
5. **不要为不同保护位置随意切换 hard/soft。** 当前正式普通 baseline 全部 posterior-visible soft；hard 只属于 Lab。
6. **不要按测试集最优结果选择 frozen/finetune 或 replace/adapter。** 当前正式协议已固定为全模型 finetune；分类头按可见性使用 exposed、mixed 或 replace，不再逐实验选择。
7. **不要在分类头部分保护时丢弃整个头。** 必须逐标量混合复制，否则保护成本被低估。
8. **不要重新计算 TensorShield rank。** 只使用作者给出的 `ResNet18+C100` 固定列表；不要恢复已删除的 importance 公式代码。
9. **不要把 TensorShield rank 迁移到其他组合。** rank 与模型、数据集和 victim 有关。
10. **不要把 L1-L7 当成 ResNet18 官方层或新 unit。** 基础是 122 state unit，完整层是官方 18 层。
11. **不要把完整层 baseline 描述成通道块保护。** shallow/middle/deep 选择的是完整层。
12. **不要把大权重理解成 tensor 范数。** 它是全局逐标量绝对值排序。
13. **不要把 TEESlice 加回普通 victim `metrics.tsv`。** 它改变 victim 结构，只能 standalone。
14. **不要用 `eval_ms` 调整 TEESlice 剪枝拓扑。** 动态剪枝只看 victim_train 内部验证集。
15. **不要把 TEESlice 私有状态复制给黑盒攻击者。** 黑盒只知道 topology 和公开 backbone；完整状态只属于白盒。
16. **不要只改训练轮数而保留不匹配的 scheduler。** 修改正式协议意味着旧权重、结果和索引全部失效，必须清理并重跑。
17. **不要保留 v1/v2/old/backup/日期目录。** 同语义实现和结果直接覆盖或删除，历史交给 Git。
18. **不要保留失效脚本、结果、缓存或兼容入口。** 但仍承担结论的 Lab 不能误删。
19. **不要新增冗长目录名。** 遵守 `AGENTS.md` 的目录层级和单目录最多一个下划线约束。
20. **不要新增或修改实验后漏掉文档。** `README.md` 必须中文；涉及实验结构或流转时同步 `STRUCTURE.md` 与 `FLOW.md`。
21. **不要先跑正式实验再补协议。** 必须先固化 README，再核对代码默认值和命令，最后运行。
22. **不要创建独立 query 文件或 ratio/seed/run 子目录。** 元数据写入 manifest/params，目录保持稳定。

## 8. 关键路径与命令

### 文档

```text
AGENTS.md
README.md
STRUCTURE.md
FLOW.md
exp/MS/train_surrogate/README.md
exp/MS/train_surrogate/teeslice/README.md
results/MS/README.md
```

### 数据与权重

```text
dataset/public/                              原始公开数据
dataset/MS/<dataset>/                        MS split manifest
dataset/MS/c100/resnet18/                    普通 victim query 输出
dataset/MS/c100/teeslice_r18/                TEESlice query 输出
weights/pre_train/                           官方 ImageNet 权重
weights/MS/victim/                           victim checkpoint
weights/MS/surrogate/                        surrogate checkpoint 与 mask
results/MS/resnet18/c100/metrics.tsv         普通固定 victim 主索引
results/MS/resnet18/c100/teeslice/            TEESlice standalone 结果
```

### 验证

```bash
make env
make gpu
make unit
make verify
```

TSDP 只认 `~/venvs/dl-py310-torch210-cu121`。`requirements.txt` 保存直接依赖，`requirements.lock.txt` 保存完整解析版本；不得退回系统 `/usr/bin/python3`。`make gpu` 会严格核对 WSL GPU 桥接并运行真实 CUDA 前向和反向计算，正式实验前必须通过。当前 `make unit` 应通过 33 个测试。运行后清理任何意外生成的 `__pycache__` 和 `*.pyc`。

### 普通 baseline

```bash
bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense head_only \
  --budget 500 --training-mode finetune --label-mode soft

bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense tensorshield \
  --budget 500 --training-mode finetune --label-mode soft

"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_surrogate/sweep.py layers --jobs 4
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_surrogate/sweep.py large_weight --jobs 4
```

### TEESlice

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_victim/teeslice/train.py resnet18 c100
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/transfer/get_label.py teeslice_r18 c100
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_surrogate/teeslice/attack.py resnet18 c100 \
  --budget 500 --training-mode finetune --label-mode soft
```

## 9. 外部参考仓库

当前服务器上参考代码位于 TSDP 同级目录：

```text
../Demo/TEESlice-artifact/
../Demo/TensorShield/
```

这些目录不是 TSDP Git 仓库的一部分。迁移到新机器后如需重新核对原实现，应单独复制或重新获取。TSDP 正式实验不能在运行时依赖 Demo 目录；作者固定 rank 已固化在 TSDP 内部。
