# 实验流程

```text
MS 随机重叠协议/
├── dataset/public/                  读取四个公开数据集
├── exp/MS/transfer/prepare_splits.py
│   └── dataset/MS/<dataset>/         写入 manifest.json 与 splits.tsv
│       ├── victim_train              官方训练集全量，用于训练 victim
│       ├── query_pool_ms             同源随机无放回的 1% 子集，预算使用前缀
│       └── eval_ms                   官方 test 或 val 评估集
├── exp/MS/train_victim/<model>/
│   ├── 读取 victim_train
│   └── weights/MS/victim/<model>/<dataset>/
│       ├── best.pth                  eval_ms accuracy 最高的 checkpoint，也是 query 默认权重
│       ├── end.pth                   最末 epoch 模型
│       ├── train.log.tsv             训练和评估日志
│       └── params.json               可复现实验参数
├── exp/MS/transfer/get_label.py
│   ├── 读取 victim best.pth
│   └── dataset/MS/<dataset>/<model>/
│       ├── labels.tsv                hard pseudo label 与 confidence
│       └── posteriors.pt             test transform 下固定顺序的 posterior
└── exp/MS/train_surrogate/
    ├── 读取 query_pool_ms 的预算前缀与 victim posterior/hard label
    ├── 将预算内 query 按 seed 42、offset 100 固定拆为 80% train 与 20% validation
    ├── soft_query_validation_best_v1
    │   ├── 部分保护与 full_protection 均读取 soft posterior
    │   └── full_protection 是 soft-posterior 正式黑盒边界
    ├── hard_query_validation_best_v1
    │   └── <artifact_id=hard_blackbox> 完整保护并读取 hard label，是第二条正式黑盒边界
    ├── 两种协议均按 validation cross-entropy 选最早 best，eval_ms 只最终评估一次
    ├── plan.py
    │   ├── exp/MS/train_surrogate/baseline.json    固化 32 组 baseline 配置与保护统计
    │   └── weights/MS/surrogate/resnet18/c100/baseline.pt
    │                                           保存对应的紧凑保护 mask
    ├── sweep.py                       按 plan_id 并行训练完整层及全局标量 baseline
    ├── defense/                      注册策略、生成掩码并组合公开与 victim 权重
    │   └── head_only                 仅保护 last_linear.weight/bias，复制全部 backbone
    ├── core/                         统一读取数据、训练、评估和写入产物
    ├── weights/MS/surrogate/<model>/<dataset>/<artifact_id>/
    │   ├── best.pth                  query validation loss 最低的正式 checkpoint
    │   ├── protection_mask.pt        策略的唯一 unit 保护方案
    │   ├── params.json               训练参数与掩码摘要
    │   └── train.log.tsv             每轮 query train 与 validation 指标
    └── results/MS/<model>/<dataset>/
        ├── metrics.tsv               各 run 的选模信息与最终原始指标索引
        └── <artifact_id>/metrics.json
                                             validation 选模与一次 eval_ms 原始指标

TEESlice 独立复现/
├── exp/MS/train_victim/teeslice/train.py
│   ├── 读取 victim_train、eval_ms 与官方 ImageNet 权重
│   ├── 在 victim_train 内临时划分 90% 训练部分与 10% 内部验证部分
│   ├── source 阶段                  监督训练 CIFAR-stem ResNet18 source victim
│   ├── teacher 阶段                 将 source posterior 蒸馏到同结构 teacher
│   ├── full 阶段                    冻结公开参数，训练 private slice、alpha、分类头并适配 BN 状态
│   ├── prune 阶段                   只用内部验证集迭代删除低 alpha proxy
│   ├── weights/MS/victim/teeslice_r18/c100/
│   │   ├── source/{best,end}.pth    source 阶段 checkpoint
│   │   ├── teacher/{best,end}.pth   teacher 阶段 checkpoint
│   │   ├── full/{best,end}.pth      未剪枝 TEESlice checkpoint
│   │   ├── best.pth                 最终 defended victim，供 query 使用
│   │   ├── end.pth                  prune 阶段固定训练终点
│   │   ├── params.json              训练、剪枝与输入追踪信息
│   │   └── train.log.tsv            四阶段逐 epoch 记录
│   └── results/MS/resnet18/c100/teeslice/victim.json
│                                       四阶段效用、剪枝容忍判断与保护成本
├── exp/MS/transfer/get_label.py teeslice_r18 c100
│   ├── 读取通用 query_pool_ms 与 TEESlice best.pth
│   └── dataset/MS/c100/teeslice_r18/
│       ├── labels.tsv               defended victim hard label 与 confidence
│       └── posteriors.pt            确定性 test transform 下的 soft posterior
└── exp/MS/train_surrogate/teeslice/attack.py
    ├── blackbox_known_pruned_topology
    │   ├── 复制最终 keep_flags 连接关系与官方 ImageNet backbone
    │   └── fresh 初始化 proxy、alpha、分类头及任务 BN 状态
    ├── 不读取 source、teacher 或训练后的私有状态
    ├── 使用 400 条 soft query 训练、100 条 query validation 选最早 best
    ├── whitebox_full_state           重新加载最终状态并实际执行 eval_ms
    ├── weights/MS/surrogate/resnet18/c100/teeslice/
    │   ├── best.pth                  validation-best 黑盒 surrogate checkpoint
    │   └── topology.json             公开 keep_flags 与拓扑摘要
    └── results/MS/resnet18/c100/teeslice/metrics.json
                                            黑盒/白盒原始指标，不写入主 metrics.tsv

TensorShield 固定 rank baseline/
├── exp/MS/train_surrogate/selector/tensorshield.py
│   ├── 保存作者确认的 ResNet18+CIFAR-100 41-weight rank
│   ├── 保存按论文规则得到的 17-weight eligible rank
│   └── 固定 Figure 12(d) 的 10-weight 集合与分类头 bias
├── exp/MS/train_surrogate/defense/tensorshield.py
│   └── 直接生成 11/122 unit 的固定保护 mask
└── exp/MS/train_surrogate/train.py --defense tensorshield
    ├── 读取通用 500 条 soft posterior，并固定拆为 400 train / 100 validation
    ├── 全参数最多微调 100 epoch，按 validation soft cross-entropy 选择 best
    ├── checkpoint 固定后只在 eval_ms 上评估一次
    ├── weights/MS/surrogate/resnet18/c100/tensorshield/protection_mask.pt
    └── results/MS/resnet18/c100/tensorshield/metrics.json

Lab 验证实验/
├── lab/01_kmeans/
│   ├── 读取 dataset/public/c100 的 CIFAR-100 test split 与 ImageNet 预训练 ResNet18
│   └── results/lab/01_kmeans/
│       ├── metrics.json              100 类 KMeans、Hungarian 匹配和 NMI
│       └── confusion_matrix_*.png    聚类混淆矩阵
├── lab/02_head/
│   ├── 读取 500 条 soft posterior，并固定拆为 400 train / 100 validation
│   ├── 全保护                        不读取 victim 权重，比较四种攻击配置
│   ├── 随机保护 61/122 unit          分类头固定保护，复制其余暴露 victim 权重
│   ├── 训练 replace/adapter 与 frozen/finetune 共八组配置
│   ├── top10.py                      固定 TensorShield Top-10，比较 public/victim 两侧冻结方向与 joint finetune
│   └── results/lab/02_head/
│       ├── metrics.json              八组 validation-best 与单次 eval_ms 原始指标
│       ├── history.tsv               八组共 800 条 train/validation 记录
│       └── top10_trainability.*      三种 seed-42 Top-10 trainability 结果、历史与图
├── lab/03_baseline/
│   ├── 读取 results/MS/resnet18/c100/<artifact_id>/metrics.json
│   ├── 同时读取 full_protection soft 黑盒与 hard_blackbox label-only 黑盒
│   ├── 以普通 ResNet18 的 11,227,812 个参数统一归一化跨方法保护比例
│   ├── 汇总四种扫描曲线及 head_only/TensorShield 单点
│   ├── TEESlice 同时保留原生 private ratio，并以统一分母比例绘制 standalone 点
│   ├── no_protection 作为白盒上界，soft/hard full protection 均作为正式黑盒参考线
│   └── results/lab/03_baseline/
│       ├── accuracy.png              保护比例与 surrogate accuracy
│       ├── fidelity.png              保护比例与 fidelity
│       ├── posterior_kl.png          保护比例与 posterior KL
│       ├── metrics.png               三项原始 MS 指标的统一三联图
│       ├── data.tsv                  绘图使用的原始点
│       └── manifest.json             输入协议与 artifact 清单
├── lab/04_tensorshield/
│   ├── 读取作者确认的 ResNet18+CIFAR-100 eligible rank
│   ├── run.py 扫描 Top-1 至 Top-17 前缀
│   ├── ablate.py 对 Top-12 执行单项与联合删除
│   ├── window.py 比较前 10、后 10 与分散 10 项
│   └── results/lab/04_tensorshield/
│       ├── metrics.json/data.tsv/history.tsv 与三张前缀指标图
│       ├── ablation.json/tsv/history.tsv 与三张删除消融图
│       ├── window.json/tsv/history.tsv/png
│       └── 前缀、删除和窗口保护 mask
├── lab/05_state/
│   ├── 按完整 state 类型与参数语义组构造十八种保护集合
│   ├── 区分卷积、BN affine、BN buffer、分类头和完整分支
│   ├── 使用统一 400/100 soft-query validation-best 协议
│   └── results/lab/05_state/
│       ├── metrics.json/data.tsv/history.tsv
│       ├── accuracy/fidelity/posterior_kl.png
│       └── 十八组紧凑保护 mask
├── lab/06_weight/
│   ├── run.py 在 TensorShield Top-10 至 Top-17 上补充 BN gamma、downsample Conv 与 Stem Conv
│   ├── candidate.py 以 seed 43–52 比较 Top-10、BN gamma 闭包及两种删点候选
│   ├── 候选实验同时复用 matched soft 黑盒，形成后续 Lab 的固定基础集合
│   └── results/lab/06_weight/
│       ├── metrics.json/data.tsv/history.tsv/metrics.png 与各扩展 mask
│       └── candidate.json/tsv/history.tsv/png 与五组候选/黑盒 mask
├── lab/07_bn/
│   ├── drop.py 将 20 个 BN gamma 分为 Stem、Block BN1、Block BN2 与 Downsample
│   ├── 以 seed 43–52 执行 No/All gamma 和四种 leave-one-group-out
│   ├── add.py 以 seed 42 从 No gamma 分别加入四类 gamma
│   ├── feature.py 固定 Feature Conv Top-5，再加入三个 downsample Conv 与 Stem BN1
│   └── results/lab/07_bn/
│       ├── drop.json/tsv/history.tsv/png 与六组 mask
│       ├── add.json/tsv/history.tsv/png 与五组 mask
│       └── feature.json/tsv/history.tsv/png 与单组 mask
├── lab/08_structure/
│   ├── run.py 展开 ResNet18 逐算子与 122 个 state unit
│   ├── dependency.py 以十个 seed 逐一暴露固定集合中的五个 conv1
│   ├── swap.py 将五个 conv1 一一替换为同块 conv2
│   ├── pair.py 以 seed 42 比较 conv1+BN2 gamma 与 conv2+BN1 gamma
│   └── results/lab/08_structure/
│       ├── resnet18_c100.tsv 与 tensorshield_top17.tsv
│       ├── dependency.json/tsv/history.tsv/png 与 mask
│       ├── swap.json/tsv/history.tsv/png 与 mask
│       └── pair.json/tsv/history.tsv/png 与两组 mask
├── lab/09_leakage/
│   ├── 固定 Lab06 的 5.7529% 基础集合
│   ├── 将未保护浮点状态从 public 向 victim 取 0%/25%/50%/75%/100%
│   ├── 以 query-validation loss 在五个强度中模拟适应性选择
│   └── results/lab/09_leakage/
│       ├── metrics.json/data.tsv/history.tsv
│       ├── probe.tsv
│       └── metrics.png
├── playground/01_raw/
│   ├── 读取固定 500 张 query、ImageNet public 与 CIFAR-100 victim
│   ├── 只选择 20 个 Conv weight 和 20 个 BN gamma，排除全部 bias 与分类头
│   ├── 保存每项 z_pp/z_pv/z_vp/z_vv 和紧凑公式交叉残差 I
│   ├── 计算未归一化 cross × natural 主分数
│   └── results/playground/01_raw/
│       ├── manifest.json/data.tsv/main.tsv 与 6 张 all/main 图
│       └── activations/             40 个 weight 的 float32 原始四路张量与 I
├── playground/02_rank/
│   ├── 只读 PG01 原始张量，逐图片计算谱有效秩
│   ├── 计算秩差、rank interaction 与 cross-rank × natural-rank
│   ├── 对 all 40 项、main 16 项与 BN gamma 20 项分别重排
│   └── results/playground/02_rank/  metrics/data/main/bn 与 21 张 all/main/bn 图
├── playground/03_feature/
│   ├── 将 PG01 两项残差总量分别除以 C×H×W
│   ├── 以归一化 cross × natural 为主分数
│   ├── 对 all 40 项、main 16 项与 BN gamma 20 项分别重排
│   └── results/playground/03_feature/  metrics/data/main/bn 与 9 张 all/main/bn 图
├── playground/04_param/
│   ├── 将 PG01 两项残差总量分别除以 numel(weight)
│   ├── 以归一化 cross × natural 为保护效率代理
│   ├── 对 all 40 项、main 16 项与 BN gamma 20 项分别重排
│   └── results/playground/04_param/  metrics/data/main/bn 与 9 张 all/main/bn 图
├── playground/05_diagnose/
│   ├── 读取 PG03/PG04 的 bn/main 独立排名并各取 Top-5
│   ├── 每种归一化分别构造 BN、main Conv 与二者 10 项并集，共六组
│   ├── 另构造 Feature Conv+Parameter BN 与 Feature BN+Parameter Conv 两个交叉组
│   ├── 八组共同固定保护分类头，使用 seed-42 soft-query validation-best 协议训练
│   ├── checkpoint 固定后每组只在完整 eval_ms 上评估一次
│   ├── 图中读取正式 full_protection soft-posterior 黑盒参考线
│   └── results/playground/05_diagnose/  metrics/data/history、八个 mask 与三指标图
├── playground/06_mix/
│   ├── 将 PG01 两项残差分别除以 sqrt(C×H×W×numel(weight))
│   ├── 以归一化 cross × natural 为特征量与参数量联合分数
│   ├── 对 all 40 项、main 16 项与 BN gamma 20 项分别重排
│   └── results/playground/06_mix/  metrics/all/main/bn 与 9 张 all/main/bn 图
└── playground/07_topk/
    ├── 固定替换分类头、Stem BN1 gamma 与三个 downsample Conv
    ├── 读取 PG03 Feature main 排名并按顺序形成最多 Top-0–16 的嵌套 mask
    ├── 每组独立重放 seed-42 canonical 初始化并按 query validation 选模
    ├── 任一指标相对前一级反弹时保留反弹点并停止后续 Top-k
    ├── checkpoint 固定后每组只在完整 eval_ms 上评估一次
    └── results/playground/07_topk/  metrics/data/history、实际 case mask 与两张 Top-k 曲线
```
