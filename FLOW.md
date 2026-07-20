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
│   └── results/lab/02_head/
│       ├── metrics.json              八组 validation-best 与单次 eval_ms 原始指标
│       └── history.tsv               八组共 800 条 train/validation 记录
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
│   ├── 读取 ResNet18+CIFAR-100 victim 与 500 条 soft posterior query
│   ├── 固定拆为 400 train / 100 validation，选模后每组只评估一次 eval_ms
│   ├── 从作者确认的 41-weight rank 派生 17-weight eligible rank
│   ├── 分别构造 Top-1 至 Top-17 前缀 mask，每组固定加入 unit 121；Top-10 对应 Figure 12(d)
│   ├── 每个 k 重置种子并最多微调 100 轮，按 validation soft cross-entropy 选 best
│   ├── 从 16 个非分类头 eligible 候选构造前 10、后 10 与分散 10 三组同数量集合
│   ├── spread_10 固定候选位置 1,2,3,5,7,9,11,13,15,16
│   ├── 三组均固定保护分类头 weight/bias，只比较非头候选位置与实际参数成本
│   ├── 三组均独立重放 canonical 初始化并按相同 validation-best 协议训练
│   ├── 构造 Top-10、+BN gamma、删除 rank-5/8/10 及额外删除 rank-6 四种受控策略
│   ├── 四组都保护分类头 weight/bias；两组删点候选分别保留 rank-6/7 或只保留 rank-7
│   ├── 四组分别保护 8.9934%、9.0362%、7.0662%、5.7529%，以 seed 43–52 验证
│   ├── 每个 seed 同时训练四种策略与 matched soft full-protection 黑盒
│   └── results/lab/04_tensorshield/
│       ├── metrics.json              作者 rank、17 组 validation-best 与最终指标
│       ├── history.tsv               17 组共 1,700 轮 train/validation 记录
│       ├── data.tsv                  Top-k 曲线原始点
│       ├── accuracy.png              参数占比断轴与 surrogate accuracy，放大 0–15% 区间
│       ├── fidelity.png              参数占比断轴与 fidelity，放大 0–15% 区间
│       ├── posterior_kl.png          参数占比断轴与 posterior KL，放大 0–15% 区间
│       ├── top_<k>_mask.pt           17 组紧凑保护掩码
│       ├── ablation.json/tsv         Top-12 完整 leave-one-out、五组联合删除与黑白盒边界
│       ├── ablation_history.tsv      十七组删除消融共 1,700 轮 query 训练记录
│       ├── ablation_<metric>.png     accuracy、fidelity 与 posterior KL 三张独立消融图
│       ├── drop_<rank>_mask.pt       十二组单删与五组联合删除集合的紧凑保护掩码
│       ├── candidate.json/tsv/png    四策略十种子、配对效应、黑盒参考与聚合统计
│       ├── candidate_history.tsv     五组十种子共 5,000 轮 query 训练记录
│       ├── candidate*_mask.pt        四种候选策略的紧凑保护掩码
│       ├── candidate_full_mask.pt    soft full-protection 对照掩码
│       ├── window.json/tsv/png       三个候选位置集合的指标和参数占比三联直方图
│       ├── window_history.tsv        三组集合共 300 轮 query 训练记录
│       └── <first|spread|last>_10_mask.pt  三个集合的紧凑保护掩码
├── lab/05_state/
    ├── 分别只保护五种完整 state 类型或十三种参数语义组，其余 victim 状态全部复制
    ├── 语义组拆分主路径/Stem/downsample Conv、局部/全局 BN affine、分类头与完整分支
    ├── 使用统一 400/100 soft-posterior validation-best 协议训练十八个 surrogate
    ├── 以 protected_state_byte_ratio 为横坐标绘制独立散点
    └── results/lab/05_state/
        ├── metrics.json              十八组保护统计与 validation-best 原始指标
        ├── history.tsv               十八组各 100 轮 train/validation 记录
        ├── data.tsv                  绘图使用的单次 eval_ms 原始点
        ├── accuracy/fidelity/posterior_kl.png
        └── <group>_mask.pt           十八组紧凑保护掩码
├── lab/06_weight/
    ├── 读取 Lab04 Top-10 至 Top-17 的八个前缀点、mask 与固定 eligible rank
    ├── 分别补充全部 BN gamma、三个 downsample Conv、二者组合、Stem Conv 或三类并集
    ├── 每个新增组合重置种子并按统一 400/100 validation-best 协议训练
    ├── Lab04 的八个原始 Top-k 点只复用，新增五条曲线共训练四十组
    ├── 读取 Lab05 weight 结果作为 Top-17 并集的跨实验终点参考
    └── results/lab/06_weight/
        ├── metrics.json              四十八个组合、保护统计、输入哈希与最终指标
        ├── history.tsv               四十个新增组合共 4,000 轮 train/validation 记录
        ├── data.tsv                  六条曲线的成本、原始指标与相对 Top-k 差值
        ├── metrics.png               三项 MS 指标与 soft/hard 黑盒边界三联曲线
        └── <case>_mask.pt            四十个新增组合的紧凑保护掩码
├── lab/07_structure/
    ├── 读取 models/imagenet.py 的 ImageNet-style ResNet18 结构
    ├── 读取 structure.tsv 的不重复计算节点与输入输出尺寸
    ├── 按 state_dict 顺序将有状态模块展开为一个 unit 一行
    ├── 读取 TensorShield eligible rank，并在 H 列标注 Top-1 至 Top-17
    ├── 以 1×3×32×32 虚拟输入逐模块核对输出尺寸
    ├── 显式展开 8 个 BasicBlock 的主分支、shortcut 与残差相加
    ├── 读取 Lab04 的五个 conv1+BN gamma+分类头基础集合及 matched soft 黑盒
    ├── 分别暴露 rank-1/2/4/7/9，以 seed 43–52 训练五十个 leave-one-out surrogate
    ├── 使用同 seed 配对 accuracy/fidelity/KL 反弹检验成员级条件攻击依赖
    ├── 固定分类头和 BN gamma，将五个受保护 conv1 一一换成对应 conv2
    ├── conv1 与黑盒复用既有十种子结果，conv2 新增十组100%直接拼接攻击
    └── results/lab/07_structure/
        ├── resnet18_c100.tsv         逐算子、逐 unit 结构及 TensorShield Top-17 标注
        ├── tensorshield_top17.tsv    按 unit 排序的 17 个 TensorShield weight 子集
        ├── dependency.json/tsv/png   五个 conv1 的十种子配对消融、聚合与可视化
        ├── dependency_history.tsv    五十组共 5,000 轮 query train/validation 记录
        ├── dependency_*_mask.pt      基础、五组 leave-one-out 与黑盒紧凑 mask
        ├── swap.json/tsv/png         conv1、对应 conv2 与 soft 黑盒三指标对比
        ├── swap_history.tsv          conv2 十组共 1,000 轮训练/验证记录
        └── swap_conv2_mask.pt        conv2、全部 BN gamma 与分类头的紧凑 mask
├── lab/08_leakage/
    ├── 固定 Lab04 的 5.7529% 系统保护集合，不改变 tensor 数量或可训练参数
    ├── 令未保护浮点状态从 public 向 victim 取 0%/25%/50%/75%/100% 利用强度
    ├── 0%/100% 复用 Lab04 matched soft 黑盒与最终候选，只训练三十个中间点
    ├── 以 seed 43–52 的相同 400/100 query 划分执行全参数 validation-best finetune
    ├── 五个强度均补测 epoch-0 query train/validation，区分初始化失配与泛化负迁移
    ├── 适应性攻击者只按 query-validation loss 在五个强度中选择，不读取 eval_ms
    └── results/lab/08_leakage/
        ├── metrics.json              五十个强度点、十种子聚合、配对效应与适应性攻击
        ├── data.tsv                  最终 MS 指标、训练/验证目标和相对 0% 的逐 seed 差值
        ├── history.tsv               三十组共 3,000 轮 query train/validation 记录
        ├── probe.tsv                 五十个 epoch-0 初始化探针
        └── metrics.png               最终三指标与训练前后攻击目标的六联图
├── lab/09_mechanism/
    ├── 读取 Lab04 的 5.7529% 固定保护集合与 Lab08 的五个泄露状态利用强度
    ├── 仅在 seed 43–52 的 query-validation 上执行未经训练的前向因果干预
    ├── 将各强度分类头输入缩放到同图片 public 特征范数，区分幅值与方向失配
    ├── 把 27 个受保护 state 分成七组，枚举全部 128 个 oracle-reveal 组合
    ├── 计算每组 posterior KL 的单独恢复、条件损失、精确 Shapley 与分类头交互
    ├── 在八个 BasicBlock 内比较 conv1/conv2、对应 BN gamma 及二者组合的接口损伤
    ├── 把 20 个 BN gamma 分成四类，枚举 16 个 public 替换组合分析跨层尺度闭包
    ├── 将五个 conv1 的近-victim KL 损伤与 Lab07 十种子 MS 反弹做五点相关核对
    └── results/lab/09_mechanism/
        ├── metrics.json              来源、协议、三类机制结果与十种子聚合
        ├── lambda.tsv                五个利用强度及分类头输入范数反事实
        ├── lattice.tsv               七组状态的 128 组合 × 十 seed 原始结果
        ├── attribution.tsv           七组的逐 seed 因果贡献与聚合
        ├── seam.tsv                  八个残差块的 conv1/conv2 成对接口干预
        ├── bn.tsv                    四类 BN gamma 的 16 组合 × 十 seed 原始结果
        └── metrics.png               范数反事实、组贡献和块内对照图
└── test/MS/01_cross/
    ├── 读取 query_pool_ms 按 query_rank 排序的全部 500 张 CIFAR-100 图片
    ├── 读取 official_train 全部 50,000 张图片进行一次全训练集对照
    ├── 读取官方 ImageNet ResNet18 public 权重与 victim best.pth
    ├── 候选固定为全部 20 个 Conv weight 与 20 个 BN gamma
    ├── BN running state 只构造标准化输入，beta 与分类头不进入联合排名
    ├── Conv 计算输入差与权重差的乘性交叉项
    ├── BN gamma 计算标准化输入差与 gamma 差的乘性交叉项
    ├── 每张图片对交叉项取绝对值并除 C×H×W，再对全部图片取平均
    ├── 分别生成 500-query 与 50,000-image 的 40 项统一排名
    ├── 从 500-query 统一表直接抽取 16 个 BasicBlock 主分支卷积
    ├── 不重复计算原 16 个卷积的全训练集版本
    ├── 不生成保护 mask，不训练 surrogate，也不读取 eval_ms
    └── results/test/MS/01_cross/
        ├── weights.json/tsv          500 张 query 的协议、检查与 40 项排名
        ├── weights_conv/bn.tsv       500 张结果按候选类型拆分的排名
        ├── weights.png               500 张 query 的 40 项统一柱状图
        ├── weights_full.json/tsv     全部 50,000 张训练图片的协议与 40 项排名
        ├── weights_full_conv/bn.tsv  全训练集结果按候选类型拆分的排名
        ├── weights_full.png          全训练集的 40 项统一柱状图
        ├── tensors.tsv               从 500 张统一表抽取的 16 个主分支卷积
        └── tensors.png               16 个主分支卷积的绝对残差均值柱状图

```
