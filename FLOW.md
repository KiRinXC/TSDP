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
    ├── posterior_replace_finetune_v2 统一确定性 query、mask 权威初始化、微调与 step60 协议
    │   └── 分类头完整暴露时复制、部分暴露时 mixed、完整保护时替换
    ├── hard_label_replace_finetune_v1
    │   ├── 仅 ResNet18+C100 全保护，确定性 test transform 与 500 条 hard query
    │   └── <artifact_id=hard_blackbox> 只作输出能力对比，不替换 soft 主黑盒
    ├── plan.py
    │   ├── exp/MS/train_surrogate/baseline.json    固化 32 组 baseline 配置与保护统计
    │   └── weights/MS/surrogate/resnet18/c100/baseline.pt
    │                                           保存对应的紧凑保护 mask
    ├── sweep.py                       按 plan_id 并行训练完整层及全局标量 baseline
    ├── defense/                      注册策略、生成掩码并组合公开与 victim 权重
    │   └── head_only                 仅保护 last_linear.weight/bias，复制全部 backbone
    ├── core/                         统一读取数据、训练、评估和写入产物
    ├── weights/MS/surrogate/<model>/<dataset>/<artifact_id>/
    │   ├── best.pth                  surrogate_acc 最高点，仅作训练诊断
    │   ├── end.pth                   固定训练终点，也是正式主结果 checkpoint
    │   ├── protection_mask.pt        策略的唯一 unit 保护方案
    │   ├── params.json               训练参数与掩码摘要
    │   └── train.log.tsv             每个 epoch 的原始观测
    └── results/MS/<model>/<dataset>/
        ├── metrics.tsv               各 run 的 end 原始指标索引
        └── <artifact_id>/metrics.json
                                             best 与 end 的 MS 原始指标

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
    ├── 使用 500 条 soft posterior query 全模型微调 100 epoch
    ├── whitebox_full_state           重新加载最终状态并实际执行 eval_ms
    ├── weights/MS/surrogate/resnet18/c100/teeslice/
    │   ├── {best,end}.pth            黑盒 surrogate checkpoint
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
    ├── 读取通用 500 条 soft posterior query 与 eval_ms
    ├── 全参数微调 100 epoch，只以 end.pth 作为主评估点
    ├── weights/MS/surrogate/resnet18/c100/tensorshield/protection_mask.pt
    └── results/MS/resnet18/c100/tensorshield/metrics.json

Lab 验证实验/
├── lab/01_kmeans/
│   ├── 读取 dataset/public/c100 的 CIFAR-100 test split 与 ImageNet 预训练 ResNet18
│   └── results/lab/01_kmeans/
│       ├── metrics.json              100 类 KMeans、Hungarian 匹配和 NMI
│       └── confusion_matrix_*.png    聚类混淆矩阵
├── lab/02_head/
│   ├── 读取 dataset/MS/c100/resnet18/labels.tsv 的 500 条 hard query
│   ├── 全保护                        不读取 victim 权重，比较四种攻击配置
│   ├── 随机保护 61/122 unit          分类头固定保护，复制其余暴露 victim 权重
│   ├── 训练 replace/adapter 与 frozen/finetune 共八组配置
│   └── results/lab/02_head/
│       ├── metrics.json              保护计划与八组 best/end 原始 MS 指标
│       └── history.tsv               八组共 800 条逐 epoch 训练记录
├── lab/03_baseline/
│   ├── 读取 results/MS/resnet18/c100/<artifact_id>/metrics.json
│   ├── 读取正式 hard_blackbox/metrics.json 的 hard-label 辅助参考
│   ├── 按实际 protected_param_ratio 汇总四种扫描曲线及 head_only/TensorShield 单点
│   ├── TEESlice 只使用 standalone 黑盒 end 指标并以独立标记展示
│   ├── 普通 victim 的 no_protection 与 soft full_protection 作为主参考线
│   ├── hard full_protection 只作输出能力消融参考线
│   └── results/lab/03_baseline/
│       ├── accuracy.png              保护比例与 surrogate accuracy
│       ├── fidelity.png              保护比例与 fidelity
│       ├── posterior_kl.png          保护比例与 posterior KL
│       ├── metrics.png               三项原始 MS 指标的统一三联图
│       ├── data.tsv                  绘图使用的原始点
│       └── manifest.json             输入协议与 artifact 清单
├── lab/04_tensorshield/
│   ├── 读取 ResNet18+CIFAR-100 victim、500 条 soft posterior query 与 eval_ms
│   ├── 从作者确认的 41-weight rank 派生 17-weight eligible rank
│   ├── 分别构造 Top-1 至 Top-17 前缀 mask，每组固定加入 unit 121；Top-10 对应 Figure 12(d)
│   ├── 每个 k 重置种子并微调 100 轮，只在 end 读取 eval_ms
│   ├── 从 16 个非分类头 eligible 候选构造前 10、后 10 与分散 10 三组同数量集合
│   ├── spread_10 固定候选位置 1,2,3,5,7,9,11,13,15,16
│   ├── 三组均固定保护分类头 weight/bias，只比较非头候选位置与实际参数成本
│   ├── 三组均独立重放 canonical 初始化并训练 100 轮，只在 end 读取 eval_ms
│   └── results/lab/04_tensorshield/
│       ├── metrics.json              作者 rank、17 组保护统计与 end 原始指标
│       ├── history.tsv               17 组共 1,700 轮 query 训练记录
│       ├── data.tsv                  Top-k 曲线原始点
│       ├── accuracy.png              参数占比断轴与 surrogate accuracy，放大 0–15% 区间
│       ├── fidelity.png              参数占比断轴与 fidelity，放大 0–15% 区间
│       ├── posterior_kl.png          参数占比断轴与 posterior KL，放大 0–15% 区间
│       ├── top_<k>_mask.pt           17 组紧凑保护掩码
│       ├── ablation.json/tsv/png     Top-12 内 rank-5/rank-10 删除消融、黑白盒边界与对比图
│       ├── ablation_history.tsv      三组新增消融共 300 轮 query 训练记录
│       ├── drop_<rank>_mask.pt       三组新增删除集合的紧凑保护掩码
│       ├── window.json/tsv/png       三个候选位置集合的指标和参数占比三联直方图
│       ├── window_history.tsv        三组集合共 300 轮 query 训练记录
│       └── <first|spread|last>_10_mask.pt  三个集合的紧凑保护掩码
├── lab/05_state/
    ├── 分别只保护五种完整 state 类型或十三种参数语义组，其余 victim 状态全部复制
    ├── 语义组拆分主路径/Stem/downsample Conv、局部/全局 BN affine、分类头与完整分支
    ├── 使用统一 soft posterior 与 finetune 协议分别训练十八个 surrogate
    ├── 以 protected_state_byte_ratio 为横坐标绘制独立散点
    └── results/lab/05_state/
        ├── metrics.json              十八组保护统计与 end 原始指标
        ├── history.tsv               十八组各 100 轮训练和评估记录
        ├── data.tsv                  绘图使用的 end 原始点
        ├── accuracy/fidelity/posterior_kl.png
        └── <group>_mask.pt           十八组紧凑保护掩码
└── lab/06_weight/
    ├── 读取 Lab04 Top-10 至 Top-17 的八个前缀点、mask 与固定 eligible rank
    ├── 分别补充全部 BN gamma、三个 downsample Conv、二者组合、Stem Conv 或三类并集
    ├── 每个新增组合重置种子并微调 100 轮，只在 end 读取 eval_ms
    ├── Lab04 的八个原始 Top-k 点只复用，新增五条曲线共训练四十组
    ├── 读取 Lab05 weight 结果作为 Top-17 并集的跨实验终点参考
    └── results/lab/06_weight/
        ├── metrics.json              四十八个组合、保护统计、输入哈希与 end 指标
        ├── history.tsv               四十个新增组合共 4,000 轮 query 训练记录
        ├── data.tsv                  六条曲线的成本、原始指标与相对 Top-k 差值
        ├── metrics.png               三项 MS 指标与 soft 黑盒边界三联曲线
        └── <case>_mask.pt            四十个新增组合的紧凑保护掩码

临时 ARC 验证/
└── temp/run.py
    ├── 从 victim_train 排除正式 query 后构造 discovery query/holdout
    ├── 在 272 个计算图对齐通道块上优化攻击可恢复性门，分类头固定保护
    ├── 固定 8% 参数上限并硬化为全局静态 mask，再运行正式 500-query soft MS
    └── temp/output/
        ├── selection.json/tsv        数据隔离、候选块、优化协议与选择轨迹
        ├── mask.pt                   37 个通道块加完整分类头的保护 mask
        ├── attack.tsv                最终 surrogate 的 100 轮训练记录
        ├── metrics.json              end 指标与正式边界对照
        └── end.pth                   仅本地生成并由 Git 忽略的临时 checkpoint
```
