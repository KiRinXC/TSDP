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
│       ├── best.pth                  最佳验证准确率 checkpoint，也是 query 默认权重
│       ├── end.pth                   最末 epoch 模型
│       ├── train.log.tsv             训练和评估日志
│       └── params.json               可复现实验参数
├── exp/MS/transfer/get_label.py
│   ├── 读取 victim best.pth
│   └── dataset/MS/<dataset>/<model>/
│       ├── labels.tsv                hard pseudo label 与 confidence
│       └── posteriors.pt             test transform 下固定顺序的 posterior
└── exp/MS/train_surrogate/
    ├── 读取 query_pool_ms 的预算前缀与 victim soft posterior
    ├── posterior_replace_finetune_v2 统一确定性 query、mask 权威初始化、微调与 step60 协议
    │   └── 分类头完整暴露时复制、部分暴露时 mixed、完整保护时替换
    ├── plan.py
    │   ├── exp/MS/train_surrogate/baseline.json    固化 32 组 baseline 配置与保护统计
    │   └── weights/MS/surrogate/resnet18/c100/baseline.pt
    │                                           保存对应的紧凑保护 mask
    ├── sweep.py                       按 plan_id 并行训练完整层及全局标量 baseline
    ├── defense/                      注册策略、生成掩码并组合公开与 victim 权重
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

TEESlice 基线/
├── exp/MS/train_victim/teeslice/train.py
│   ├── 读取普通 ResNet18 victim best.pth、victim_train 与官方 ImageNet 权重
│   ├── 在 victim_train 内临时划分 90% 训练部分与 10% 内部验证部分
│   ├── teacher 阶段                 将普通 victim 蒸馏到 CIFAR ResNet18 teacher
│   ├── full 阶段                    冻结公开参数，训练 private slice、alpha、分类头并适配 BN 状态
│   ├── prune 阶段                   只用内部验证集迭代删除低 alpha proxy
│   ├── weights/MS/victim/teeslice_r18/c100/
│   │   ├── teacher/{best,end}.pth   teacher 阶段 checkpoint
│   │   ├── full/{best,end}.pth      未剪枝 TEESlice checkpoint
│   │   ├── best.pth                 最终 defended victim，供 query 使用
│   │   ├── end.pth                  prune 阶段固定训练终点
│   │   ├── params.json              训练、剪枝与输入追踪信息
│   │   └── train.log.tsv            三阶段逐 epoch 记录
│   └── results/MS/resnet18/c100/teeslice/victim.json
│                                       defended victim 的效用与保护成本
├── exp/MS/transfer/get_label.py teeslice_r18 c100
│   ├── 读取通用 query_pool_ms 与 TEESlice best.pth
│   └── dataset/MS/c100/teeslice_r18/
│       ├── labels.tsv               defended victim hard label 与 confidence
│       └── posteriors.pt            确定性 test transform 下的 soft posterior
└── exp/MS/train_surrogate/teeslice/attack.py
    ├── 以官方 ImageNet CIFAR ResNet18 和随机 C100 头初始化攻击者
    ├── 使用 500 条 soft posterior query 全模型微调 100 epoch
    ├── weights/MS/surrogate/resnet18/c100/teeslice/{best,end}.pth
    └── results/MS/resnet18/c100/teeslice/metrics.json
                                            相对 defended victim 的原始 MS 指标

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
│   ├── 按实际 protected_param_ratio 汇总四种 baseline 的 end 指标
│   ├── no_protection 与 full_protection 仅作为水平参考线
│   └── results/lab/03_baseline/
│       ├── accuracy.png              保护比例与 surrogate accuracy
│       ├── fidelity.png              保护比例与 fidelity
│       ├── posterior_kl.png          保护比例与 posterior KL
│       ├── data.tsv                  绘图使用的原始点
│       └── manifest.json             输入协议与 artifact 清单
├── lab/04_tensorshield/
│   ├── 读取 ResNet18+CIFAR-100 victim、500 条 soft posterior query 与 eval_ms
│   ├── 从作者确认的 41-weight rank 派生 17-weight eligible rank
│   ├── 分别构造 Top-1 至 Top-12 前缀 mask；Top-10 对应 Figure 12(d)
│   ├── 每个 k 重置种子并微调 100 轮，只在 end 读取 eval_ms
│   ├── 分别保护作者原始 41-weight rank 的 11-20 与 32-41 窗口，各训练一次
│   └── results/lab/04_tensorshield/
│       ├── metrics.json              作者 rank、12 组保护统计与 end 原始指标
│       ├── history.tsv               12 组共 1,200 轮 query 训练记录
│       ├── data.tsv                  Top-k 曲线原始点
│       ├── metrics.png               accuracy、fidelity 与 KL 三联曲线
│       ├── top_<k>_mask.pt           12 组紧凑保护掩码
│       ├── ablation.json/tsv/png     rank-5/rank-10 删除消融与对比图
│       ├── ablation_history.tsv      两组新增消融共 200 轮 query 训练记录
│       ├── drop_<rank>_mask.pt       两组新增删除集合的紧凑保护掩码
│       ├── window.json/tsv/png       两个原始 rank 窗口的指标和保护成本对照
│       ├── window_history.tsv        两组窗口共 200 轮 query 训练记录
│       └── rank_<range>_mask.pt      两个窗口的紧凑保护掩码
└── lab/05_state/
    ├── 分别只保护 weight、bias 和三种 BN buffer，其余 victim 状态全部复制
    ├── 使用统一 soft posterior 与 finetune 协议分别训练五个 surrogate
    ├── 以 protected_state_byte_ratio 为横坐标绘制独立散点
    └── results/lab/05_state/
        ├── metrics.json              五种 state 类型的保护统计与原始指标
        ├── history.tsv               五组各 100 轮训练和评估记录
        ├── data.tsv                  绘图使用的 end 原始点
        ├── accuracy/fidelity/posterior_kl.png
        └── <type>_mask.pt            五种类型的紧凑保护掩码
```
