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
│       └── posteriors.pt             query_pool_ms 固定顺序的 posterior
└── exp/MS/train_surrogate/
    ├── 读取 query_pool_ms 的预算前缀与 victim 伪标签
    ├── defense/                      注册策略、生成掩码并组合公开与 victim 权重
    ├── core/                         统一读取数据、训练、评估和写入产物
    ├── weights/MS/surrogate/<model>/<dataset>/<run_id>/
    │   ├── best.pth                  surrogate_acc 最高的 checkpoint
    │   ├── end.pth                   最后一个 epoch 的 checkpoint
    │   ├── protection_mask.pt        策略的唯一 unit 保护方案
    │   ├── params.json               训练参数与掩码摘要
    │   └── train.log.tsv             每个 epoch 的原始观测
    └── results/MS/<model>/<dataset>/
        ├── metrics.tsv               各 run 的原始指标索引
        └── <run_id>/metrics.json     best 与 end 的 MS 原始指标

Lab 验证实验/
├── lab/01_kmeans/
│   ├── 读取 dataset/public/c100 的 CIFAR-100 test split 与 ImageNet 预训练 ResNet18
│   └── results/lab/01_kmeans/
│       ├── metrics.json              100 类 KMeans、Hungarian 匹配和 NMI
│       └── confusion_matrix_*.png    聚类混淆矩阵
└── lab/02_head/
    ├── 读取 dataset/MS/c100/resnet18/labels.tsv 的 500 条 hard query
    ├── 全保护                        不读取 victim 权重，比较四种攻击配置
    ├── 随机保护 61/122 unit          分类头固定保护，复制其余暴露 victim 权重
    ├── 训练 replace/adapter 与 frozen/finetune 共八组配置
    └── results/lab/02_head/
        ├── metrics.json              保护计划与八组 best/end 原始 MS 指标
        └── history.tsv               八组共 800 条逐 epoch 训练记录
```
