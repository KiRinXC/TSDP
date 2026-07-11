# 目录结构

```text
TSDP/
├── dataset/                         公开数据与实验协议
│   ├── public/                      原始公开数据
│   └── MS/                          MS 随机重叠划分与 victim 查询产物
│       ├── c10/                     CIFAR-10 的 manifest、splits 与模型标签
│       ├── c100/                    CIFAR-100 的 manifest、splits 与模型标签
│       ├── s10/                     STL10 的 manifest、splits 与模型标签
│       └── t200/                    Tiny-ImageNet 的 manifest、splits 与模型标签
├── exp/                             正式实验代码
│   └── MS/                          Model Stealing 实验
│       ├── train_victim/            训练受害者模型
│       ├── transfer/                构造 MS 划分与查询 victim
│       └── train_surrogate/         初始化 baseline 并训练 surrogate
│           ├── core/                公共数据、训练、评估与产物管理
│           └── defense/             策略插件、保护掩码与 unit 映射
├── weights/                         可复用模型权重
│   ├── pre_train/                   ImageNet 预训练权重
│   └── MS/                          MS victim/surrogate 权重与 unit 保护掩码
├── results/                         实验指标与可视化结果
│   ├── MS/                          surrogate 原始指标与 run 索引
│   └── lab/                         Lab 实验结果与可视化
│       ├── 01_kmeans                ResNet18+CIFAR-100 特征聚类
│       └── 02_head                  全保护/随机保护的分类头与权重消融
├── models/                          统一模型结构接口
├── verify/                          数据协议与 surrogate 语义验证脚本
├── lab/                             小型验证实验
│   ├── 01_kmeans                    ResNet18+CIFAR-100 特征聚类
│   └── 02_head                      分类头与权重训练方式消融
└── docs/                            论文证据与研究文档
```
