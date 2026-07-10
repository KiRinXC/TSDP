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
│       └── transfer/                构造 MS 划分与查询 victim
├── weights/                         可复用模型权重
│   ├── pre_train/                   ImageNet 预训练权重
│   └── MS/                          MS victim 与 surrogate 权重
├── results/                         实验指标与可视化结果
├── models/                          统一模型结构接口
├── verify/                          数据与协议验证脚本
├── lab/                             小型验证实验
└── docs/                            论文证据与研究文档
```
