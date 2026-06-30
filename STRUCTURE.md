# 目录结构图

本图只展示同级目录和下一层目录的作用。随机种子、比例、日期、超参数和运行名等细节不进入目录图，应写入就近的 `README.md`、`manifest.json`、`params.json` 或实验日志。

```text
TSDP/
├── dataset/
│   ├── public/                 原始公开数据
│   │   ├── cifar10/
│   │   ├── cifar100/
│   │   ├── stl10/
│   │   └── tiny-imagenet-200/
│   ├── derived/                无标签查询索引等中间数据
│   │   ├── cifar10/
│   │   ├── cifar100/
│   │   ├── stl10/
│   │   └── tiny-imagenet-200/
│   ├── pseudo_labels/          victim 模型生成的伪标签数据集
│   │   ├── cifar10/             按 victim 模型分层保存 CIFAR-10 伪标签
│   │   ├── cifar100/            按 victim 模型分层保存 CIFAR-100 伪标签
│   │   ├── stl10/               按 victim 模型分层保存 STL-10 伪标签
│   │   └── tiny-imagenet-200/   按 victim 模型分层保存 Tiny-ImageNet 伪标签
│   ├── download_datasets.sh    下载和整理公开数据集
│   └── make_unlabeled_query_set.py
│                                从评估 split 构造无标签查询索引
├── exp/
│   ├── train_victim/           训练 victim 模型
│   │   ├── common/
│   │   ├── resnet18/
│   │   ├── resnet50/
│   │   ├── vgg16_bn/
│   │   └── mobilenetv2/
│   └── make_pseudo_labels/     构造伪标签数据集
│       ├── common/
│       ├── resnet18/
│       ├── resnet50/
│       ├── vgg16_bn/
│       └── mobilenetv2/
├── lab/
│   └── 01_resnet18_cifar10_kmeans/
│                                ImageNet 预训练特征的 CIFAR-10 聚类探测
├── models/
│   └── imagenet.py             ImageNet 模型 wrapper 和统一接口
├── weights/
│   ├── pre_train/              官方 ImageNet 预训练权重
│   └── victim/                 训练得到的 victim 权重
├── results/
│   └── lab/                    lab 实验结果
├── docs/                       论文材料、计划和补充说明
├── verify/                     数据或环境验证脚本
├── FLOW.md                     实验流程图
├── STRUCTURE.md                目录结构图
└── AGENTS.md                   项目协作约定
```
