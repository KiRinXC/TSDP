# 实验流程图

本图用目录树形式记录当前项目已经纳入流程的主要实验、输入和输出。每次新增实验、调整关键输入输出或新增数据产物时，都应同步更新本图。

```text
TSDP 实验流程
├── 1. 实验数据集下载
│   ├── 命令
│   │   └── bash dataset/download_datasets.sh
│   └── 输出
│       └── dataset/public/
│           ├── cifar10/
│           ├── cifar100/
│           ├── stl10/
│           └── tiny-imagenet-200/
├── 2. 实验模型下载和结构适配
│   ├── 下载 ImageNet-1k 官方预训练权重
│   │   ├── 命令
│   │   │   └── bash weights/pre_train/download_pretrained_weights.sh
│   │   └── 输出
│   │       └── weights/pre_train/
│   └── 修改和统一模型接口
│       └── models/imagenet.py
│           ├── 统一 last_linear
│           ├── 暴露 features()
│           └── 暴露 logits()
├── 3. 构造无标签查询集
│   ├── 命令
│   │   └── python3 dataset/make_unlabeled_query_set.py --dataset <dataset>
│   ├── 输入
│   │   └── dataset/public/<dataset>/
│   ├── 规则
│   │   ├── 样本从验证集中抽取
│   │   └── 默认样本数为 floor(训练集大小 * 1%)
│   └── 输出
│       └── dataset/derived/<dataset>/
│           ├── manifest.json
│           └── samples.tsv
├── 4. exp 实验：训练 victim 模型
│   ├── 命令
│   │   └── bash exp/train_victim/<model>/run.sh <dataset>
│   ├── 输入
│   │   ├── dataset/public/<dataset>/
│   │   ├── weights/pre_train/<model>.pth
│   │   └── models/imagenet.py
│   └── 输出
│       └── weights/victim/<model>/<dataset>/
│           ├── target.pth
│           ├── checkpoint.pth.tar
│           ├── params.json
│           └── train.log.tsv
├── 5. lab 实验：观察预训练模型特征空间的分类能力
│   ├── 命令
│   │   └── python3 lab/01_resnet18_cifar10_kmeans/run.py
│   ├── 输入
│   │   ├── dataset/public/cifar10/
│   │   ├── weights/pre_train/resnet18-5c106cde.pth
│   │   └── models/imagenet.py
│   ├── 方法
│   │   ├── 提取 ResNet18 ImageNet 预训练特征
│   │   ├── KMeans 聚类
│   │   └── 评估阶段做 cluster 到 label 的映射
│   └── 输出
│       └── results/lab/01_resnet18_cifar10_kmeans/
│           ├── confusion_matrix_optimal.png
│           ├── confusion_matrix_greedy.png
│           └── metrics.json
└── 6. exp 实验：制作伪标签数据集
    ├── 命令
    │   └── bash exp/make_pseudo_labels/<model>/run.sh <dataset>
    ├── 输入
    │   ├── dataset/derived/<dataset>/manifest.json
    │   ├── dataset/public/<dataset>/
    │   └── weights/victim/<model>/<dataset>/target.pth
    └── 输出
        └── dataset/pseudo_labels/<dataset>/<model>/
            ├── manifest.json
            └── samples.tsv
```
