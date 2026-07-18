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
│       ├── train_victim/            训练普通及 TEESlice defended victim
│       │   ├── common/              普通 victim 公共训练逻辑
│       │   ├── mobilenetv2/         MobileNetV2 victim 入口
│       │   ├── resnet18/            ResNet18 victim 入口
│       │   ├── resnet50/            ResNet50 victim 入口
│       │   ├── vgg16_bn/            VGG16-BN victim 入口
│       │   └── teeslice/            TEESlice 四阶段独立训练与动态剪枝
│       ├── transfer/                构造 MS 划分与查询 victim
│       └── train_surrogate/         初始化权重隐藏 baseline 并训练 surrogate
│           ├── core/                公共数据、训练、评估与产物管理
│           ├── defense/             策略插件、保护掩码与 unit 映射
│           ├── selector/            TensorShield 作者确认的固定 rank 列表
│           └── teeslice/            TEESlice 已知剪枝拓扑的黑盒攻击与完整状态白盒评估
├── weights/                         可复用模型权重
│   ├── pre_train/                   ImageNet 预训练权重
│   └── MS/                          MS victim/surrogate 权重与 unit 保护掩码
├── results/                         实验指标与可视化结果
│   ├── MS/                          surrogate 原始指标与 run 索引
│   └── lab/                         Lab 实验结果与可视化
│       ├── 01_kmeans                ResNet18+CIFAR-100 特征聚类
│       ├── 02_head                  全保护/随机保护的分类头与权重消融
│       ├── 03_baseline              普通 MS 策略、双黑盒参考线与 TEESlice 独立点总览
│       ├── 04_tensorshield          TensorShield 前缀、三策略十种子候选及消融
│       ├── 05_state                 State 类型与参数语义保护的 MS 对比结果
│       └── 06_weight                TensorShield Top-k 的遗漏 weight 语义闭包结果
├── models/                          统一模型结构及 TEESlice slice/backbone 接口
├── verify/                          环境/GPU、数据协议、surrogate 与固定 rank mask 验证
├── lab/                             小型验证实验
│   ├── 01_kmeans                    ResNet18+CIFAR-100 特征聚类
│   ├── 02_head                      分类头与权重训练方式消融
│   ├── 03_baseline                  MS 策略保护比例、双黑盒参考线与三项指标总览
│   ├── 04_tensorshield              TensorShield 前缀、三策略十种子候选与集合验证
│   ├── 05_state                     State 类型与参数语义保护对比
│   └── 06_weight                    TensorShield Top-k 的遗漏 weight 语义闭包验证
├── temp/                            交叉残差与因果残差的 filter 级临时验证
│   └── output/                      残差分数、filter mask、统一 MS 指标与图
└── docs/                            参考论文
```
