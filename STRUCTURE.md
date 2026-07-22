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
│   ├── lab/                         Lab 实验结果与可视化
│   │   ├── 01_kmeans                ResNet18+CIFAR-100 特征聚类
│   │   ├── 02_head                  分类头、权重训练方式与 TensorShield Top-10 trainability 消融
│   │   ├── 03_baseline              普通 MS 策略、双黑盒参考线与 TEESlice 独立点总览
│   │   ├── 04_tensorshield          TensorShield 前缀、删除消融与位置窗口结果
│   │   ├── 05_state                 State 类型与参数语义分类结果
│   │   ├── 06_weight                遗漏 weight 语义闭包与多种子候选结果
│   │   ├── 07_bn                    BN gamma 分组 drop/add 结果
│   │   ├── 08_structure             结构、条件依赖、位置替换与局部配对结果
│   │   └── 09_leakage               泄露状态利用强度与 MS 负迁移结果
│   └── playground/                  编号探索的独立结果
│       ├── 01_raw                   40 项四路原始 weight 输出与残差乘积
│       ├── 02_rank                  all/main/bn 有效秩与秩乘积结果
│       ├── 03_feature               all/main/bn 特征图归一化残差乘积
│       ├── 04_param                 all/main/bn 参数量归一化残差乘积
│       └── 05_diagnose              BN/Conv 同源联合与跨归一化交叉保护诊断
├── models/                          统一模型结构及 TEESlice slice/backbone 接口
├── verify/                          环境/GPU、数据协议及 MS/Lab/Playground 结果验证
├── lab/                             小型验证实验
│   ├── 01_kmeans                    ResNet18+CIFAR-100 特征聚类
│   ├── 02_head                      分类头、权重训练方式与 TensorShield Top-10 trainability 消融
│   ├── 03_baseline                  MS 策略保护比例、双黑盒参考线与三项指标总览
│   ├── 04_tensorshield              TensorShield 前缀、删除消融与位置窗口验证
│   ├── 05_state                     State 类型与参数语义分类
│   ├── 06_weight                    遗漏 weight 语义闭包与多种子候选验证
│   ├── 07_bn                        BN gamma 分组 drop/add 验证
│   ├── 08_structure                 结构、条件依赖、位置替换与局部配对验证
│   └── 09_leakage                   泄露状态利用强度与 MS 负迁移验证
├── playground/                      未进入 Lab 或正式实验的编号探索
│   ├── 01_raw                       四路原始 weight 输出提取
│   ├── 02_rank                      all/main/bn 有效秩派生分析
│   ├── 03_feature                   all/main/bn 特征图归一化残差分析
│   ├── 04_param                     all/main/bn 参数量归一化残差分析
│   └── 05_diagnose                  BN/Conv 联合与交叉 Top-5 单种子诊断
└── docs/                            参考论文
```
