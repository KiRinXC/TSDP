# 项目协作约定

## 文档语言

本项目中所有名为 `README.md` 的文件都必须使用中文撰写。

如果需要引用英文术语、命令、路径、类名、函数名、数据集原名或论文标题，可以保留英文原文；但说明性文字、章节标题和面向读者的描述应使用中文。

## 目录命名与项目结构

目录名优先保持简短、稳定、易输入。不要把完整实验说明、随机种子、比例、日期、超参数或长句编码进目录名；这些细节应写入就近的 `README.md`、`manifest.json`、`params.json` 或实验日志。

新增目录时遵循以下规则：

1. 顶层目录只表达职责边界，例如 `dataset/`、`exp/`、`lab/`、`models/`、`weights/`、`results/`、`docs/`、`verify/`。
2. 层级尽量少。只有当数据来源、产物类型或代码职责真的发生变化时，才新增一层目录。
3. 代码模块目录使用简短英文和 `snake_case`，例如 `train_victim`。数据集、模型名可以保留通用写法，例如 `cifar10`、`tiny-imagenet-200`、`resnet18`、`vgg16_bn`。
4. 正式实验脚本放在 `exp/`，小型验证和临时探索放在 `lab/`。实验目录可使用短编号加关键词，例如 `01_resnet18_cifar10`；详细目的写在该目录的 `README.md`。
5. 实验输出放在 `results/` 或 `weights/`，不要混入 `exp/`、`lab/` 或 `models/` 代码目录。可以被后续训练或评估直接消费的数据集产物应放在 `dataset/` 下的专门目录。
6. `dataset/public/` 只保存原始公开数据；`dataset/derived/` 只保存由公开数据派生出的索引、查询集等中间数据；`dataset/pseudo_labels/` 保存由 victim 模型生成的伪标签数据集。
7. `dataset/derived/` 顶层只保留 `README.md` 和四个数据集目录：`cifar10/`、`cifar100/`、`stl10/`、`tiny-imagenet-200/`。不要新增 `README/` 目录，也不要在顶层再新增 `unlabeled/`、`pseudo_labels/` 这类产物类型目录；具体产物类型和生成规则写入各数据集目录下的 manifest。派生数据文件直接放在对应 split 目录下，例如 `dataset/derived/cifar10/test/manifest.json` 和 `dataset/derived/cifar10/test/samples.tsv`；不要再用 `ratio-*`、`seed-*`、日期或 run name 增加额外子目录，这些元数据统一写入 `manifest.json`。
8. `dataset/pseudo_labels/` 顶层只保留 `README.md` 和四个数据集目录：`cifar10/`、`cifar100/`、`stl10/`、`tiny-imagenet-200/`。各数据集目录下按模型分层，例如 `resnet18/`、`resnet50/`、`vgg16_bn/`、`mobilenetv2/`。
9. 新增或调整目录结构时，同步更新最近一层的 `README.md`，让可读性主要来自文档而不是冗长路径。
10. 项目需要维护两个总览图：`STRUCTURE.md` 是目录结构图，只用目录树形式展示同级目录和下一层目录的作用；`FLOW.md` 是实验流程图，只用目录树形式展示当前实验主要做了什么、读写哪些关键产物。每次新增任何 `exp/` 或 `lab/` 实验，都必须完善这两个图；调整关键目录、改变实验输入输出或新增数据产物时，也必须同步更新这两个图。不能只新增实验代码和 README 而遗漏 `STRUCTURE.md` 与 `FLOW.md`。
11. 实验代码目录和结果目录都需要 `README.md` 时，应明确分工：实验代码目录的 `README.md` 说明实验目的、方法、运行方式和参数含义；结果目录的 `README.md` 只记录本次运行的结果文件、关键指标和必要解读，不重复实验方法说明。若用户已经删减结果说明，应以删减后的版本为准，不主动补回重复描述。
