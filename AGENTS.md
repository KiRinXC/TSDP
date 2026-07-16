# 项目协作约定

## 文档语言

本项目中所有名为 `README.md` 的文件都必须使用中文撰写。

如果需要引用英文术语、命令、路径、类名、函数名、数据集原名或论文标题，可以保留英文原文；但说明性文字、章节标题和面向读者的描述应使用中文。



## 目录命名与项目结构

目录名优先保持简短、稳定、易输入。不要把完整实验说明、随机种子、比例、日期、超参数或长句编码进目录名；这些细节应写入就近的 `README.md`、`manifest.json`、`params.json` 或实验日志。

新增目录时遵循以下规则：

1. 顶层目录只表达职责边界，例如 `dataset/`、`exp/`、`lab/`、`models/`、`weights/`、`results/`、`docs/`、`verify/`。
2. 层级尽量少。只有当数据来源、产物类型或代码职责真的发生变化时，才新增一层目录。有不确定是否需要新增目录的时候，需要你向我询问。
3. 代码模块目录使用简短英文和 `snake_case`，例如 `train_victim`。数据集、模型名可以保留通用写法，例如 `c10`、`t200`、`resnet18`、`vgg16_bn`。
   新增或重命名目录时，单个目录名最多只允许出现一个下划线 `_`；例如 `get_auxiliary`、`train_victim`、`vgg16_bn` 可以使用，`make_pseudo_labels` 这类包含两个以上下划线的目录名不得新增或继续作为正式入口。
   MS 作为 Model Stealing 缩写时使用大写，例如 `exp/MS/`、`weights/MS/`、`results/MS/` 和 `dataset/MS/`；不要新增小写 `ms` 目录作为正式入口或结果目录。
   MIA 作为 Membership Inference Attack 缩写时使用大写，例如 `dataset/MIA/`、`exp/MIA/`、`weights/MIA/` 和 `results/MIA/`；不要新增小写 `mia` 目录作为正式入口或结果目录。
   MS 中训练得到的攻击代理模型统一称为 `surrogate`；MIA 中用于模拟目标分布或训练攻击器的影子模型统一称为 `shadow`。文档、目录、变量和结果字段不要把两者混用。
4. 正式实验脚本放在 `exp/`，小型验证和临时探索放在 `lab/`。实验目录可使用短编号加关键词，例如 `01_resnet18_c10`；详细目的写在该目录的 `README.md`。
5. exp中的实验输出放在 `results/` 或 `weights/`，不要混入 `exp/`、`lab/` 或 `models/` 代码目录。可以被后续训练或评估直接消费的数据集产物应放在 `dataset/` 下的专门目录。
6. `dataset/public/` 只保存原始公开数据；`dataset/MS/` 保存 MS 协议下的 split、labels、posteriors 等派生产物；`dataset/MIA/` 预留给 MIA 协议下的 target/shadow split、labels、posteriors 等派生产物。query 不单独保存为 `query.tsv`，而是在协议 `manifest.json` 中通过 `query.split` 指向 `splits.tsv` 内的指定 split。`dataset/query/` 和 `dataset/auxiliary/` 不作为正式入口或正式结果目录。
7. `dataset/MS/` 顶层只保留 `README.md` 和四个数据集目录：`c10/`、`c100/`、`s10/`、`t200/`。各数据集目录直接保存 `manifest.json` 和 `splits.tsv`；模型相关标签产物按模型分层，例如 `dataset/MS/c100/resnet18/manifest.json`、`labels.tsv` 和可选 `posteriors.pt`。MS 的 query 只能由 manifest 指向 `query_pool_ms`，不得新增独立 `query.tsv`。不要再用 `ratio-*`、`seed-*`、日期、预算或 run name 增加额外子目录，这些元数据统一写入 manifest。
8. `dataset/MIA/` 顶层只保留 `README.md` 和四个数据集目录：`c10/`、`c100/`、`s10/`、`t200/`。各数据集目录直接保存 `manifest.json` 和 `splits.tsv`；模型相关标签产物按模型分层，例如 `dataset/MIA/c100/resnet18/manifest.json`、`labels.tsv` 和可选 `posteriors.pt`。MIA 的 query 只能由 manifest 指向 `shadow_train`，不得从 `target_train` 构造，也不得新增独立 `query.tsv`。
9. 新增或调整目录结构时，同步更新最近一层的 `README.md`，让可读性主要来自文档而不是冗长路径。
10. 项目需要维护两个总览图：`STRUCTURE.md` 是目录结构图以结构树的方式，只用目录树形式展示同级目录和下一层目录的作用；`FLOW.md` 是实验流程图，只用目录树形式展示当前实验主要做了什么、读写哪些关键产物。每次新增任何 `dataset/` `exp/` 或 `lab/` 实验，都必须完善这两个图；调整关键目录、改变实验输入输出或新增数据产物时，也必须同步更新这两个图。不能只新增实验代码和 README 而遗漏 `STRUCTURE.md` 与 `FLOW.md`。
11. 实验代码目录和结果目录都需要 `README.md` 时，应明确分工：实验代码目录的 `README.md` 说明实验目的、方法、运行方式和参数含义，在撰写的时候尽量不要说什么测试一下能不能启动，而是只给我完整的启动流程，不做冒烟测试；结果目录的 `README.md` 只记录本次运行的结果文件、关键指标和必要解读，不重复实验方法说明。若用户已经删减结果说明，应以删减后的版本为准，不主动补回重复描述。


## 实验可复现性
- 随机种子统一采用 42


## 失效内容清理

1. 项目工作树只保留当前有效且仍被实验流程消费的代码、数据、权重、结果、索引和文档。确认不再需要的实现或产物应直接删除；同一实验的新实现或新结果应修改或覆盖原有语义入口，不通过并存副本完成迁移。
2. 不为已被替代的内容新增 `v1/`、`v2/`、日期、`old/`、`backup/` 或其他历史归档目录，也不保留已失效的兼容入口、重复脚本、缓存和中间文件。需要查看或恢复历史时使用版本控制。
3. 清理前必须确认目标不再被当前代码、文档、manifest、索引、绘图或后续实验消费；仍然有效且承担独立实验结论的 Lab 代码和结果不属于失效内容。


## 正式实验协议与产物

1. 每次运行 `exp/` 下的正式实验前，必须先在该实验最近一层的 `README.md` 中固化本次协议。协议至少应明确数据划分与输入产物、模型初始化、攻击者可观测输出、标签模式、query budget、保护策略、训练方式、主要评估 checkpoint 和原始指标。不得先运行实验、再根据结果补写协议。
2. 正式实验启动前必须核对代码实现、命令行默认值、运行命令和 `README.md` 协议一致。若不一致，应先修改代码或文档并完成验证；不能以命令行临时覆盖但不更新文档的方式运行主实验。
3. 正式协议发生变化后，由旧协议生成且不再被当前流程消费的 `weights/`、`results/`、索引记录和中间数据均视为失效产物。失效产物应从正式目录和汇总索引中删除，不保留为历史版本，以免参与后续汇总、选模或绘图。需要保留的协议信息应写入文档或版本控制，不通过并存旧结果目录保存。
4. `lab/` 同样遵循全项目的失效内容清理规则。正式协议可以引用仍有效的 Lab 结论，但不得把 Lab 结果混入正式主实验汇总。
