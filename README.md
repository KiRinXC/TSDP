# TSDP

新会话接手项目前先阅读 `HANDOFF.md`，其中记录当前有效协议、已完成结果、扩展边界和禁止重复的错误。

TSDP 当前实现 Model Stealing（MS）实验。受害者模型使用官方训练集全量训练，query pool 从同一训练集随机无放回抽取 1%，并使用最佳验证模型生成 posterior 和 hard pseudo label。正式 MS baseline 统一使用 posterior-visible 查询接口和 soft posterior；posterior 生成与 surrogate query 均使用确定性 test transform，攻击训练固定 `lr_step=60`。hard label 仅用于 Lab 输出能力消融。

MS 的基本流程：

```bash
python3 exp/MS/transfer/prepare_splits.py all
bash exp/MS/train_victim/resnet18/run.sh c100
python3 exp/MS/transfer/get_label.py resnet18 c100
bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense full_protection \
  --budget 500 \
  --training-mode finetune \
  --label-mode soft
```

surrogate 阶段实现无保护、全保护、浅层、中间层、深层、大权重和 TensorShield baseline，并支持自定义 unit mask。TensorShield 的 `ResNet18+CIFAR-100` 方案直接使用作者确认 rank 派生的 Figure 12 固定集合，不再运行公式评分代码。当前正式协议固定使用全模型微调；分类头完整暴露时复制 victim 分类头，部分暴露时按 mask 混合随机初始化与 victim 标量，完整保护时使用替换头。保护范围使用明确的官方层或绝对 unit 索引而不是比例，每种策略统一保存为按模型 `state_dict` unit 排列的 `protection_mask.pt`；训练与评估只保存 accuracy、fidelity、posterior KL 及其原始计数，固定终点 `end.pth` 作为主结果，派生对比指标留给后续绘图阶段计算。

TEESlice 属于先训练专用 defended victim、再实施攻击的独立 baseline，不使用普通 victim 的 122-unit mask。当前 `ResNet18+C100` 完整流程为：

```bash
python3 exp/MS/train_victim/teeslice/train.py resnet18 c100
python3 exp/MS/transfer/get_label.py teeslice_r18 c100
python3 exp/MS/train_surrogate/teeslice/attack.py resnet18 c100 \
  --budget 500 --training-mode finetune --label-mode soft
```

目录职责见 `STRUCTURE.md`，完整数据与训练流程见 `FLOW.md`。
