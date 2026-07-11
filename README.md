# TSDP

TSDP 当前实现 Model Stealing（MS）实验。受害者模型使用官方训练集全量训练，query pool 从同一训练集随机无放回抽取 1%，并使用最佳验证模型生成 posterior 和 hard pseudo label。

MS 的基本流程：

```bash
python3 exp/MS/transfer/prepare_splits.py all
bash exp/MS/train_victim/resnet18/run.sh c100
python3 exp/MS/transfer/get_label.py resnet18 c100
bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense full_protection \
  --budget 500 \
  --training-mode frozen \
  --label-mode hard
```

surrogate 阶段实现无保护、全保护、浅层、中间层、深层和大权重六种 baseline，并支持自定义 unit mask 以及暴露权重冻结或共同微调。保护范围使用明确的官方层或绝对 unit 索引而不是比例，每种策略统一保存为按模型 `state_dict` unit 排列的 `protection_mask.pt`；训练与评估只保存 accuracy、fidelity、posterior KL 及其原始计数，派生对比指标留给后续绘图阶段计算。

目录职责见 `STRUCTURE.md`，完整数据与训练流程见 `FLOW.md`。
