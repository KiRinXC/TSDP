# TSDP

TSDP 当前实现 Model Stealing（MS）实验。受害者模型使用官方训练集全量训练，query pool 从同一训练集随机无放回抽取 1%，并使用最佳验证模型生成伪标签。

MS 的基本流程：

```bash
python3 exp/MS/transfer/prepare_splits.py all
bash exp/MS/train_victim/resnet18/run.sh c100
python3 exp/MS/transfer/get_label.py resnet18 c100
```

目录职责见 `STRUCTURE.md`，完整数据与训练流程见 `FLOW.md`。
