# TSDP

新会话接手项目前先阅读 `HANDOFF.md`，其中记录当前有效协议、已完成结果、扩展边界和禁止重复的错误。

## 唯一运行环境

本项目只使用以下虚拟环境，不在系统 Python 或其他环境中运行实验：

```text
环境名       dl-py310-torch210-cu121
路径         ~/venvs/dl-py310-torch210-cu121
Python       3.10
PyTorch      2.1.0+cu121
CUDA runtime 12.1
```

`requirements.txt` 记录仓库直接依赖，`requirements.lock.txt` 固定完整解析环境。环境安装和验证统一使用：

```bash
make install
make env
make gpu
make unit
```

`make gpu` 必须在正式训练会话中通过。它会核对 WSL `/dev/dxg`、`nvidia-smi`、固定 PyTorch/CUDA 版本，并执行真实 CUDA 前向和反向计算。不要使用系统 `/usr/bin/python3` 运行 TSDP。

TSDP 当前实现 Model Stealing（MS）实验。受害者模型使用官方训练集全量训练，query pool 从同一训练集随机无放回抽取 1%。普通 victim 不另划 validation split，而是沿用 TensorShield 与 TEESlice 参考代码的训练风格，在官方 test/validation 对应的 `eval_ms` 上逐轮评估并保存 accuracy 最高的 `best.pth`，再用该 checkpoint 生成 posterior 和 hard pseudo label。正式 surrogate 在当前 query budget 内按 seed 42 与固定 offset 100 划分 80% query train 和 20% query validation，最多训练 100 轮，并按 validation cross-entropy 选择最早的最优 `best.pth`；`eval_ms` 不参与选模，只在 checkpoint 固定后完整评估一次。正式结果同时保留 soft-posterior `full_protection` 黑盒与 label-only `hard_blackbox` 黑盒，两者都必须进入汇总图。

MS 的基本流程：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/transfer/prepare_splits.py all
bash exp/MS/train_victim/resnet18/run.sh c100
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/transfer/get_label.py resnet18 c100
bash exp/MS/train_surrogate/run.sh resnet18 c100 \
  --defense full_protection \
  --budget 500 \
  --training-mode finetune \
  --label-mode soft
```

surrogate 阶段实现无保护、全保护、仅分类头保护、浅层、中间层、深层、大权重和 TensorShield baseline，并支持自定义 unit mask。TensorShield 的 `ResNet18+CIFAR-100` 方案直接使用作者确认 rank 派生的 Figure 12 固定集合，不再运行公式评分代码。当前正式协议固定使用全模型微调；分类头完整暴露时复制 victim 分类头，部分暴露时按 mask 混合随机初始化与 victim 标量，完整保护时使用替换头。保护范围使用明确的官方层或绝对 unit 索引而不是比例，每种策略统一保存为按模型 `state_dict` unit 排列的 `protection_mask.pt`；正式 surrogate 只保存 validation-best `best.pth`，训练日志不读取 `eval_ms`，最终保存 accuracy、fidelity、posterior KL 及其原始计数，派生对比指标留给绘图阶段计算。

TEESlice 属于先训练专用 defended victim、再实施攻击的独立 baseline，不使用普通 victim 的 122-unit mask。当前 `ResNet18+C100` 完整流程为：

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_victim/teeslice/train.py resnet18 c100
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/transfer/get_label.py teeslice_r18 c100
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" exp/MS/train_surrogate/teeslice/attack.py resnet18 c100 \
  --budget 500 --training-mode finetune --label-mode soft
```

目录职责见 `STRUCTURE.md`，完整数据与训练流程见 `FLOW.md`。
