# 实验 03：MS baseline 保护比例曲线

本实验汇总 `ResNet18+CIFAR-100` 的浅层保护、中间层保护、深层保护和全局大权重标量保护结果，绘制保护参数比例与 MS 原始指标的关系。

## 固定协议

```text
输入目录          results/MS/resnet18/c100/
攻击协议          posterior_replace_finetune_v2
query budget      500
主要 checkpoint   end.pth
横坐标            protected_param_ratio × 100%
曲线              shallow、middle、deep、large_weight
纵坐标            surrogate_acc、fidelity、posterior_kl
参考线            no_protection、full_protection 的对应 end 指标
分类头语义        large_weight 部分保护时按 mask 使用 mixed 初始化
```

脚本直接读取各语义 `artifact_id` 目录中的 `metrics.json`。每种策略必须恰好包含 8 个扫描点；横坐标使用 mask 实际保护的参数比例，不使用来源比例、官方层数或 unit 比例代替。各点按保护参数比例排序后直接连线，不做平滑、插值或派生指标计算。上下界只画为水平参考线，不并入四种策略的扫描序列。

## 运行方式

```bash
python3 lab/03_baseline/run.py
```

## 输出

```text
results/lab/03_baseline/accuracy.png       准确率曲线
results/lab/03_baseline/fidelity.png       相似度曲线
results/lab/03_baseline/posterior_kl.png   posterior KL 曲线
results/lab/03_baseline/data.tsv           绘图使用的原始点
results/lab/03_baseline/manifest.json      输入协议、artifact 与输出清单
```
