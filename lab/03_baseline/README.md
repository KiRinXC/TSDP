# 实验 03：MS 策略保护比例总览

本实验汇总 `ResNet18+CIFAR-100` 的浅层保护、中间层保护、深层保护、全局大权重标量保护、分类头保护和 TensorShield 结果，并以独立标记展示 TEESlice standalone 复现，绘制保护参数比例与 MS 原始指标的关系。三张图同时展示正式 soft-posterior 黑盒与 hard-label 黑盒；Lab03 只消费正式结果，不承担 surrogate 训练。

## 固定协议

```text
输入目录          results/MS/resnet18/c100/
双黑盒输入        full_protection/metrics.json、hard_blackbox/metrics.json
soft 攻击协议     soft_query_validation_best_v1
hard 攻击协议     hard_query_validation_best_v1
query budget      500
query train/val   400/100，seed 42，offset 100
主要 checkpoint   validation loss 最低的 best.pth；并列取更早 epoch
横坐标            baseline_normalized_param_ratio × 100%
统一分母          普通 ResNet18+CIFAR-100 的 11,227,812 个可训练参数
曲线              shallow、middle、deep、large_weight
普通单点          head_only、TensorShield
独立单点          TEESlice standalone，不与普通 victim 曲线相连
纵坐标            surrogate_acc、fidelity、posterior_kl
白盒参考线        普通 victim 的 no_protection epoch-0 恒等指标
soft 黑盒参考线   full_protection 的 validation-best 指标
hard 黑盒参考线   hard_blackbox 的 validation-best 指标
分类头语义        large_weight 部分保护时按 mask 使用 mixed 初始化
```

脚本直接读取各语义 `artifact_id` 目录中的新协议 `metrics.json.result`。四种扫描策略必须各自恰好包含 8 个点；扫描点按统一保护参数比例排序后直接连线，不做平滑或插值。`head_only` 与 TensorShield 作为普通固定 victim 单点展示；TEESlice 使用 validation-best 黑盒 surrogate 指标，以 `standalone_reproduction` 独立标记展示。

普通策略的模型结构相同，原生 `protected_param_ratio` 已经使用普通 ResNet18 的
`11,227,812` 个参数作为分母。TEESlice 会在公开 backbone 上增加 proxy slice、
路径 alpha 和私有分类头，所以它自身报告的 private parameter ratio 使用更大的
TEESlice 总参数作为分母。为使同一横轴可比，`data.tsv` 同时保留：

```text
protected_param_ratio              各方法自身定义下的原生比例
native_private_param_ratio         仅 TEESlice 填写的原生私有参数比例
baseline_normalized_param_ratio    受保护参数 / 11,227,812，主图统一使用
```

最终剪枝 TEESlice 保护 `703,092` 个可训练参数，原生比例为 `5.9223%`；换用普通
ResNet18 统一分母后为 `6.2621%`。这只调整 Lab03 的跨方法横坐标，不修改
TEESlice 正式 `metrics.json` 中忠于其自身架构的成本定义。

soft 与 hard 都是正式黑盒边界。二者使用相同 victim、500 条 query 的固定 400/100 划分、完整保护初始化和最多 100 轮训练；区别只在于 soft 黑盒读取 posterior 并按 validation soft cross-entropy 选模，hard 黑盒读取 argmax label 并按 validation hard cross-entropy 选模。soft 黑盒是所有部分保护策略的同接口直接对照，hard 黑盒展示 label-only 查询能力下的边界。TEESlice 的白盒/黑盒结果不替换普通 victim 的边界，也不与普通策略连线。

## 运行方式

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/03_baseline/run.py
```

## 输出

```text
results/lab/03_baseline/accuracy.png       准确率曲线
results/lab/03_baseline/fidelity.png       相似度曲线
results/lab/03_baseline/posterior_kl.png   posterior KL 曲线
results/lab/03_baseline/metrics.png        三项原始指标的统一三联图
results/lab/03_baseline/data.tsv           38 个原始点、双参数比例、label mode 与选模 epoch
results/lab/03_baseline/manifest.json      输入协议、统一分母、artifact 与双黑盒定义
```
