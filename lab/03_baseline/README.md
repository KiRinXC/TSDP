# 实验 03：MS 策略保护比例总览

本实验汇总 `ResNet18+CIFAR-100` 的浅层保护、中间层保护、深层保护、全局大权重标量保护、分类头保护和 TensorShield 结果，并以独立标记展示 TEESlice standalone 复现，绘制保护参数比例与 MS 原始指标的关系。除正式 soft-posterior 黑盒外，三张图读取正式 `hard_blackbox` 输出能力对比结果作为辅助参考线；Lab03 不再承担 surrogate 训练。

## 固定协议

```text
输入目录          results/MS/resnet18/c100/
辅助输入          results/MS/resnet18/c100/hard_blackbox/metrics.json
攻击协议          posterior_replace_finetune_v2
query budget      500
主要 checkpoint   end.pth
横坐标            protected_param_ratio × 100%
曲线              shallow、middle、deep、large_weight
普通单点          head_only、TensorShield
独立单点          TEESlice standalone，不与普通 victim 曲线相连
纵坐标            surrogate_acc、fidelity、posterior_kl
主参考线          普通 victim 的 no_protection 与 soft-label full_protection end 指标
辅助参考线        hard-label full_protection end 指标，只作输出能力对比
分类头语义        large_weight 部分保护时按 mask 使用 mixed 初始化
```

脚本直接读取各语义 `artifact_id` 目录中的 `metrics.json`。四种扫描策略必须各自恰好包含 8 个点；横坐标使用实际保护参数比例，不使用来源比例、官方层数或 unit 比例代替。扫描点按保护参数比例排序后直接连线，不做平滑、插值或派生指标计算。`head_only` 与 TensorShield 作为普通固定 victim 单点展示；TEESlice 使用自身 private parameter ratio 和黑盒 surrogate `end` 指标，以 `standalone_reproduction` 独立标记展示。

主图的正式上下限仍采用普通 victim 的 `no_protection` 与 soft-posterior `full_protection`。Hard-label 黑盒使用相同 victim、500 条 query、全保护初始化、100 轮训练和 `end.pth`，只改变 query 输出为 argmax hard label；它标记为 `ordinary_fixed_victim_output_ablation`，只作辅助对比，不替换主黑盒下界。TEESlice 的白盒/黑盒结果同样不用于替换普通 victim 上下界，也不与普通策略连线。

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
results/lab/03_baseline/data.tsv           绘图使用的 38 个原始点及 label mode
results/lab/03_baseline/manifest.json      输入协议、artifact、hard-label 输入哈希与输出清单
```
