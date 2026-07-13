# 实验 04 结果

本目录保存 TensorShield 作者确认 rank 的 Top-1 至 Top-12 MS 前缀曲线。所有主要结果均为相同初始化与 query 顺序下第 100 轮的 `end` 指标。

```text
k   新增 weight                    参数比例   accuracy  fidelity  posterior KL
1   layer1.1.conv1.weight           0.3283%     0.5983    0.8155       0.167813
2   layer2.0.conv1.weight           0.9850%     0.5493    0.6987       0.447348
3   last_linear.weight              1.4419%     0.4059    0.4657       1.456648
4   layer1.0.conv1.weight           1.7702%     0.3922    0.4435       1.535794
5   layer1.1.conv2.weight           2.0985%     0.3909    0.4432       1.523675
6   layer2.0.conv2.weight           3.4118%     0.3623    0.4076       1.669434
7   layer2.1.conv1.weight           4.7252%     0.3374    0.3742       1.783701
8   layer1.0.conv2.weight           5.0535%     0.3274    0.3644       1.804010
9   layer3.0.conv1.weight           7.6801%     0.2596    0.2821       2.137994
10  layer2.1.conv2.weight           8.9934%     0.2569    0.2785       2.167961
11  layer3.0.conv2.weight          14.2467%     0.2202    0.2355       2.332683
12  layer4.0.conv1.weight          24.7531%     0.1926    0.2120       2.453924
```

当前正式参考线为：无保护 accuracy `0.6182`、fidelity `1.0000`、KL 约为 `0`；全保护 accuracy `0.1545`、fidelity `0.1610`、KL `2.835290`。

Top-10 的保护 mask 与被替换的旧单点实验完全相同，逻辑 SHA256 均为 `1e3aa38124f084dd39eab42a4d3f1ddf1ca86807812796c66a8318c05e7aa2cb`。旧脚本在构造 victim 后继续使用已经推进的 RNG 状态初始化受保护分类头，得到 accuracy `0.1913`；不落盘复核可精确重现该值。当前曲线在每个 surrogate 初始化前立即重置种子，得到 Top-10 accuracy `0.2569`，从而保证 12 个 k 使用相同随机头起点和 query 顺序。当前受控曲线是 Lab 04 的唯一有效结果。

## 结果解读

曲线整体接近单调：从 Top-1 扩展到 Top-12 后，accuracy 从 `0.5983` 降至 `0.1926`，fidelity 从 `0.8155` 降至 `0.2120`，KL 从 `0.167813` 增至 `2.453924`。因此作者 rank 的前缀集合确实具有持续累积的 MS 防护效果。

k=3 新增分类头 weight，并同步保护 bias，产生全曲线最大跳变：相对 k=2，accuracy 降低 `14.34` 个百分点，fidelity 降低 `23.30` 个百分点，KL 增加 `1.009300`。这说明前缀早期效果很大程度上受到分类头联动规则影响，不能把该跳变单独解释为一个普通 tensor 的排序优势。

Figure 12(d) 对应的 k=10 尚未进入稳定平台。k=10 相比 k=9 只使 accuracy 降低 `0.27` 个百分点、fidelity 降低 `0.36` 个百分点、KL 增加 `0.029968`；排名更低的 k=11 却分别改善 `3.67`、`4.30` 个百分点和 `0.164722` KL。继续到 k=12 后，accuracy 又降低 `2.76` 个百分点。k=5 的 KL 还比 k=4 低 `0.012119`，并非严格单调。

因此，本实验支持的是“作者 Top-k 前缀集合随 k 增大总体有效”，不能证明 17 个 eligible tensor 的精细先后顺序严格对应其边际 MS 防护贡献。若要直接证明排序准确性，下一项必要对照是在固定 k 和保护成本下，将 Top-k 与随机 k 个 tensor 或用低排名 tensor 替换边界项进行比较。

k=12 与全保护仍有 `3.81` 个百分点 accuracy、`5.10` 个百分点 fidelity 和 `0.381367` KL 的差距，但只保护 `24.7531%` 参数，已经接近当前全保护攻击下界。

```text
metrics.json       作者 rank、输入哈希、12 组保护统计与 end 原始指标
history.tsv       1,200 条 query 训练记录，不包含中途 eval_ms 指标
data.tsv          Top-k 曲线的原始绘图数据
metrics.png       accuracy、fidelity、posterior KL 三联曲线
top_01_mask.pt    Top-1 紧凑保护掩码
...
top_12_mask.pt    Top-12 紧凑保护掩码
```

## Rank-5/Rank-10 冗余消融

```text
方案          保护参数比例  accuracy  fidelity  posterior KL
完整 Top-10       8.9934%     0.2569    0.2785       2.167961
删除 rank-5       8.6651%     0.2536    0.2791       2.166214
删除 rank-10      7.6801%     0.2596    0.2821       2.137994
同时删除 5/10     7.3518%     0.2593    0.2831       2.145638
```

删除 rank-5 `layer1.1.conv2.weight` 后，accuracy 反而降低 `0.33` 个百分点，fidelity 只提高 `0.06` 个百分点，KL 只降低 `0.001748`。三项变化没有形成一致且有实际幅度的攻击增益，说明 rank-5 在完整 Top-10 中没有表现出独立的 MS 保护贡献。

删除 rank-10 `layer2.1.conv2.weight` 后，accuracy 和 fidelity 分别提高 `0.27`、`0.36` 个百分点，KL 降低 `0.029968`，存在方向一致但较小的保护贡献。该结果与前缀曲线 k=9→10 的小幅变化一致。

同时删除两项后，accuracy 仅提高 `0.24` 个百分点，fidelity 提高 `0.46` 个百分点，KL 降低 `0.022324`。保护参数由 `1,009,764` 降至 `825,444`，减少 `184,320` 个参数，即节省完整 Top-10 保护成本的 `18.25%`，但攻击效果仍接近完整集合。这表明两项的贡献没有叠加，并在当前集合中存在明显功能冗余。

因此，本实验进一步说明 TensorShield 分数不等价于“加入完整保护集合后的边际 MS 防护贡献”：一个 tensor 可以拥有较高 rank，却在已有高排名 tensor 同时受保护时几乎不改变攻击结果。该结论限定于当前模型、数据集、攻击协议和固定种子；若作为论文中的统计等价性结论，仍需增加多个训练种子或预先定义等价区间。

```text
ablation.json          四组集合、相对完整 Top-10 的差值与输入哈希
ablation.tsv           可直接绘图和统计的原始指标
ablation_history.tsv   drop-5 与 drop-5/10 两组共 200 轮 query 训练记录
ablation.png           accuracy、fidelity 与 posterior KL 三联柱状图
drop_05_mask.pt        删除 rank-5 的紧凑保护掩码
drop_05_10_mask.pt     同时删除 rank-5/rank-10 的紧凑保护掩码
```

## 原始 rank 窗口消融

两个窗口严格取自作者确认的原始 41-weight rank，分别为第 11 至 20 项和末尾第 32 至 41 项；每组只运行一次完整训练。

```text
方案          ranked weight  保护 unit  参数比例  head mode  accuracy  fidelity  posterior KL
rank 11-20              10         11    1.9384%  replace      0.3337    0.3731       1.802660
rank 32-41              10         10   89.3142%  exposed      0.3133    0.3384       1.860727
```

末尾窗口相对 rank 11-20 使 accuracy 降低 `2.04` 个百分点、fidelity 降低 `3.47` 个百分点、KL 增加 `0.058067`，但保护参数从 `217,636` 增至 `10,028,032`，约为前者的 `46.08` 倍。也就是说，末尾 10 项依靠覆盖大量深层卷积参数获得了稍强的 MS 保护，但保护成本效率明显较低。

该差异还受到分类头状态影响：rank 11-20 包含 `last_linear.weight`，并联动保护 bias、使用 `replace`；末尾窗口不含分类头，攻击者可以复制 victim 分类头。因此不能把两组差值解释为严格的等成本排序优势。结合已有 Figure 12 eligible Top-10 结果（参数比例 `8.9934%`、accuracy `0.2569`、fidelity `0.2785`、KL `2.167961`），作者最终筛选集合在远低于末尾窗口的参数成本下取得了更强保护，但其中仍包含前述 rank-5/rank-10 条件冗余。

```text
window.json          两个原始 rank 窗口、保护统计、输入哈希和 end 原始指标
window.tsv           两组保护成本与三项 MS 原始指标
window_history.tsv   两组各 100 轮、共 200 轮 query 训练记录
window.png           accuracy、fidelity 与 posterior KL 三联柱状图
rank_11_20_mask.pt   原始 rank 11-20 的紧凑保护掩码
rank_32_41_mask.pt   原始 rank 32-41 的紧凑保护掩码
```
