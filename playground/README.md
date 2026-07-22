# Playground

本目录保存尚未进入 `lab/` 或正式 `exp/MS/` 的 Model Stealing 探索。PG01–PG04
共享同一批 seed-42、500-query 原始四路输出，PG05 使用统一 soft-query 协议诊断
归一化 Top-5 的实际 MS 保护效果。

```text
01_raw      保存 20 个 Conv weight 与 20 个 BN gamma 的 z_pp/z_pv/z_vp/z_vv 和精确 I
02_rank     从 PG01 计算 all/main/bn 有效秩及 cross-rank × natural-rank
03_feature  按输出特征元素数归一化，分别给出 all/main/bn 的 cross × natural 排名
04_param    按 weight 参数量归一化，分别给出 all/main/bn 的保护效率代理排名
05_diagnose 比较 PG03/PG04 的 BN、Conv、同源联合及跨归一化交叉 Top-5
```

所有 bias 和最终分类层均不参与残差计算。`main` 固定为 16 个 BasicBlock 主路径 Conv，
`bn` 固定为 20 个 BN gamma；二者必须从相同 `all` 原始数据直接抽取，并在各自候选集
内重新排序。结果统一写入
`results/playground/<编号_关键词>/`，不进入正式 MS 索引，也不覆盖 Lab 结果。

PG05 固定保护分类头，但分类头不作为残差候选；八组仅使用 seed 42。未经用户明确
指示，不扩展到十随机种子。
