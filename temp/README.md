# 攻击可恢复性残差割验证

本目录验证 Attack-Recoverability Cut（ARC）。它是临时科研验证，不属于
`exp/` 或 `lab/` 正式入口，也不会写入正式 `results/`、`weights/` 或主
`metrics.tsv`。

上一轮按一般 victim-public 功能残差选择节点，在 5% producer-only 保护下只得到
`0.4083/0.4679/1.498233`；增加直接 consumer 输入切片后，9.7901% 保护也只有
`0.4066/0.4658/1.511940`。这些实现和产物已经删除。本实验不再扩大闭包，而是把
选块目标改为攻击者训练后的恢复能力。

## 单一目标

设 `m_j=1` 表示保护通道块 `j`。内部 shadow surrogate 的有效参数为：

```text
theta_shadow = theta_public
             + (1 - m) * (theta_victim - theta_public)
             + delta_attack
```

分类头始终完整保护，因此从目标类别随机初始化开始，只通过 `delta_attack` 学习。
内部攻击者在 discovery query 上最小化 soft-posterior cross-entropy；门在攻击者未
查询的 discovery holdout 上最大化同一个损失。soft cross-entropy 与
`KL(victim || shadow)` 只相差固定的 victim posterior entropy，因此选块没有拼接
其他 XAI、权重或激活评分。

优化采用 first-order 交替近似：每个 query batch 先更新一次攻击者，再用一个互不
重叠的 holdout batch 更新一次门。门使用温度退火的 sigmoid、固定参数预算投影和
二值化正则，最终一次性硬化为全局静态掩码。内部 shadow 选择阶段固定 BN 运行状态
并使用 eval forward；由于 PyTorch 不支持对 BN `running_mean/var` 求梯度，选择阶段
对二者使用 stop-gradient，但硬化后仍随配对通道一起保护。最终 MS 验证仍使用当前
正式 train-mode BN 协议。

## 图对齐比例块

候选节点建立在 ResNet18 的 feature graph 上，而不是 122 个 tensor unit：

```text
stem 输出                    1 组
8 个 BasicBlock 的 conv1     8 组
8 个 BasicBlock 的残差输出    8 组
合计                         17 组
```

每组选择能整除通道数的 2 的幂作为块大小，使块数量最接近 16：

| 输出通道数 | 块大小 | 块数量 |
|---:|---:|---:|
| 64 | 4 | 16 |
| 128 | 8 | 16 |
| 256 | 16 | 16 |
| 512 | 32 | 16 |

因此共有 272 个全局候选块。downsample 的相同输出通道范围与对应 BasicBlock 残差
输出共享一个门，不作为游离节点。一个块保护 producer Conv 输出 filters、配对 BN
affine 与运行状态；分类头 weight 和 bias 固定完整保护。所有块进入同一个实际参数
预算，不要求每层被选中。

## 数据隔离

节点选择不读取正式 `query_pool_ms` 的 posterior 或任何 MS 结果。首先从
`victim_train` 中排除全部 500 个正式 `query_pool_ms` 索引，再按 seed 42 固定抽取：

```text
discovery query       500 个样本，仅供内部 shadow 更新
discovery holdout    4096 个样本，仅供保护门更新
正式 query_pool_ms    500 个样本，只在掩码固定后运行最终 MS
正式 eval_ms        10000 个样本，只在最终 MS 评估时读取
```

三个集合的 source index 必须互不重叠。discovery 数据只使用 victim 在线生成的
posterior，不保存或读取正式 `dataset/MS/c100/resnet18/posteriors.pt`。

## 固定协议

```text
模型与数据集           ResNet18 + CIFAR-100
victim                 weights/MS/victim/resnet18/c100/best.pth
公开初始化             weights/pre_train/resnet18-5c106cde.pth
随机种子               42
随机初始化轨迹         formal_victim_then_public_v1
总参数保护上限         8%，包含完整分类头
候选块                 17 组 × 16 块，共 272 块
shadow discovery       100 epoch，query/holdout batch size 64
shadow 优化器          SGD，lr=0.01，momentum=0.5，weight_decay=5e-4
shadow 调度            第 60 轮后学习率乘 0.1
门优化器               Adam，lr=0.02
门温度                 2.0 退火到 0.1
门二值正则             0.01 × 参数成本加权的 m(1-m)
正式 MS query          query_pool_ms 固定前 500 个样本
正式 MS 输出           victim 完整 softmax posterior
正式 MS 训练           SGD，100 epoch，batch size 64，lr=0.01
正式 MS 调度           第 60 轮后学习率乘 0.1
正式主评估点           第 100 轮 end
```

8% 点低于 TensorShield 的 8.9934%，先用于判断攻击恢复目标能否达到相近安全效果。
只有该点有效，后续才继续压缩到 5% 和 3%。

## 运行

只核对输入哈希、图块、参数覆盖、数据隔离和预算，不训练或写产物：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  "$HOME/venvs/dl-py310-torch210-cu121/bin/python" temp/run.py --dry-run
```

完整运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  "$HOME/venvs/dl-py310-torch210-cu121/bin/python" temp/run.py
```

如果选块已经完成而最终 MS 因进程中断，可只消费已固定且摘要核对通过的掩码恢复：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  "$HOME/venvs/dl-py310-torch210-cu121/bin/python" temp/run.py --final-only
```

产物全部写入 `temp/output/`：

```text
output/
├── mask.pt          ARC 静态保护掩码
├── selection.json   图块、数据隔离、选择协议与最终块清单
├── selection.tsv    每轮 shadow/门优化记录
├── attack.tsv       每轮正式 surrogate query 训练记录
├── end.pth          第 100 轮临时 surrogate checkpoint，仅本地生成并由 Git 忽略
└── metrics.json     正式 MS end 指标及现有基线比较
```

## 本轮结果

8% 单点已经按上述协议完成。硬化后选择 37/272 个块，分布在 17 个 feature group
中的 10 组；它没有要求每层都有块。分类头与通道块共保护 897,796/11,227,812 个
参数，即 7.9962%。掩码 SHA256 为
`be1a5bb25a7401ce0db7f0d91dfd01a48c434b48cd157be7b54b7ec6ff91c1a5`。

| 策略 | 参数保护比例 | MS accuracy | fidelity | posterior KL |
|---|---:|---:|---:|---:|
| full protection（soft black-box） | 100% | 0.1545 | 0.1610 | 2.835290 |
| TensorShield | 8.9934% | 0.1913 | 0.2099 | 2.505831 |
| ARC（本轮） | 7.9962% | 0.2694 | 0.2968 | 2.133637 |
| head only | 0.4569% | 0.4404 | 0.5135 | 1.347578 |

ARC 明显强于只保护分类头，但没有达到 TensorShield 或 soft black-box 水平，因此
当前规则未通过 8% scientific gate，不继续跑 5% 和 3%。这不能否定攻击依赖路径
本身：本轮只把可训练门放在计算图对齐节点上，并未建立显式边变量、残差注入传播
或连通路径约束。它否定的是“独立节点门 + first-order recoverability 目标已经足够”
这一具体规则。后续若继续，应先把门改为由计算图边传播得到的静态路径，再重新做同一
8% 单点，不能直接增加多个代理分数。
