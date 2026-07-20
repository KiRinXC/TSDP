# 环境与代码验证

本目录保存不会启动正式训练的验证入口。TSDP 只认以下唯一 Python 环境：

```text
~/venvs/dl-py310-torch210-cu121
```

仓库根目录提供统一命令：

```bash
make install     # 按 requirements.lock.txt 补齐或复现唯一环境
make env         # 检查 Python 与依赖版本，允许当前会话暂时看不到 GPU
make gpu         # 严格检查 WSL GPU，并执行 CUDA 矩阵乘法和卷积反向传播
make unit        # 运行 MS、TensorShield、TEESlice 的 43 项单元测试
make results     # 核对正式 MS 及 Lab 的指标、日志、mask、输入哈希和图片
make check       # 依次运行 GPU、单元测试、数据协议和结果一致性检查
```

其中 Lab 验证会拒绝以下情况：Lab03 没有使用普通 ResNet18 统一参数分母；
Lab02/04/05/06 不是 400/100 query
train/validation、主结果不是最早的 validation-best、单个 case 多次读取
`eval_ms` 或 mask 文件与 JSON 哈希不一致。Lab04 candidate 还会核对 seed 43–52
的十组独立 query 划分、五十组完整 history、四策略受控配对及黑盒判定、均值和
样本标准差。
Lab07 还会核对五个 `conv1.weight` 的 seed 43–52 leave-one-out 笛卡尔积、六组
固定 mask 与保护成本、50 组共 5,000 轮 history、逐 seed 配对差值及 Lab04
基础集合/黑盒来源哈希；同时核对将五个 `conv1` 一一换成对应 `conv2` 的十种子
直接保护对照、1,000 轮 history、配对统计和 conv2 mask，并拒绝已完成实验残留
的进度文件。
Lab08 会核对五个利用强度 × 十 seed、三十组共 3,000 轮新增 history、五十个
epoch-0 探针、固定 5.7529% 系统保护集合、0%/100% 的 Lab04 来源、逐 seed
黑盒配对以及只按 query-validation loss 产生的适应性强度选择。
Lab09 会核对五十个分类头范数反事实、七组 1,280 个 oracle-reveal 端点及 Shapley
闭合、八块 480 个 conv/BN 接口干预、四类 BN gamma 的 160 个组合端点、Lab07
五点相关和全部来源哈希，并拒绝把该后验机制分析标为选择器或 MS 训练结果。

`make verify` 默认检查四个公开数据集的 canonical layout 和 `dataset/MS/` 划分协议，但跳过公开数据压缩包的 MD5。需要同时核对压缩包时使用 `make verify VERIFY_ARGS=""`。旧的 `dataset/query/` 已退出当前协议，不再由验证器读取；MS query 只由 `dataset/MS/<dataset>/manifest.json` 指向 `splits.tsv` 中的 `query_pool_ms`。

`make gpu` 默认要求以下条件全部成立：

```text
Python                         3.10
virtualenv                     dl-py310-torch210-cu121
PyTorch                        2.1.0+cu121
torchvision / torchaudio       0.16.0+cu121 / 2.1.0+cu121
torch.version.cuda             12.1
WSL /dev/dxg                   存在且当前用户可读写
nvidia-smi                     能识别至少一块 GPU
torch.cuda.is_available()      True
CUDA 前向与反向计算            结果为有限值
```

WSL 中 `nvidia-smi` 显示的 `CUDA Version` 是 Windows 驱动能够支持的最高 CUDA 版本，不要求与 `torch.version.cuda` 完全相同。较新的驱动可以运行 CUDA 12.1 构建的 PyTorch，因此不要为了让两个数字相同而降级驱动或重装 PyTorch。

如果普通 WSL 终端中的 `nvidia-smi` 正常，但某个受限工具会话提示 `GPU access blocked` 或缺少 `/dev/dxg`，问题属于该工具没有映射 GPU 设备，不是 TSDP 环境损坏。正式训练前必须在实际训练会话中重新运行 `make gpu`。

WSL 使用 Windows 主机侧 NVIDIA 驱动提供 CUDA 桥接，不要在 Ubuntu 内安装 Linux NVIDIA 显示驱动。`nvidia-smi` 不在 `PATH` 时，验证脚本会自动尝试 `/usr/lib/wsl/lib/nvidia-smi`。
