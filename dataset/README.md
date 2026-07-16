# 数据集说明

本目录保存 TSDP 当前使用的公开数据、实验划分协议和由模型生成的标签产物。项目唯一接受的数据集 id 为：

```text
c10   CIFAR-10
c100  CIFAR-100
s10   STL10
t200  Tiny-ImageNet-200
```
下载或刷新公开数据：

```bash
bash dataset/download_datasets.sh all
```

## 目录分工

```text
dataset/public/      原始公开数据
dataset/MS/          Model Stealing 协议划分和模型标签产物
dataset/MIA/         Membership Inference Attack 协议预留目录，当前尚未展开
```

MS 当前使用 `reference_random_overlap` 协议：victim 使用官方训练集全量，query 从同一
训练集均匀随机无放回抽取固定 1%。具体样本索引、query 顺序和预算以
`dataset/MS/<dataset>/manifest.json` 与 `splits.tsv` 为准。

公开数据固定布局：

```text
dataset/public/c10/cifar-10-batches-py
dataset/public/c100/cifar-100-python
dataset/public/s10/stl10_binary
dataset/public/t200/
```


## 输入尺寸约定

```text
CIFAR-10           原始 32 x 32，训练输入 32 x 32
CIFAR-100          原始 32 x 32，训练输入 32 x 32
STL10              原始 96 x 96，训练输入 128 x 128
Tiny-ImageNet-200  原始 64 x 64，训练输入 224 x 224
```

数据目录只保留原始公开数据及必要整理结果；真正送入模型前的尺寸处理由训练脚本中的 transform 决定。

## CIFAR-10

路径：

```text
dataset/public/c10/cifar-10-batches-py
```

构成：训练集 50000 张，测试集 10000 张，类别数 10，图像尺寸 32 x 32 RGB。如果官方 Toronto 源可达，目录中会是官方 Python pickle 文件；当前脚本也支持在同一 canonical 目录下保存 `train/`、`test/` 图片 fallback，并由训练、查询和验证入口显式读取。

## CIFAR-100

路径：

```text
dataset/public/c100/cifar-100-python
```

构成：训练集 50000 张，测试集 10000 张，细类别数 100，粗类别数 20，图像尺寸 32 x 32 RGB。当前项目按 100 分类任务处理；如果使用图片 fallback，目录下会保存按 fine class 分层的 `train/` 和 `test/`。

## STL10

路径：

```text
dataset/public/s10/stl10_binary
```

构成：训练集 5000 张有标签图像，测试集 8000 张有标签图像，官方无标签集 100000 张，类别数 10，图像尺寸 96 x 96 RGB。当前监督训练只使用 `train` 和 `test` split；MS/MIA 协议不会使用官方 unlabeled split。

## Tiny-ImageNet-200

路径：

```text
dataset/public/t200
```

整理后的目录结构：

```text
t200/
  train/
  val/
  val2 -> val
  test/images/
  wnids.txt
  words.txt
```

训练集包含 200 个类别目录，每类 500 张图像；验证集包含 200 个类别目录，每类 50 张图像。下载脚本会把压缩包内部原始目录整理为 `dataset/public/t200`，并将验证集整理为 `ImageFolder` 可读取的类别目录。MS 的 `victim_train` 与 `query_pool_ms` 只使用 `train`，`eval_ms` 使用 `val`。
