# victim 训练入口

本目录保存受害者模型训练入口，按四个模型拆分：

```text
mobilenetv2/
resnet18/
resnet50/
vgg16_bn/
```

公共训练逻辑位于 `common/trainer.py`。四个入口只负责指定模型结构和默认 ImageNet 预训练权重。

## 支持的数据集 id

```text
c10   CIFAR-10
c100  CIFAR-100
s10   STL10
t200  Tiny-ImageNet-200
```

公开数据从 `dataset/public/<dataset>/` 读取。

## 默认配置

```text
公开数据集根目录: dataset/public/
训练 split: dataset/MS/<dataset>/splits.tsv 中的 victim_train
输出目录: weights/MS/victim/<model>/<dataset>
batch size: 64
epochs: 100
learning rate: 0.1
momentum: 0.5
weight decay: 5e-4
lr step: 60
lr gamma: 0.1
num workers: 10
device: auto
```

MS victim 只读取 canonical `victim_train`。当前 `reference_random_overlap` 协议中，
该 split 覆盖官方训练集全量；`query_pool_ms` 是同一训练集随机无放回抽取的 1% 子集。
训练前先执行 `python3 exp/MS/transfer/prepare_splits.py all`。STL10 的 unlabeled split
不参与训练，Tiny-ImageNet 的 val/test 侧样本也不参与 victim 训练。

模型默认加载 `weights/pre_train/` 下的 ImageNet-1K 官方预训练权重，并替换最后分类层。

## 输入尺寸协议

```text
c10 / c100
  train: RandomCrop(32, padding=4) + RandomHorizontalFlip
  test : ToTensor + Normalize
  输入尺寸: 32 x 32

s10
  train: Resize(128) + RandomCrop(128, padding=4) + RandomHorizontalFlip
  test : Resize(128) + CenterCrop(128)
  输入尺寸: 128 x 128

t200
  train: Resize(256) + RandomCrop(224) + RandomHorizontalFlip
  test : Resize(256) + CenterCrop(224)
  输入尺寸: 224 x 224
```

## 使用方式

训练 `ResNet18 + c100`：

```bash
bash exp/MS/train_victim/resnet18/run.sh c100
```

其他示例：

```bash
bash exp/MS/train_victim/resnet50/run.sh c10
bash exp/MS/train_victim/vgg16_bn/run.sh c100
bash exp/MS/train_victim/mobilenetv2/run.sh s10
```

`run.sh` 的第一个参数就是数据集 id。也可以用环境变量覆盖：

```bash
DATASET=s10 EPOCHS=50 BATCH_SIZE=32 bash exp/MS/train_victim/vgg16_bn/run.sh
```

只检查路径和模型构造，不进入训练：

```bash
python3 exp/MS/train_victim/resnet18/train.py --dataset c100 --dry-run
```

恢复训练：

```bash
bash exp/MS/train_victim/resnet18/run.sh c100 --resume weights/MS/victim/resnet18/c100/best.pth
```

## 输出内容

默认输出目录为：

```text
weights/MS/victim/<model>/<dataset>/
```

其中包含：

```text
best.pth
end.pth
train.log.tsv
params.json
```
