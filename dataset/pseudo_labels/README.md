# 伪标签数据集说明

本目录用于保存由 victim 模型对无标签查询集预测得到的伪标签数据集。顶层只保留本说明文件和四个数据集目录：

```text
dataset/pseudo_labels/
  README.md
  cifar10/
  cifar100/
  stl10/
  tiny-imagenet-200/
```

各数据集目录下按模型分层：

```text
dataset/pseudo_labels/<dataset>/
  resnet18/
  resnet50/
  vgg16_bn/
  mobilenetv2/
```

具体伪标签数据集直接保存在对应模型目录下：

```text
dataset/pseudo_labels/<dataset>/<model>/
  manifest.json
  samples.tsv
```

查询样本来自哪个验证来源、比例、样本数、随机种子和来源查询集路径等信息只写入 `manifest.json`，不编码进目录名。

## 生成方式

伪标签数据集由 `exp/make_pseudo_labels/` 中的入口生成。它读取 `dataset/derived/` 中已有的无标签查询集，再使用训练好的 victim 模型写出预测标签。

运行示例：

```bash
bash exp/make_pseudo_labels/resnet18/run.sh cifar10
bash exp/make_pseudo_labels/resnet50/run.sh tiny-imagenet-200
```

默认读取：

```text
dataset/derived/<dataset>/manifest.json
weights/victim/<model>/<dataset>/target.pth
```

默认输出：

```text
dataset/pseudo_labels/<dataset>/<model>/
  manifest.json
  samples.tsv
```

`samples.tsv` 只保存 `source_index`、`pseudo_label`、类别名和置信度，不保存真实标签，也不复制图像文件。后续实验需要图像时，应根据 `source_index` 回到 `dataset/public/` 中读取。
