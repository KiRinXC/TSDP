.PHONY: help datasets dataset download-datasets cifar10 cifar100 stl10 tiny-imagenet

DATASETS ?= all

help:
	@printf '%s\n' 'Available targets:'
	@printf '  %-18s %s\n' 'datasets' 'download TensorShield datasets; override with DATASETS="cifar10 cifar100"'
	@printf '  %-18s %s\n' 'cifar10' 'download CIFAR-10 only'
	@printf '  %-18s %s\n' 'cifar100' 'download CIFAR-100 only'
	@printf '  %-18s %s\n' 'stl10' 'download STL-10 only'
	@printf '  %-18s %s\n' 'tiny-imagenet' 'download Tiny-ImageNet only'

datasets download-datasets:
	bash dataset/download_datasets.sh $(DATASETS)
