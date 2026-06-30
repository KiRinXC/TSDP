.PHONY: help prepare verify datasets dataset download-datasets weights pre-train cifar10 cifar100 stl10 tiny-imagenet

DATASETS ?= all
VERIFY_ARGS ?= --skip-archives

help:
	@printf '%s\n' 'Available targets:'
	@printf '  %-18s %s\n' 'prepare' 'download public datasets and ImageNet pretrained weights; no training'
	@printf '  %-18s %s\n' 'verify' 'verify local public dataset layout; override with VERIFY_ARGS=""'
	@printf '  %-18s %s\n' 'datasets' 'download TensorShield datasets; override with DATASETS="cifar10 cifar100"'
	@printf '  %-18s %s\n' 'weights' 'download ImageNet pretrained weights only'
	@printf '  %-18s %s\n' 'cifar10' 'download CIFAR-10 only'
	@printf '  %-18s %s\n' 'cifar100' 'download CIFAR-100 only'
	@printf '  %-18s %s\n' 'stl10' 'download STL-10 only'
	@printf '  %-18s %s\n' 'tiny-imagenet' 'download Tiny-ImageNet only'

prepare: datasets weights

verify:
	python3 verify/verify_datasets.py $(VERIFY_ARGS)

datasets download-datasets:
	bash dataset/download_datasets.sh $(DATASETS)

dataset: datasets

cifar10:
	bash dataset/download_datasets.sh cifar10

cifar100:
	bash dataset/download_datasets.sh cifar100

stl10:
	bash dataset/download_datasets.sh stl10

tiny-imagenet:
	bash dataset/download_datasets.sh tiny-imagenet

weights pre-train:
	bash weights/pre_train/download_pretrained_weights.sh
