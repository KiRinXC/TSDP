.PHONY: help prepare verify datasets dataset download-datasets weights pre-train c10 c100 s10 t200

DATASETS ?= all
VERIFY_ARGS ?= --skip-archives --skip-query

help:
	@printf '%s\n' 'Available targets:'
	@printf '  %-18s %s\n' 'prepare' 'download public datasets and ImageNet pretrained weights; no training'
	@printf '  %-18s %s\n' 'verify' 'verify local public dataset layout; override with VERIFY_ARGS=""'
	@printf '  %-18s %s\n' 'datasets' 'download TensorShield datasets; override with DATASETS="c10 c100"'
	@printf '  %-18s %s\n' 'weights' 'download ImageNet pretrained weights only'
	@printf '  %-18s %s\n' 'c10' 'download CIFAR-10 only'
	@printf '  %-18s %s\n' 'c100' 'download CIFAR-100 only'
	@printf '  %-18s %s\n' 's10' 'download STL-10 only'
	@printf '  %-18s %s\n' 't200' 'download Tiny-ImageNet only'

prepare: datasets weights

verify:
	python3 verify/verify_datasets.py $(VERIFY_ARGS)
	python3 verify/verify_ms_splits.py

datasets download-datasets:
	bash dataset/download_datasets.sh $(DATASETS)

dataset: datasets

c10:
	bash dataset/download_datasets.sh c10

c100:
	bash dataset/download_datasets.sh c100

s10:
	bash dataset/download_datasets.sh s10

t200:
	bash dataset/download_datasets.sh t200

weights pre-train:
	bash weights/pre_train/download_pretrained_weights.sh
