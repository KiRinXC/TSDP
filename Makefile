.PHONY: help install env gpu unit results check prepare verify datasets dataset download-datasets weights pre-train c10 c100 s10 t200

ENV_NAME ?= dl-py310-torch210-cu121
VENV ?= $(HOME)/venvs/$(ENV_NAME)
PYTHON ?= $(VENV)/bin/python
REQUIREMENTS ?= requirements.lock.txt
DATASETS ?= all
VERIFY_ARGS ?= --skip-archives

help:
	@printf '%s\n' 'Available targets:'
	@printf '  %-18s %s\n' 'install' 'install the locked dependencies into the only TSDP virtualenv'
	@printf '  %-18s %s\n' 'env' 'verify Python and dependency versions; GPU may be unavailable'
	@printf '  %-18s %s\n' 'gpu' 'strictly verify WSL GPU and run CUDA forward/backward smoke tests'
	@printf '  %-18s %s\n' 'unit' 'run the MS, TensorShield, and TEESlice unit tests'
	@printf '  %-18s %s\n' 'results' 'verify formal MS and Lab/temp metrics, histories, masks, hashes, and plots'
	@printf '  %-18s %s\n' 'check' 'run gpu, unit, and dataset/protocol verification'
	@printf '  %-18s %s\n' 'prepare' 'download public datasets and ImageNet pretrained weights; no training'
	@printf '  %-18s %s\n' 'verify' 'verify public datasets and MS splits; use VERIFY_ARGS="" to check archives'
	@printf '  %-18s %s\n' 'datasets' 'download TensorShield datasets; override with DATASETS="c10 c100"'
	@printf '  %-18s %s\n' 'weights' 'download ImageNet pretrained weights only'
	@printf '  %-18s %s\n' 'c10' 'download CIFAR-10 only'
	@printf '  %-18s %s\n' 'c100' 'download CIFAR-100 only'
	@printf '  %-18s %s\n' 's10' 'download STL-10 only'
	@printf '  %-18s %s\n' 't200' 'download Tiny-ImageNet only'

install:
	test -x "$(PYTHON)"
	"$(PYTHON)" -m pip install --requirement "$(REQUIREMENTS)"

env:
	"$(PYTHON)" verify/verify_runtime.py --allow-cpu --skip-compute

gpu:
	"$(PYTHON)" verify/verify_runtime.py

unit:
	PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" -m unittest \
		verify.test_ms_surrogate \
		verify.test_tensorshield \
		verify.test_teeslice

results:
	PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" verify/verify_ms_results.py
	PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" verify/verify_lab.py

check: gpu unit verify results

prepare: datasets weights

verify:
	"$(PYTHON)" verify/verify_datasets.py $(VERIFY_ARGS)
	"$(PYTHON)" verify/verify_ms_splits.py

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
