#!/usr/bin/env python3
"""保存 40 个 Conv weight/BN gamma 候选的四路原始输出。"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
TRAIN_VICTIM_ROOT = ROOT / "exp" / "MS" / "train_victim"
for import_root in (ROOT, TRAIN_ROOT, TRAIN_VICTIM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common.trainer import (  # noqa: E402
    build_generator,
    build_public_split_dataset,
    build_transforms,
    configure_reproducibility,
    seed_worker,
)
from exp.MS.train_surrogate.core.artifacts import sha256_file  # noqa: E402
from exp.MS.train_surrogate.core.config import resolve_device  # noqa: E402
from exp.MS.train_surrogate.core.data import (  # noqa: E402
    build_victim,
    read_query_indices,
)
from exp.MS.train_surrogate.defense import build_resnet18_tensor_units  # noqa: E402
from models import imagenet as imagenet_models  # noqa: E402
from playground.common import (  # noqa: E402
    MAIN_MODULES,
    ROUTES,
    extract_main_rows,
    hash_integer_sequence,
    plot_metric,
    sha256_file as sha256_local,
    write_json,
    write_tsv,
)


EXPERIMENT = "01_raw"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
QUERY_COUNT = 500
TRAIN_COUNT = 50_000
SEED = 42
BATCH_SIZE = 64
EXPECTED_CONV_COUNT = 20
EXPECTED_BN_COUNT = 20
DATA_FIELDS = (
    "candidate_index",
    "unit_index",
    "operator_type",
    "module",
    "state_name",
    "bias_state",
    "weight_shape",
    "parameter_count",
    "output_shape",
    "feature_count",
    "image_count",
    "raw_cross_l1",
    "raw_natural_l1",
    "product_score",
    "activation_path",
    "activation_sha256",
    "activation_bytes",
    "query_source_indices_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只处理首个 batch 并核对四路公式，不写任何结果。",
    )
    return parser.parse_args()


def select_candidates(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    candidates: dict[str, torch.nn.Module] = {}
    conv_count = 0
    bn_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.bias is not None:
                raise ValueError(f"{name} Conv 意外包含 bias。")
            candidates[name] = module
            conv_count += 1
        elif isinstance(module, torch.nn.BatchNorm2d):
            if not module.affine or module.weight is None or module.bias is None:
                raise ValueError(f"{name} BN 缺少 affine 参数。")
            candidates[name] = module
            bn_count += 1
    if conv_count != EXPECTED_CONV_COUNT or bn_count != EXPECTED_BN_COUNT:
        raise ValueError(
            f"ResNet18 候选应为 20 Conv weight + 20 BN gamma，"
            f"实际为 {conv_count}+{bn_count}。"
        )
    if len(candidates) != 40 or "last_linear" in candidates:
        raise ValueError("PG01 候选不是排除分类头后的 40 个共享 backbone weight。")
    return candidates


def validate_candidate_pairs(
    public: dict[str, torch.nn.Module],
    victim: dict[str, torch.nn.Module],
) -> None:
    if list(public) != list(victim):
        raise ValueError("public/victim 候选名称或顺序不一致。")
    for name, public_module in public.items():
        victim_module = victim[name]
        if type(public_module) is not type(victim_module):
            raise ValueError(f"{name} 的 public/victim 类型不一致。")
        if public_module.weight.shape != victim_module.weight.shape:
            raise ValueError(f"{name}.weight 的 public/victim 形状不一致。")
        if isinstance(public_module, torch.nn.Conv2d):
            for field in ("stride", "padding", "dilation", "groups"):
                if getattr(public_module, field) != getattr(victim_module, field):
                    raise ValueError(f"{name} 的卷积几何字段 {field} 不一致。")
        elif (
            public_module.running_mean.shape != victim_module.running_mean.shape
            or public_module.running_var.shape != victim_module.running_var.shape
            or public_module.eps != victim_module.eps
        ):
            raise ValueError(f"{name} 的 BN running state 形状或 eps 不一致。")


def register_capture(
    candidates: dict[str, torch.nn.Module],
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []
    for name, module in candidates.items():
        def capture(_module, current_inputs, output, current=name):
            if len(current_inputs) != 1 or not torch.is_tensor(current_inputs[0]):
                raise ValueError(f"{current} 的算子输入不可识别。")
            inputs[current] = current_inputs[0].detach().clone()
            outputs[current] = output.detach().clone()

        handles.append(module.register_forward_hook(capture))
    return handles


def apply_conv(
    input_tensor: torch.Tensor,
    geometry: torch.nn.Conv2d,
    weight: torch.Tensor,
) -> torch.Tensor:
    return functional.conv2d(
        input_tensor,
        weight,
        bias=None,
        stride=geometry.stride,
        padding=geometry.padding,
        dilation=geometry.dilation,
        groups=geometry.groups,
    )


def normalize_bn_input(
    input_tensor: torch.Tensor,
    module: torch.nn.BatchNorm2d,
) -> torch.Tensor:
    mean = module.running_mean.reshape(1, -1, 1, 1)
    inverse_std = torch.rsqrt(
        module.running_var.reshape(1, -1, 1, 1) + module.eps
    )
    return (input_tensor - mean) * inverse_std


def apply_gamma(input_tensor: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return input_tensor * gamma.reshape(1, -1, 1, 1)


def four_routes(
    public_input: torch.Tensor,
    victim_input: torch.Tensor,
    public_module: torch.nn.Module,
    victim_module: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if isinstance(public_module, torch.nn.Conv2d):
        routes = {
            "z_pp": apply_conv(public_input, public_module, public_module.weight),
            "z_pv": apply_conv(public_input, public_module, victim_module.weight),
            "z_vp": apply_conv(victim_input, public_module, public_module.weight),
            "z_vv": apply_conv(victim_input, public_module, victim_module.weight),
        }
        compact = apply_conv(
            victim_input - public_input,
            public_module,
            victim_module.weight - public_module.weight,
        )
    elif isinstance(public_module, torch.nn.BatchNorm2d):
        normalized_public = normalize_bn_input(public_input, public_module)
        normalized_victim = normalize_bn_input(victim_input, victim_module)
        routes = {
            "z_pp": apply_gamma(normalized_public, public_module.weight),
            "z_pv": apply_gamma(normalized_public, victim_module.weight),
            "z_vp": apply_gamma(normalized_victim, public_module.weight),
            "z_vv": apply_gamma(normalized_victim, victim_module.weight),
        }
        compact = apply_gamma(
            normalized_victim - normalized_public,
            victim_module.weight - public_module.weight,
        )
    else:
        raise TypeError(f"不支持的候选类型：{type(public_module).__name__}")
    return routes, compact


def shape_text(tensor: torch.Tensor) -> str:
    return "×".join(str(value) for value in tensor.shape[1:])


def save_activation(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def collect_routes(
    loader: DataLoader,
    public_model: torch.nn.Module,
    victim_model: torch.nn.Module,
    public_candidates: dict[str, torch.nn.Module],
    victim_candidates: dict[str, torch.nn.Module],
    *,
    device: torch.device,
    dry_run: bool,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, float],
    int,
]:
    public_inputs: dict[str, torch.Tensor] = {}
    public_outputs: dict[str, torch.Tensor] = {}
    victim_inputs: dict[str, torch.Tensor] = {}
    victim_outputs: dict[str, torch.Tensor] = {}
    handles = [
        *register_capture(public_candidates, public_inputs, public_outputs),
        *register_capture(victim_candidates, victim_inputs, victim_outputs),
    ]
    storage: dict[str, dict[str, torch.Tensor]] = {}
    correctness = {
        "max_conv_public_hook_error": 0.0,
        "max_conv_victim_hook_error": 0.0,
        "max_conv_compact_identity_error": 0.0,
        "max_bn_gamma_public_hook_error": 0.0,
        "max_bn_gamma_victim_hook_error": 0.0,
        "max_bn_gamma_compact_identity_error": 0.0,
        "max_stem_compact_cross_abs": 0.0,
    }
    processed = 0
    try:
        for batch_index, (images, _labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            public_inputs.clear()
            public_outputs.clear()
            victim_inputs.clear()
            victim_outputs.clear()
            public_model(images)
            victim_model(images)
            expected = set(public_candidates)
            if any(
                set(captured) != expected
                for captured in (
                    public_inputs,
                    public_outputs,
                    victim_inputs,
                    victim_outputs,
                )
            ):
                raise ValueError("forward hook 没有完整捕获 40 个候选。")
            batch = images.size(0)
            for name, public_module in public_candidates.items():
                victim_module = victim_candidates[name]
                routes, compact = four_routes(
                    public_inputs[name],
                    victim_inputs[name],
                    public_module,
                    victim_module,
                )
                expanded = (
                    routes["z_vv"]
                    - routes["z_vp"]
                    - routes["z_pv"]
                    + routes["z_pp"]
                )
                if isinstance(public_module, torch.nn.Conv2d):
                    correctness["max_conv_public_hook_error"] = max(
                        correctness["max_conv_public_hook_error"],
                        float((routes["z_pp"] - public_outputs[name]).abs().max().item()),
                    )
                    correctness["max_conv_victim_hook_error"] = max(
                        correctness["max_conv_victim_hook_error"],
                        float((routes["z_vv"] - victim_outputs[name]).abs().max().item()),
                    )
                    correctness["max_conv_compact_identity_error"] = max(
                        correctness["max_conv_compact_identity_error"],
                        float((expanded - compact).abs().max().item()),
                    )
                else:
                    public_beta = public_module.bias.reshape(1, -1, 1, 1)
                    victim_beta = victim_module.bias.reshape(1, -1, 1, 1)
                    correctness["max_bn_gamma_public_hook_error"] = max(
                        correctness["max_bn_gamma_public_hook_error"],
                        float(
                            (routes["z_pp"] + public_beta - public_outputs[name])
                            .abs()
                            .max()
                            .item()
                        ),
                    )
                    correctness["max_bn_gamma_victim_hook_error"] = max(
                        correctness["max_bn_gamma_victim_hook_error"],
                        float(
                            (routes["z_vv"] + victim_beta - victim_outputs[name])
                            .abs()
                            .max()
                            .item()
                        ),
                    )
                    correctness["max_bn_gamma_compact_identity_error"] = max(
                        correctness["max_bn_gamma_compact_identity_error"],
                        float((expanded - compact).abs().max().item()),
                    )
                if name == "conv1":
                    correctness["max_stem_compact_cross_abs"] = max(
                        correctness["max_stem_compact_cross_abs"],
                        float(compact.abs().max().item()),
                    )
                if not dry_run:
                    if name not in storage:
                        storage[name] = {
                            route: torch.empty(
                                (QUERY_COUNT, *routes[route].shape[1:]),
                                dtype=torch.float32,
                                device="cpu",
                            )
                            for route in ROUTES
                        }
                        storage[name]["cross"] = torch.empty(
                            (QUERY_COUNT, *compact.shape[1:]),
                            dtype=torch.float32,
                            device="cpu",
                        )
                    for route in ROUTES:
                        storage[name][route][processed : processed + batch].copy_(
                            routes[route].detach().to(device="cpu", dtype=torch.float32)
                        )
                    storage[name]["cross"][processed : processed + batch].copy_(
                        compact.detach().to(device="cpu", dtype=torch.float32)
                    )
            processed += batch
            print(
                f"[QUERY {batch_index + 1:03d}/{len(loader):03d}] "
                f"processed={processed}/{QUERY_COUNT}",
                flush=True,
            )
            if dry_run:
                break
    finally:
        for handle in handles:
            handle.remove()
    limits = {
        "max_conv_public_hook_error": 1e-6,
        "max_conv_victim_hook_error": 1e-6,
        "max_conv_compact_identity_error": 2e-5,
        "max_bn_gamma_public_hook_error": 2e-6,
        "max_bn_gamma_victim_hook_error": 2e-6,
        "max_bn_gamma_compact_identity_error": 2e-6,
        "max_stem_compact_cross_abs": 0.0,
    }
    for field, limit in limits.items():
        if correctness[field] > limit:
            raise RuntimeError(f"PG01 正确性检查失败：{field}={correctness[field]}")
    if not dry_run and processed != QUERY_COUNT:
        raise RuntimeError(f"PG01 只处理了 {processed}/{QUERY_COUNT} 张 query。")
    return storage, correctness, processed


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = resolve_device(args.device)
    configure_reproducibility(SEED, deterministic=True)
    victim_checkpoint = (
        ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    )
    public_checkpoint = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    victim, victim_metadata = build_victim(MODEL, NUM_CLASSES, victim_checkpoint)
    public = imagenet_models.resnet18(num_classes=1000)
    imagenet_models.load_official_imagenet_weights(
        MODEL,
        public,
        str(public_checkpoint),
        strict=True,
    )
    public_candidates = select_candidates(public)
    victim_candidates = select_candidates(victim)
    validate_candidate_pairs(public_candidates, victim_candidates)
    unit_by_state = {
        unit.state_name: unit for unit in build_resnet18_tensor_units(victim)
    }
    candidate_states = tuple(f"{name}.weight" for name in public_candidates)
    if set(candidate_states) - set(unit_by_state):
        raise ValueError("PG01 候选无法映射到正式 122-unit 注册表。")

    _, test_transform = build_transforms(DATASET)
    dataset = build_public_split_dataset(
        DATASET,
        ROOT / "dataset" / "public",
        "train",
        test_transform,
    )
    if len(dataset) != TRAIN_COUNT:
        raise ValueError(f"CIFAR-100 train 应为 {TRAIN_COUNT} 张。")
    query_indices = read_query_indices(ROOT / "dataset" / "MS", DATASET)[:QUERY_COUNT]
    if len(query_indices) != QUERY_COUNT or len(set(query_indices)) != QUERY_COUNT:
        raise ValueError("PG01 没有得到固定且不重复的 500 张 query。")
    query_hash = hash_integer_sequence(query_indices)
    loader = DataLoader(
        Subset(dataset, query_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(SEED, offset=0),
    )
    public = public.to(device).eval()
    victim = victim.to(device).eval()
    storage, correctness, processed = collect_routes(
        loader,
        public,
        victim,
        public_candidates,
        victim_candidates,
        device=device,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"[INFO] dry-run 通过：processed={processed} correctness={correctness}")
        return 0

    output_root = ROOT / "results" / "playground" / EXPERIMENT
    activation_root = output_root / "activations"
    output_root.mkdir(parents=True, exist_ok=True)
    activation_root.mkdir(parents=True, exist_ok=True)
    for path in activation_root.glob("*.pt"):
        path.unlink()

    rows: list[dict[str, object]] = []
    total_activation_bytes = 0
    for candidate_index, (name, module) in enumerate(public_candidates.items(), start=1):
        state_name = f"{name}.weight"
        unit = unit_by_state[state_name]
        routes = storage.pop(name)
        cross = routes["cross"]
        natural = routes["z_vv"] - routes["z_pp"]
        raw_cross = float(cross.flatten(1).abs().sum(dim=1).double().mean().item())
        raw_natural = float(natural.flatten(1).abs().sum(dim=1).double().mean().item())
        if not all(math.isfinite(value) for value in (raw_cross, raw_natural)):
            raise ValueError(f"{state_name} 的原始残差不是有限值。")
        output_shape = shape_text(routes["z_pp"])
        feature_count = routes["z_pp"][0].numel()
        activation_path = activation_root / f"unit_{unit.index:03d}.pt"
        save_activation(
            activation_path,
            {
                "schema_version": 1,
                "module": name,
                "state_name": state_name,
                "unit_index": unit.index,
                "operator_type": (
                    "conv_weight"
                    if isinstance(module, torch.nn.Conv2d)
                    else "bn_gamma"
                ),
                "parameter_count": module.weight.numel(),
                "query_source_indices_sha256": query_hash,
                "routes": {route: routes[route] for route in ROUTES},
                "cross": routes["cross"],
            },
        )
        activation_bytes = activation_path.stat().st_size
        total_activation_bytes += activation_bytes
        rows.append(
            {
                "candidate_index": candidate_index,
                "unit_index": unit.index,
                "operator_type": (
                    "conv_weight"
                    if isinstance(module, torch.nn.Conv2d)
                    else "bn_gamma"
                ),
                "module": name,
                "state_name": state_name,
                "bias_state": "",
                "weight_shape": "×".join(str(value) for value in module.weight.shape),
                "parameter_count": module.weight.numel(),
                "output_shape": output_shape,
                "feature_count": feature_count,
                "image_count": QUERY_COUNT,
                "raw_cross_l1": raw_cross,
                "raw_natural_l1": raw_natural,
                "product_score": raw_cross * raw_natural,
                "activation_path": str(activation_path.relative_to(ROOT)),
                "activation_sha256": sha256_local(activation_path),
                "activation_bytes": activation_bytes,
                "query_source_indices_sha256": query_hash,
            }
        )
        del routes, cross, natural
        print(
            f"[SAVE {candidate_index:02d}/40] {state_name} "
            f"raw_product={rows[-1]['product_score']:.6g}",
            flush=True,
        )

    write_tsv(output_root / "data.tsv", rows, DATA_FIELDS)
    main_rows = extract_main_rows(rows)
    write_tsv(output_root / "main.tsv", main_rows, DATA_FIELDS)
    outputs: dict[str, str] = {
        "data": "results/playground/01_raw/data.tsv",
        "main": "results/playground/01_raw/main.tsv",
        "activations": "results/playground/01_raw/activations",
    }
    plot_specs = (
        ("raw_cross_l1", "cross", "Raw cross-residual L1", "mean_image(sum_CHW(|I|))"),
        ("raw_natural_l1", "natural", "Raw natural-residual L1", "mean_image(sum_CHW(|N|))"),
        (
            "product_score",
            "product",
            "Raw cross × natural residual score",
            "raw_cross_l1 × raw_natural_l1",
        ),
    )
    for scope, scope_rows in (("all", rows), ("main", main_rows)):
        for field, suffix, title, xlabel in plot_specs:
            path = output_root / f"{scope}_{suffix}.png"
            plot_metric(
                path,
                scope_rows,
                field=field,
                title=title,
                xlabel=xlabel,
                scope=scope,
            )
            outputs[f"{scope}_{suffix}"] = str(path.relative_to(ROOT))

    manifest = {
        "schema_version": 1,
        "experiment": "01_raw_weight_routes",
        "scientific_status": "raw_routes_no_normalization_no_ms_feedback",
        "dataset": DATASET,
        "model": MODEL,
        "seed": SEED,
        "query": {
            "split": "query_pool_ms",
            "count": QUERY_COUNT,
            "selection": "canonical_query_rank_prefix",
            "source_indices": query_indices,
            "source_indices_sha256": query_hash,
            "transform": "test",
        },
        "models": {
            "public": {
                "checkpoint": str(public_checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(public_checkpoint),
                "num_classes": 1000,
            },
            "victim": {
                "checkpoint": str(victim_checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(victim_checkpoint),
                "checkpoint_epoch": victim_metadata.get("epoch"),
                "num_classes": NUM_CLASSES,
            },
        },
        "candidate_count": len(rows),
        "conv_weight_count": sum(row["operator_type"] == "conv_weight" for row in rows),
        "bn_gamma_count": sum(row["operator_type"] == "bn_gamma" for row in rows),
        "main_candidate_count": len(main_rows),
        "main_modules": list(MAIN_MODULES),
        "candidate_rule": "shared_backbone_modules_with_state_suffix_weight",
        "excluded_states": {
            "bias": "all_bias_states_excluded",
            "classifier": "last_linear_weight_and_bias_excluded_from_residuals",
        },
        "bn_definition": {
            "candidate": "gamma_weight_only",
            "beta_in_routes": False,
            "running_state_role": "normalize_public_and_victim_inputs_separately",
        },
        "routes": {
            "index_order": "first_input_source_second_weight_source",
            "z_pp": "operator(h_public, weight_public)",
            "z_pv": "operator(h_public, weight_victim)",
            "z_vp": "operator(h_victim, weight_public)",
            "z_vv": "operator(h_victim, weight_victim)",
            "dtype": "float32",
            "image_order": "query.source_indices",
        },
        "residuals": {
            "cross": "I=z_vv-z_vp-z_pv+z_pp",
            "natural": "N=z_vv-z_pp",
            "raw_cross_l1": "mean_image(sum_output(abs(I)))",
            "raw_natural_l1": "mean_image(sum_output(abs(N)))",
            "product_score": "raw_cross_l1*raw_natural_l1",
            "primary_score": "product_score",
            "normalization": "none",
        },
        "activation_storage": {
            "format": "one_torch_file_per_weight",
            "file_count": len(rows),
            "total_bytes": total_activation_bytes,
            "exact_compact_I_saved": True,
            "derived_N_saved": False,
        },
        "correctness": correctness,
        "execution": {
            "device": str(device),
            "batch_size": BATCH_SIZE,
            "num_workers": args.num_workers,
            "processed_image_count": processed,
            "model_mode": "eval",
            "gradient_enabled": False,
        },
        "outputs": outputs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_root / "manifest.json", manifest)
    print(
        f"[OK] PG01 写入 40 个 weight 的四路原始输出，"
        f"共 {total_activation_bytes / 1024**3:.3f} GiB。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
