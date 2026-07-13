#!/usr/bin/env python3
"""按 baseline.json 顺序执行并校验正式 baseline。"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

from core.config import ATTACK_PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = Path(__file__).resolve().with_name("baseline.json")
TRAIN_PATH = Path(__file__).resolve().with_name("train.py")
RESULTS_ROOT = ROOT / "results" / "MS" / "resnet18" / "c100"
WEIGHTS_ROOT = ROOT / "weights" / "MS" / "surrogate" / "resnet18" / "c100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group",
        choices=("layers", "large_weight"),
        help="运行完整层配置或全局大权重标量配置。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只校验全部配置，不写训练产物。")
    parser.add_argument("--only", default=None, help="只处理指定 plan_id，多个 id 使用逗号分隔。")
    parser.add_argument("--jobs", type=int, default=1, help="并行训练进程数，默认 1。")
    return parser.parse_args()


def load_completed() -> dict[str, dict]:
    completed = {}
    for path in RESULTS_ROOT.glob("*/metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        plan_id = payload.get("plan_id")
        if plan_id:
            completed[str(plan_id)] = payload
    return completed


def load_incomplete() -> set[str]:
    incomplete = set()
    for path in WEIGHTS_ROOT.glob("*/params.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        plan_id = payload.get("plan_id")
        results_dir = Path(payload.get("results_dir", ""))
        if plan_id and not (results_dir / "metrics.json").is_file():
            incomplete.add(str(plan_id))
    return incomplete


def validate_result(config: dict, payload: dict) -> None:
    protocol_actual = {
        "attack_protocol": payload.get("attack_protocol"),
        "query_budget": payload.get("query_budget"),
        "label_mode": payload.get("label_mode"),
        "training_mode": payload.get("training_mode"),
        "query_transform": payload.get("query_transform"),
        "lr_step": payload.get("lr_step"),
        "primary": payload.get("primary"),
    }
    protocol_expected = {
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "query_budget": 500,
        "label_mode": "soft",
        "training_mode": "finetune",
        "query_transform": "test",
        "lr_step": 60,
        "primary": {"checkpoint": "end.pth", "epoch": 100},
    }
    if protocol_actual != protocol_expected:
        raise RuntimeError(
            f"plan_id={config['id']} 的正式攻击协议不一致："
            f"expected={protocol_expected}, actual={protocol_actual}"
        )
    protection = payload["protection"]
    expected = {
        "artifact_id": config["id"],
        "plan_id": config["id"],
        "protected_layer_count": config.get("protected_layer_count"),
        "source_ratio": config.get("source_ratio"),
    }
    actual = {
        "artifact_id": payload.get("artifact_id"),
        "plan_id": payload.get("plan_id"),
        "protected_layer_count": payload.get("protected_layer_count"),
        "source_ratio": payload.get("source_ratio"),
    }
    if actual != expected:
        raise RuntimeError(f"结果计划元数据不一致：expected={expected}, actual={actual}")
    for name in (
        "defense",
        "protected_unit_count",
        "protected_param_count",
        "total_param_count",
        "protected_param_ratio",
        "classifier_protected",
        "head_mode",
        "magnitude_eligible_count",
        "protection_mask_sha256",
    ):
        if protection[name] != config.get(name):
            raise RuntimeError(
                f"plan_id={config['id']} 的 {name} 不一致："
                f"expected={config.get(name)}, actual={protection[name]}"
            )
    if protection["magnitude_protected_count"] != config.get("protected_scalars"):
        raise RuntimeError(
            f"plan_id={config['id']} 的 magnitude_protected_count 不一致："
            f"expected={config.get('protected_scalars')}, "
            f"actual={protection['magnitude_protected_count']}"
        )


def build_command(config: dict, dry_run: bool, overwrite: bool) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_PATH),
        "resnet18",
        "c100",
        "--defense",
        config["defense"],
        "--plan-id",
        config["id"],
        "--budget",
        "500",
        "--training-mode",
        "finetune",
        "--label-mode",
        "soft",
        "--epochs",
        "100",
        "--batch-size",
        "64",
        "--lr",
        "0.01",
        "--momentum",
        "0.5",
        "--weight-decay",
        "0.0005",
        "--lr-step",
        "60",
        "--lr-gamma",
        "0.1",
    ]
    if config.get("protected_layers") is not None:
        command.extend(("--protected-layers", str(config["protected_layers"])))
    if config.get("protected_scalars") is not None:
        command.extend(("--protected-scalars", str(config["protected_scalars"])))
    if dry_run:
        command.append("--dry-run")
    elif overwrite:
        command.append("--overwrite")
    return command


def run_configuration(config: dict, command: list[str], capture_output: bool) -> None:
    plan_id = config["id"]
    print(f"[START] {plan_id}", flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        raise subprocess.CalledProcessError(result.returncode, command)

    if "--dry-run" not in command:
        updated = load_completed()
        if plan_id not in updated:
            raise RuntimeError(f"plan_id={plan_id} 运行后缺少 metrics.json。")
        validate_result(config, updated[plan_id])
    print(f"[DONE] {plan_id}", flush=True)


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs 必须大于等于 1。")
    manifest = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    group_name = "layer_sweep" if args.group == "layers" else "large_weight_sweep"
    configurations = manifest[group_name]["configurations"]
    selected = None if args.only is None else {item.strip() for item in args.only.split(",")}
    if selected is not None:
        known = {config["id"] for config in configurations}
        unknown = selected - known
        if unknown:
            raise ValueError(f"{args.group} 中存在未知 plan_id：{sorted(unknown)}")
        configurations = [config for config in configurations if config["id"] in selected]

    completed = load_completed()
    incomplete = load_incomplete()
    pending: list[tuple[dict, list[str]]] = []
    skipped = 0
    for ordinal, config in enumerate(configurations, start=1):
        plan_id = config["id"]
        if not args.dry_run and plan_id in completed:
            validate_result(config, completed[plan_id])
            print(f"[SKIP] {plan_id} 已完成并通过校验。")
            skipped += 1
            continue
        print(f"[QUEUE] {ordinal:02d}/{len(configurations):02d} {plan_id}")
        pending.append((config, build_command(config, args.dry_run, plan_id in incomplete)))

    worker_count = min(args.jobs, len(pending))
    if worker_count == 1:
        for config, command in pending:
            run_configuration(config, command, capture_output=False)
    elif worker_count > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(run_configuration, config, command, True)
                for config, command in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    print(f"[SUMMARY] executed={len(pending)}, skipped={skipped}, total={len(configurations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
