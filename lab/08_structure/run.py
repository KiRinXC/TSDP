#!/usr/bin/env python3
"""核对 ResNet18+CIFAR-100 逐算子结构表中的输出尺寸。"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp" / "MS" / "train_surrogate"))

from models.imagenet import resnet18  # noqa: E402
from selector import AUTHOR_RESNET18_C100_ELIGIBLE_RANK  # noqa: E402


SOURCE_TABLE = Path(__file__).resolve().with_name("structure.tsv")
TABLE = ROOT / "results" / "lab" / "08_structure" / "resnet18_c100.tsv"
TOP17_TABLE = (
    ROOT
    / "results"
    / "lab"
    / "08_structure"
    / "tensorshield_top17.tsv"
)
OCCURRENCE_PATTERN = re.compile(r"^(.*)（第([12])次）$")
NON_MODULE_ROWS = {"输入", "flatten"}
TENSORSHIELD_TOP17 = AUTHOR_RESNET18_C100_ELIGIBLE_RANK
OUTPUT_FIELDS = (
    "unit编号",
    "state名称",
    "state类型",
    "模块",
    "计算",
    "输入",
    "输出",
    "TensorShield Top-17",
)


def shape_text(tensor: torch.Tensor) -> str:
    shape = tuple(tensor.shape[1:])
    return "×".join(str(value) for value in shape)


def write_expanded_table(model: torch.nn.Module) -> list[dict[str, str]]:
    state_items = list(model.state_dict().items())
    parameter_names = set(dict(model.named_parameters()))
    states_by_module: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for index, (state_name, _value) in enumerate(state_items):
        module_name = state_name.rsplit(".", 1)[0]
        state_type = "parameter" if state_name in parameter_names else "buffer"
        states_by_module[module_name].append((index, state_name, state_type))

    with SOURCE_TABLE.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))

    rows: list[dict[str, str]] = []
    tensorshield_rank = {
        state_name: f"Top-{rank}"
        for rank, state_name in enumerate(TENSORSHIELD_TOP17, start=1)
    }
    for source in source_rows:
        table_name = source["模块"]
        match = OCCURRENCE_PATTERN.match(table_name)
        module_name = match.group(1) if match else table_name
        states = states_by_module.get(module_name, [])
        if not states:
            states = [(-1, "—", "—")]
        for index, state_name, state_type in states:
            rows.append(
                {
                    "unit编号": "—" if index < 0 else str(index),
                    "state名称": state_name,
                    "state类型": state_type,
                    "模块": table_name,
                    "计算": source["计算"],
                    "输入": source["输入"],
                    "输出": source["输出"],
                    "TensorShield Top-17": tensorshield_rank.get(
                        state_name,
                        "—",
                    ),
                }
            )

    TABLE.parent.mkdir(parents=True, exist_ok=True)

    def write_rows(path: Path, output_rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_FIELDS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(output_rows)

    write_rows(TABLE, rows)
    top17_rows = sorted(
        (row for row in rows if row["TensorShield Top-17"] != "—"),
        key=lambda row: int(row["unit编号"]),
    )
    write_rows(TOP17_TABLE, top17_rows)
    return rows


def main() -> None:
    model = resnet18(num_classes=100).eval()
    if len(model.state_dict()) != 122:
        raise AssertionError(
            f"ResNet18 unit 数量应为 122，实际为 {len(model.state_dict())}。"
        )
    rows = write_expanded_table(model)
    unit_by_state = {
        state_name: index
        for index, state_name in enumerate(model.state_dict())
    }
    observed: dict[str, list[str]] = defaultdict(list)
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"{name} 的输出不是 Tensor。")
            observed[name].append(shape_text(output))

        return hook

    for name, module in model.named_modules():
        if name and not any(module.children()):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 32, 32))
    finally:
        for handle in handles:
            handle.remove()

    if tuple(output.shape) != (1, 100):
        raise AssertionError(f"分类头输出应为 (1, 100)，实际为 {tuple(output.shape)}。")

    failures = []
    for row in rows:
        table_name = row["模块"]
        state_name = row["state名称"]
        expected_unit = (
            "—" if state_name == "—" else str(unit_by_state[state_name])
        )
        if row["unit编号"] != expected_unit:
            failures.append(
                f"{state_name}: unit 表中 {row['unit编号']}，实际 {expected_unit}"
            )
        if (
            table_name in NON_MODULE_ROWS
            or table_name.endswith(".identity")
            or table_name.endswith(".add")
        ):
            continue
        match = OCCURRENCE_PATTERN.match(table_name)
        if match:
            module_name = match.group(1)
            occurrence = int(match.group(2)) - 1
        else:
            module_name = table_name
            occurrence = 0
        actual_values = observed.get(module_name, [])
        if occurrence >= len(actual_values):
            failures.append(f"{table_name}: 未捕获模块输出")
            continue
        actual = actual_values[occurrence]
        expected = row["输出"]
        if actual != expected:
            failures.append(f"{table_name}: 表中 {expected}，实际 {actual}")

    if failures:
        raise AssertionError("\n".join(failures))
    numbered_rows = sum(row["unit编号"] != "—" for row in rows)
    if numbered_rows != 122:
        raise AssertionError(
            f"编号行数应为 122，实际为 {numbered_rows}。"
        )
    marked = {
        row["state名称"]: row["TensorShield Top-17"]
        for row in rows
        if row["TensorShield Top-17"] != "—"
    }
    expected_marked = {
        state_name: f"Top-{rank}"
        for rank, state_name in enumerate(TENSORSHIELD_TOP17, start=1)
    }
    if marked != expected_marked:
        raise AssertionError(
            f"TensorShield Top-17 标注不一致：{marked}"
        )
    print(f"[OK] 已核对 {len(rows)} 行，其中 122 行各对应一个 unit：{TABLE}")
    print("[OK] H 列已按 eligible rank 标注 TensorShield Top-1 至 Top-17")
    print(f"[OK] 已按 unit 顺序写入 17 行子集：{TOP17_TABLE}")
    print("[OK] ResNet18 输出尺寸：100 类 logits")


if __name__ == "__main__":
    main()
