#!/usr/bin/env python3
"""验证 TSDP 唯一 Python 环境、WSL GPU 桥接和 PyTorch CUDA 计算。"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ENV_NAME = "dl-py310-torch210-cu121"
EXPECTED_VERSIONS = {
    "torch": "2.1.0+cu121",
    "torchvision": "0.16.0+cu121",
    "torchaudio": "2.1.0+cu121",
    "numpy": "1.26.4",
    "pillow": "12.3.0",
    "tqdm": "4.68.4",
    "matplotlib": "3.10.9",
}
EXPECTED_CUDA = "12.1"


def locate_nvidia_smi() -> Path | None:
    """优先使用 PATH，在 WSL 中回退到驱动桥接的固定位置。"""
    executable = shutil.which("nvidia-smi")
    if executable:
        return Path(executable)
    wsl_executable = Path("/usr/lib/wsl/lib/nvidia-smi")
    return wsl_executable if wsl_executable.is_file() else None


def is_wsl() -> bool:
    try:
        proc_version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        proc_version = ""
    return "microsoft" in proc_version or "microsoft" in platform.release().lower()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="验证 TSDP 固定环境，并在默认模式下执行真实 CUDA 前向和反向计算。"
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="允许 GPU 不可见，仅检查 Python 依赖；正式实验前不要使用该选项。",
    )
    parser.add_argument(
        "--skip-compute",
        action="store_true",
        help="跳过 CUDA 矩阵乘法和卷积反向传播，只检查设备可见性。",
    )
    args = parser.parse_args()

    failures: list[str] = []

    def check(condition: bool, success: str, failure: str) -> None:
        if condition:
            print(f"[OK] {success}")
        else:
            failures.append(failure)
            print(f"[FAIL] {failure}")

    print("[INFO] TSDP runtime verification")
    print(f"[INFO] executable={sys.executable}")
    print(f"[INFO] prefix={sys.prefix}")
    print(f"[INFO] platform={platform.platform()}")

    check(
        sys.version_info[:2] == (3, 10),
        f"Python {platform.python_version()}",
        f"Python 必须为 3.10，当前为 {platform.python_version()}",
    )
    check(
        Path(sys.prefix).name == ENV_NAME,
        f"virtualenv={ENV_NAME}",
        f"必须使用唯一环境 {ENV_NAME}，当前 prefix={sys.prefix}",
    )

    for distribution, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        check(
            actual == expected,
            f"{distribution}=={actual}",
            f"{distribution} 应为 {expected}，当前为 {actual}",
        )

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        check=False,
        text=True,
    )
    check(
        pip_check.returncode == 0,
        "pip dependency graph is consistent",
        f"pip check 失败：{(pip_check.stdout + pip_check.stderr).strip()}",
    )

    running_in_wsl = is_wsl()
    print(f"[INFO] wsl={running_in_wsl}")
    if running_in_wsl:
        dxg = Path("/dev/dxg")
        dxg_accessible = dxg.exists() and os.access(dxg, os.R_OK | os.W_OK)
        if args.allow_cpu:
            print(f"[INFO] /dev/dxg accessible={dxg_accessible}")
        else:
            check(
                dxg_accessible,
                "/dev/dxg is accessible",
                "WSL 未映射或无权访问 /dev/dxg",
            )

    nvidia_smi = locate_nvidia_smi()
    if nvidia_smi is None:
        if args.allow_cpu:
            print("[INFO] nvidia-smi not found (allowed by --allow-cpu)")
        else:
            failures.append("找不到 nvidia-smi")
            print("[FAIL] 找不到 nvidia-smi")
    else:
        smi = subprocess.run(
            [
                str(nvidia_smi),
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if smi.returncode == 0:
            for line in smi.stdout.splitlines():
                print(f"[OK] nvidia-smi: {line.strip()}")
        elif args.allow_cpu:
            print(f"[INFO] nvidia-smi unavailable: {(smi.stdout + smi.stderr).strip()}")
        else:
            message = (smi.stdout + smi.stderr).strip()
            failures.append(f"nvidia-smi 失败：{message}")
            print(f"[FAIL] nvidia-smi 失败：{message}")

    try:
        import torch
        import torchaudio  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as exc:  # pragma: no cover - 失败路径用于环境诊断
        failures.append(f"PyTorch 三件套导入失败：{exc}")
        print(f"[FAIL] PyTorch 三件套导入失败：{exc}")
        print(f"[ERROR] runtime verification failed with {len(failures)} issue(s)")
        return 1

    check(
        torch.version.cuda == EXPECTED_CUDA,
        f"PyTorch CUDA runtime={torch.version.cuda}",
        f"PyTorch CUDA runtime 应为 {EXPECTED_CUDA}，当前为 {torch.version.cuda}",
    )
    print(f"[INFO] cuDNN={torch.backends.cudnn.version()}")

    cuda_available = torch.cuda.is_available()
    if args.allow_cpu and not cuda_available:
        print("[INFO] torch.cuda.is_available()=False (allowed by --allow-cpu)")
    else:
        check(
            cuda_available,
            "torch.cuda.is_available()=True",
            "PyTorch 无法访问 CUDA；请检查 /dev/dxg、Windows NVIDIA 驱动和当前 Python 环境",
        )

    if cuda_available:
        device_count = torch.cuda.device_count()
        check(device_count > 0, f"CUDA device_count={device_count}", "CUDA device_count=0")
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            total_mib = properties.total_memory / 1024**2
            print(
                f"[OK] cuda:{index} name={properties.name} "
                f"capability={properties.major}.{properties.minor} memory_mib={total_mib:.0f}"
            )

        if not args.skip_compute:
            device = torch.device("cuda:0")
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.cuda.reset_peak_memory_stats(device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            left = torch.randn((2048, 2048), device=device)
            right = torch.randn((2048, 2048), device=device)
            product = left @ right

            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(16, 10),
            ).to(device)
            inputs = torch.randn((8, 3, 128, 128), device=device)
            targets = torch.randint(0, 10, (8,), device=device)
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()

            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start.elapsed_time(end)
            peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
            gradients_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            )
            check(
                bool(torch.isfinite(product).all().item()),
                "CUDA matrix multiplication result is finite",
                "CUDA 矩阵乘法产生非有限值",
            )
            check(
                gradients_finite and bool(torch.isfinite(loss).item()),
                "CUDA convolution forward/backward is finite",
                "CUDA 卷积前向或反向传播产生非有限值",
            )
            print(f"[OK] CUDA smoke elapsed_ms={elapsed_ms:.2f} peak_memory_mib={peak_mib:.2f}")

    if failures:
        print(f"[ERROR] runtime verification failed with {len(failures)} issue(s)")
        return 1
    print("[OK] TSDP runtime verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
