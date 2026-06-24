from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class GPUInfo:
    name: str
    memory_total_mb: int | None = None
    driver_version: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class HardwareProfile:
    system: str
    machine: str
    processor: str
    cpu_count: int
    memory_total_mb: int | None
    gpus: list[GPUInfo]

    @property
    def suggested_target(self) -> str:
        return "cuda" if self.gpus else "cpu"

    def to_dict(self) -> dict[str, object]:
        return {
            "system": self.system,
            "machine": self.machine,
            "processor": self.processor,
            "cpu_count": self.cpu_count,
            "memory_total_mb": self.memory_total_mb,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "suggested_target": self.suggested_target,
        }


@dataclass
class BandwidthCalibration:
    cpu_copy_gbps: float
    cuda_h2d_gbps: float | None
    sample_mb: int
    repeats: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def profile_hardware() -> HardwareProfile:
    return HardwareProfile(
        system=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count() or 1,
        memory_total_mb=_memory_total_mb(),
        gpus=_nvidia_gpus(),
    )


def calibrate_bandwidth(*, sample_mb: int = 16, repeats: int = 5, include_cuda: bool = True) -> BandwidthCalibration:
    sample_mb = max(1, int(sample_mb))
    repeats = max(1, int(repeats))
    bytes_count = sample_mb * 1024 * 1024
    source = np.arange(bytes_count, dtype=np.uint8)
    target = np.empty_like(source)
    cpu_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        np.copyto(target, source)
        cpu_times.append(time.perf_counter() - start)
    cpu_gbps = _gbps(bytes_count, min(cpu_times))
    cuda_gbps = _cuda_h2d_gbps(bytes_count, repeats) if include_cuda else None
    return BandwidthCalibration(cpu_copy_gbps=cpu_gbps, cuda_h2d_gbps=cuda_gbps, sample_mb=sample_mb, repeats=repeats)


def _gbps(bytes_count: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(bytes_count) / seconds / 1.0e9


def _cuda_h2d_gbps(bytes_count: int, repeats: int) -> float | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        source = torch.empty(bytes_count, dtype=torch.uint8, device="cpu", pin_memory=True)
    except Exception:
        source = torch.empty(bytes_count, dtype=torch.uint8, device="cpu")
    target = torch.empty(bytes_count, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        target.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return _gbps(bytes_count, min(times))


def _memory_total_mb() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total // (1024 * 1024))
    except ImportError:
        pass

    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory", "/value"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in result.stdout.splitlines():
                if line.startswith("TotalPhysicalMemory="):
                    return int(line.split("=", 1)[1]) // (1024 * 1024)
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    return None


def _nvidia_gpus() -> list[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: list[GPUInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        memory = None
        if len(parts) > 1 and parts[1]:
            try:
                memory = int(parts[1])
            except ValueError:
                memory = None
        gpus.append(GPUInfo(name=parts[0], memory_total_mb=memory, driver_version=parts[2] if len(parts) > 2 else None))
    return gpus
