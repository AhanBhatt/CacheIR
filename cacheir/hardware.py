from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass


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


def profile_hardware() -> HardwareProfile:
    return HardwareProfile(
        system=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count() or 1,
        memory_total_mb=_memory_total_mb(),
        gpus=_nvidia_gpus(),
    )


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
