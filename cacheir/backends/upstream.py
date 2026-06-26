from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExternalSystemStatus:
    name: str
    available: bool
    module: str | None = None
    command: str | None = None
    capability: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "module": self.module,
            "command": self.command,
            "capability": self.capability,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class IREECompileResult:
    target_backend: str
    byte_count: int
    output_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"target_backend": self.target_backend, "byte_count": self.byte_count, "output_path": self.output_path}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _iree_available() -> bool:
    if not (_has_module("iree.compiler") and _has_module("iree.runtime")):
        return False
    try:
        from iree.compiler.tools import binaries

        tool = binaries.find_tool("iree-compile")
        result = subprocess.run([str(tool), "--version"], capture_output=True, text=True, check=False, timeout=5)
    except Exception:
        return False
    return result.returncode == 0


def probe_external_systems() -> dict[str, ExternalSystemStatus]:
    iree_available = _iree_available()
    tvm_available = _has_module("tvm")
    return {
        "vllm": ExternalSystemStatus(
            name="vllm",
            available=_has_module("vllm"),
            module="vllm",
            command=shutil.which("vllm"),
            capability="serving benchmark comparison",
            notes="Windows/Python 3.13 often has no compatible binary wheel; use a Linux CUDA environment for production comparisons.",
        ),
        "llama.cpp": ExternalSystemStatus(
            name="llama.cpp",
            available=bool(shutil.which("llama-bench") or _has_module("llama_cpp")),
            module="llama_cpp" if _has_module("llama_cpp") else None,
            command=shutil.which("llama-bench"),
            capability="GGUF benchmark comparison",
            notes="CacheIR can run llama-bench when the standalone executable is on PATH.",
        ),
        "iree": ExternalSystemStatus(
            name="iree",
            available=iree_available,
            module="iree.compiler+iree.runtime" if iree_available else None,
            command=shutil.which("iree-benchmark-module"),
            capability="StableHLO/MLIR compile and runtime benchmark",
            notes="Uses iree-base-compiler and iree-base-runtime wheels.",
        ),
        "tvm": ExternalSystemStatus(
            name="tvm",
            available=tvm_available,
            module="tvm" if tvm_available else None,
            command=shutil.which("tvmc"),
            capability="TVM TE/TIR runtime benchmark",
            notes="Uses the apache-tvm Python wheel when installed.",
        ),
    }


def compile_stablehlo_with_iree(
    stablehlo_text: str,
    *,
    output_path: str | Path | None = None,
    target_backend: str = "llvm-cpu",
    target_cpu: str = "host",
) -> IREECompileResult:
    try:
        import iree.compiler as ireec
    except ImportError as exc:
        raise RuntimeError("IREE compilation requires iree-base-compiler") from exc

    extra_args: list[str] = []
    if target_backend == "llvm-cpu" and target_cpu:
        extra_args.append(f"--iree-llvmcpu-target-cpu={target_cpu}")
    blob = ireec.compile_str(
        stablehlo_text,
        target_backends=[target_backend],
        input_type="stablehlo",
        extra_args=extra_args,
    )
    resolved: str | None = None
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        resolved = str(path)
    return IREECompileResult(target_backend=target_backend, byte_count=len(blob), output_path=resolved)


def run_iree_stablehlo_benchmark(
    workdir: str | Path,
    *,
    min_time: str = "0.01s",
    repetitions: int = 1,
) -> dict[str, object]:
    command = shutil.which("iree-benchmark-module")
    if not command:
        return {"name": "iree", "available": False, "reason": "iree-benchmark-module not found"}

    stablehlo = """
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""
    path = Path(workdir) / "cacheir_iree_add.vmfb"
    compile_result = compile_stablehlo_with_iree(stablehlo, output_path=path)
    cmd = [
        command,
        f"--module={path}",
        "--function=main",
        "--input=4xf32=1,2,3,4",
        "--input=4xf32=5,6,7,8",
        f"--benchmark_min_time={min_time}",
        f"--benchmark_repetitions={repetitions}",
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    return {
        "name": "iree",
        "available": True,
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "elapsed_s": time.perf_counter() - start,
        "compile": compile_result.to_dict(),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def run_tvm_vector_add_benchmark(*, n: int = 256, number: int = 10, repeat: int = 3) -> dict[str, object]:
    try:
        import numpy as np
        import tvm
        from tvm import te
    except ImportError as exc:
        return {"name": "tvm", "available": False, "reason": str(exc)}

    start = time.perf_counter()
    a_placeholder = te.placeholder((n,), name="A", dtype="float32")
    b_placeholder = te.placeholder((n,), name="B", dtype="float32")
    c_compute = te.compute((n,), lambda i: a_placeholder[i] + b_placeholder[i], name="C")
    module = tvm.IRModule({"main": te.create_prim_func([a_placeholder, b_placeholder, c_compute])})
    built = tvm.build(module, target="llvm")
    device = tvm.cpu(0)
    a_value = tvm.runtime.tensor(np.ones(n, dtype="float32"), device)
    b_value = tvm.runtime.tensor(np.ones(n, dtype="float32"), device)
    c_value = tvm.runtime.empty((n,), "float32", device)
    built["main"](a_value, b_value, c_value)
    evaluator = built.time_evaluator("main", device, number=number, repeat=repeat)
    timings = tuple(float(value) for value in evaluator(a_value, b_value, c_value).results)
    checksum = float(c_value.numpy().sum())
    return {
        "name": "tvm",
        "available": True,
        "elapsed_s": time.perf_counter() - start,
        "n": n,
        "number": number,
        "repeat": repeat,
        "timings_s": timings,
        "best_s": min(timings) if timings else None,
        "checksum": checksum,
        "target": str(tvm.target.Target("llvm")),
    }


def _command_result(
    name: str,
    cmd: list[str],
    *,
    timeout: int = 600,
    json_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout, env=env)
    except FileNotFoundError:
        return {"name": name, "available": False, "command": cmd, "reason": "executable not found"}
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "available": True,
            "command": cmd,
            "reason": "timeout",
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
        }

    parsed: object | None = None
    if json_path is not None and json_path.exists():
        try:
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        parsed = _parse_first_json_document(result.stdout)
    return {
        "name": name,
        "available": True,
        "command": cmd,
        "returncode": result.returncode,
        "elapsed_s": time.perf_counter() - start,
        "parsed": parsed,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _parse_first_json_document(text: str) -> object | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        return value
    return None


def run_vllm_latency_benchmark(
    model: str,
    workdir: str | Path,
    *,
    input_len: int = 16,
    output_len: int = 8,
    batch_size: int = 1,
    num_iters: int = 1,
    warmup_iters: int = 0,
    timeout: int = 900,
    extra_args: list[str] | None = None,
    no_uva_fallback: bool = True,
) -> dict[str, object]:
    """Run vLLM's installed latency benchmark against a real local/HF model."""

    if not probe_external_systems()["vllm"].available:
        return {"name": "vllm", "available": False, "reason": "vllm is not importable"}
    command = shutil.which("vllm")
    if command:
        cmd = [command, "bench", "latency"]
    else:
        cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "latency"]
    output_path = Path(workdir) / "vllm_latency.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    cmd.extend(
        [
            "--model",
            model,
            "--input-len",
            str(input_len),
            "--output-len",
            str(output_len),
            "--batch-size",
            str(batch_size),
            "--num-iters",
            str(num_iters),
            "--num-iters-warmup",
            str(warmup_iters),
            "--output-json",
            str(output_path),
        ]
    )
    if extra_args:
        cmd.extend(extra_args)
    env = _vllm_compat_env(output_path.parent) if no_uva_fallback else None
    result = _command_result("vllm", cmd, timeout=timeout, json_path=output_path, env=env)
    result["model"] = model
    result["input_len"] = input_len
    result["output_len"] = output_len
    result["batch_size"] = batch_size
    result["no_uva_fallback"] = bool(no_uva_fallback)
    return result


def _vllm_compat_env(workdir: Path) -> dict[str, str]:
    compat_dir = workdir / "cacheir_vllm_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    (compat_dir / "sitecustomize.py").write_text(
        "\n".join(
            [
                "import os",
                "if os.environ.get('CACHEIR_VLLM_NO_UVA_FALLBACK') == '1':",
                "    try:",
                "        from cacheir.backends.vllm_compat import install_no_uva_fallback",
                "        install_no_uva_fallback()",
                "    except Exception as exc:",
                "        print(f'CacheIR vLLM no-UVA fallback failed: {exc}', flush=True)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    project_root = Path(__file__).resolve().parents[2]
    pythonpath = [str(compat_dir), str(project_root)]
    if existing:
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["CACHEIR_VLLM_NO_UVA_FALLBACK"] = "1"
    cuda_home = Path("/usr/local/cuda")
    if cuda_home.exists():
        env.setdefault("CUDA_HOME", str(cuda_home))
        cuda_bin = str(cuda_home / "bin")
        env["PATH"] = os.pathsep.join([cuda_bin, env.get("PATH", "")])
    return env


def run_llama_cpp_benchmark(
    model_path: str | Path,
    *,
    prompt_tokens: int = 16,
    generation_tokens: int = 8,
    repetitions: int = 1,
    timeout: int = 600,
    n_gpu_layers: int | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, object]:
    """Run llama.cpp's installed llama-bench against a real GGUF model file."""

    command = shutil.which("llama-bench")
    if not command:
        return {"name": "llama.cpp", "available": False, "reason": "llama-bench not found on PATH"}
    model = Path(model_path)
    if not model.exists():
        return {"name": "llama.cpp", "available": True, "reason": f"GGUF model not found: {model}"}
    cmd = [
        command,
        "-m",
        str(model),
        "-p",
        str(prompt_tokens),
        "-n",
        str(generation_tokens),
        "-r",
        str(repetitions),
        "-o",
        "json",
    ]
    if n_gpu_layers is not None:
        cmd.extend(["-ngl", str(n_gpu_layers)])
    if extra_args:
        cmd.extend(extra_args)
    result = _command_result("llama.cpp", cmd, timeout=timeout)
    result["model"] = str(model)
    result["prompt_tokens"] = prompt_tokens
    result["generation_tokens"] = generation_tokens
    return result


def run_installed_upstream_benchmarks(
    workdir: str | Path,
    *,
    vllm_model: str | None = None,
    llama_model: str | Path | None = None,
) -> dict[str, object]:
    statuses = probe_external_systems()
    benchmarks: dict[str, object] = {}
    benchmarks["iree"] = run_iree_stablehlo_benchmark(workdir) if statuses["iree"].available else statuses["iree"].to_dict()
    benchmarks["tvm"] = run_tvm_vector_add_benchmark() if statuses["tvm"].available else statuses["tvm"].to_dict()
    benchmarks["vllm"] = (
        run_vllm_latency_benchmark(vllm_model, workdir) if vllm_model and statuses["vllm"].available else statuses["vllm"].to_dict()
    )
    benchmarks["llama.cpp"] = (
        run_llama_cpp_benchmark(llama_model) if llama_model and statuses["llama.cpp"].available else statuses["llama.cpp"].to_dict()
    )
    return {"systems": {name: status.to_dict() for name, status in statuses.items()}, "benchmarks": benchmarks}


def dumps_upstream_report(workdir: str | Path) -> str:
    return json.dumps(run_installed_upstream_benchmarks(workdir), indent=2)
