from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cacheir.benchmark import run_benchmark
from cacheir.backends.upstream import (
    probe_external_systems,
    run_iree_stablehlo_benchmark,
    run_llama_cpp_benchmark,
    run_tvm_vector_add_benchmark,
    run_vllm_latency_benchmark,
)


KNOWN_TOOLS = {
    "vllm": "vllm",
    "llama.cpp": "llama-bench",
    "tensorrt_llm": "trtllm-bench",
    "mlc_llm": "mlc_llm",
    "iree": "iree-benchmark-module",
    "tvm": "tvm",
}


def probe_tool(name: str) -> dict[str, object]:
    status = probe_external_systems()[name]
    payload = status.to_dict()
    payload["reason"] = "no command supplied"
    expected = KNOWN_TOOLS[name]
    if name in {"vllm", "tvm"}:
        payload["expected_module"] = expected
    else:
        payload["expected_executable"] = expected
    return payload


def run_command(name: str, command: str, timeout: int) -> dict[str, object]:
    start = time.perf_counter()
    try:
        result = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - start
        return {
            "name": name,
            "available": True,
            "command": command,
            "returncode": result.returncode,
            "elapsed_s": elapsed,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except FileNotFoundError:
        return {"name": name, "available": False, "command": command, "reason": "executable not found"}
    except subprocess.TimeoutExpired as exc:
        return {"name": name, "available": True, "command": command, "reason": "timeout", "stdout_tail": (exc.stdout or "")[-4000:]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CacheIR benchmark output with external benchmark commands")
    parser.add_argument("artifact", help="CacheIR artifact path")
    parser.add_argument("--prompt", default="CacheIR benchmark prompt")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--vllm-command", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--llama-command", default=None)
    parser.add_argument("--llama-model", default=None)
    parser.add_argument("--tensorrt-llm-command", default=None)
    parser.add_argument("--mlc-llm-command", default=None)
    parser.add_argument("--iree-command", default=None)
    parser.add_argument("--tvm-command", default=None)
    parser.add_argument("--run-installed-smoke", action="store_true", help="run IREE/TVM smoke benchmarks when those wheels are installed and no command is supplied")
    parser.add_argument("--output", type=Path, default=Path("benchmark_comparison.json"))
    args = parser.parse_args()

    cacheir = run_benchmark(args.artifact, prompt=args.prompt, decode_tokens=args.decode_tokens, repeats=args.repeats).to_dict()
    external_commands = {
        "vllm": args.vllm_command,
        "llama.cpp": args.llama_command,
        "tensorrt_llm": args.tensorrt_llm_command,
        "mlc_llm": args.mlc_llm_command,
        "iree": args.iree_command,
        "tvm": args.tvm_command,
    }
    external = []
    for name, command in external_commands.items():
        if command:
            external.append(run_command(name, command, args.timeout))
        elif args.run_installed_smoke and name == "vllm" and args.vllm_model:
            external.append(run_vllm_latency_benchmark(args.vllm_model, Path(".tmp") / "upstream", timeout=args.timeout))
        elif args.run_installed_smoke and name == "llama.cpp" and args.llama_model:
            external.append(run_llama_cpp_benchmark(args.llama_model, timeout=args.timeout))
        elif args.run_installed_smoke and name == "iree" and probe_external_systems()["iree"].available:
            external.append(run_iree_stablehlo_benchmark(Path(".tmp") / "upstream"))
        elif args.run_installed_smoke and name == "tvm" and probe_external_systems()["tvm"].available:
            external.append(run_tvm_vector_add_benchmark())
        else:
            external.append(probe_tool(name))

    payload = {"cacheir": cacheir, "external": external}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
