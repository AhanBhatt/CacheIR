from __future__ import annotations

import argparse
import json
from pathlib import Path

from cacheir.benchmark import run_benchmark, save_benchmark
from cacheir.compiler import compile_model
from cacheir.hardware import profile_hardware
from cacheir.importers.tiny import create_tiny_model
from cacheir.runtime import Runtime
from cacheir.runtime.artifact import CompileArtifact
from cacheir.visualize import export_graph


def _compile(args: argparse.Namespace) -> None:
    artifact = compile_model(
        args.model_path,
        target=args.target,
        quant=args.quant,
        mode=args.mode,
        max_batch=args.max_batch,
        max_seq=args.max_seq,
        output=args.output,
    )
    print(f"wrote {args.output or '<memory>'}")
    for mode, graph in artifact.graphs.items():
        print(f"{mode}: {len(graph.nodes)} nodes, arena={graph.attrs.get('memory_plan', {}).get('arena_bytes', 0)} bytes")


def _inspect(args: argparse.Namespace) -> None:
    artifact = CompileArtifact.load(args.artifact)
    graph = artifact.graph(args.mode)
    if args.pass_name:
        for trace in artifact.pass_traces.get(args.mode, []):
            if trace["name"] == args.pass_name:
                print(trace["diff"] or trace["after"])
                return
        raise SystemExit(f"pass {args.pass_name!r} not found for mode {args.mode}")
    print(graph.to_text())


def _run(args: argparse.Namespace) -> None:
    runtime = Runtime(args.artifact)
    for token in runtime.generate(args.prompt, max_new_tokens=args.max_new_tokens):
        print(token, end="", flush=True)
    print()


def _benchmark(args: argparse.Namespace) -> None:
    artifact = CompileArtifact.load(args.artifact)
    result = run_benchmark(artifact, prompt=args.prompt, decode_tokens=args.decode_tokens, repeats=args.repeats)
    text = json.dumps(result.to_dict(), indent=2)
    if args.output:
        save_benchmark(result, args.output)
        print(f"wrote {args.output}")
    print(text)


def _export(args: argparse.Namespace) -> None:
    artifact = CompileArtifact.load(args.artifact)
    path = export_graph(artifact, args.output, mode=args.mode, fmt=args.format)
    print(path)


def _profile(args: argparse.Namespace) -> None:
    print(json.dumps(profile_hardware().to_dict(), indent=2))


def _make_tiny(args: argparse.Namespace) -> None:
    path = create_tiny_model(args.output)
    print(path)


def _serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Serving requires optional dependencies: pip install cacheir[server]") from exc
    from cacheir.runtime.server import create_app

    uvicorn.run(create_app(args.artifact), host=args.host, port=args.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cacheir")
    sub = parser.add_subparsers(dest="command", required=True)

    make_tiny = sub.add_parser("make-tiny", help="create a tiny CacheIR-compatible model")
    make_tiny.add_argument("output")
    make_tiny.set_defaults(func=_make_tiny)

    compile_cmd = sub.add_parser("compile", help="compile a model into a CacheIR artifact")
    compile_cmd.add_argument("model_path")
    compile_cmd.add_argument("--target", default="cpu")
    compile_cmd.add_argument("--quant", default=None)
    compile_cmd.add_argument("--mode", nargs="+", default=["prefill", "decode"], choices=["prefill", "decode"])
    compile_cmd.add_argument("--max-batch", type=int, default=1)
    compile_cmd.add_argument("--max-seq", type=int, default=128)
    compile_cmd.add_argument("--output", "-o", default="cacheir_artifact.json")
    compile_cmd.set_defaults(func=_compile)

    inspect_cmd = sub.add_parser("inspect", help="print optimized IR or a pass diff")
    inspect_cmd.add_argument("artifact")
    inspect_cmd.add_argument("--mode", default="decode", choices=["prefill", "decode"])
    inspect_cmd.add_argument("--pass-name", default=None)
    inspect_cmd.set_defaults(func=_inspect)

    run_cmd = sub.add_parser("run", help="run local greedy generation")
    run_cmd.add_argument("artifact")
    run_cmd.add_argument("--prompt", default="")
    run_cmd.add_argument("--max-new-tokens", type=int, default=16)
    run_cmd.set_defaults(func=_run)

    bench_cmd = sub.add_parser("benchmark", help="benchmark prefill and decode separately")
    bench_cmd.add_argument("artifact")
    bench_cmd.add_argument("--prompt", default="CacheIR benchmark prompt")
    bench_cmd.add_argument("--decode-tokens", type=int, default=16)
    bench_cmd.add_argument("--repeats", type=int, default=3)
    bench_cmd.add_argument("--output", default=None)
    bench_cmd.set_defaults(func=_benchmark)

    export_cmd = sub.add_parser("export", help="export graph IR as html, dot, or cir text")
    export_cmd.add_argument("artifact")
    export_cmd.add_argument("output")
    export_cmd.add_argument("--mode", default="decode", choices=["prefill", "decode"])
    export_cmd.add_argument("--format", default=None, choices=["html", "dot", "cir", "txt", None])
    export_cmd.set_defaults(func=_export)

    profile_cmd = sub.add_parser("profile", help="print the local hardware profile")
    profile_cmd.set_defaults(func=_profile)

    serve_cmd = sub.add_parser("serve", help="start an OpenAI-compatible local server")
    serve_cmd.add_argument("artifact")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000)
    serve_cmd.set_defaults(func=_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
