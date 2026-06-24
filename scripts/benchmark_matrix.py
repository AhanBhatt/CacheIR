from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cacheir import compile_model
from cacheir.benchmark import run_benchmark
from cacheir.importers import create_tiny_model


@dataclass(frozen=True)
class ModelCase:
    name: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_seq: int


CASES = [
    ModelCase("tiny_1l_h16", 64, 16, 32, 1, 4, 2, 32),
    ModelCase("small_2l_h32", 128, 32, 64, 2, 4, 2, 64),
    ModelCase("medium_4l_h64", 256, 64, 128, 4, 8, 4, 96),
]

PROMPTS = {
    "short": "CacheIR",
    "medium": "CacheIR benchmark prompt for prefill and decode specialization.",
    "long": "CacheIR " * 32,
}


def run_matrix(output: Path, repeats: int, decode_tokens: int, include_quant: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="cacheir-bench-") as tmp:
        root = Path(tmp)
        for case in CASES:
            model_path = root / case.name
            create_tiny_model(
                model_path,
                vocab_size=case.vocab_size,
                hidden_size=case.hidden_size,
                intermediate_size=case.intermediate_size,
                num_layers=case.num_layers,
                num_attention_heads=case.num_attention_heads,
                num_key_value_heads=case.num_key_value_heads,
            )
            variants = [("fp32", None)]
            if include_quant:
                variants.append(("int4_awq", "int4_awq"))
            for variant_name, quant in variants:
                artifact = compile_model(
                    model_path,
                    target="cpu",
                    quant=quant,
                    mode=["prefill", "decode"],
                    max_seq=case.max_seq,
                )
                for prompt_name, prompt in PROMPTS.items():
                    result = run_benchmark(
                        artifact,
                        prompt=prompt,
                        decode_tokens=decode_tokens,
                        repeats=repeats,
                    )
                    rows.append(
                        {
                            "case": asdict(case),
                            "variant": variant_name,
                            "prompt": prompt_name,
                            **result.to_dict(),
                        }
                    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(rows), encoding="utf-8")
    return rows


def _markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# CacheIR Benchmark Matrix",
        "",
        "| Model | Variant | Prompt | Prefill ms | Decode ms/token | Prefill tok/s | Decode tok/s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        case = row["case"]
        assert isinstance(case, dict)
        lines.append(
            "| {model} | {variant} | {prompt} | {prefill_ms:.3f} | {decode_ms:.3f} | {prefill_tps:.1f} | {decode_tps:.1f} |".format(
                model=case["name"],
                variant=row["variant"],
                prompt=row["prompt"],
                prefill_ms=float(row["prefill_ms_avg"]),
                decode_ms=float(row["decode_ms_avg"]),
                prefill_tps=float(row["prefill_tokens_per_s"]),
                decode_tps=float(row["decode_tokens_per_s"]),
            )
        )
    lines.append("")
    lines.append("Generated with `scripts/benchmark_matrix.py` against the NumPy CPU reference backend.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CacheIR reference benchmark matrix")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--no-quant", action="store_true", help="skip int4_awq simulated quantized variant")
    args = parser.parse_args()
    rows = run_matrix(args.output, repeats=args.repeats, decode_tokens=args.decode_tokens, include_quant=not args.no_quant)
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.md')}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
