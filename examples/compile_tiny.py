from pathlib import Path

from cacheir import Runtime, compile_model
from cacheir.importers import create_tiny_model


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "tiny_model"
ARTIFACT = ROOT / "tiny_cacheir_artifact.json"


def main() -> None:
    create_tiny_model(MODEL)
    artifact = compile_model(MODEL, target="cpu", mode=["prefill", "decode"], max_seq=32, output=ARTIFACT)
    print(f"wrote {ARTIFACT}")
    for mode, graph in artifact.graphs.items():
        print(f"{mode}: {len(graph.nodes)} nodes")
    runtime = Runtime(artifact)
    print("sample:", "".join(runtime.generate("CacheIR", max_new_tokens=8)))


if __name__ == "__main__":
    main()
