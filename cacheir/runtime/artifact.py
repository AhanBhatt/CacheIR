from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cacheir.importers.hf import ModelConfig
from cacheir.ir import Graph


@dataclass
class CompileArtifact:
    target: str
    quant: str | None
    model_path: str
    config: ModelConfig
    graphs: dict[str, Graph]
    pass_traces: dict[str, list[dict[str, Any]]]
    version: str = "0.1"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def graph(self, mode: str = "decode") -> Graph:
        if mode in self.graphs:
            return self.graphs[mode]
        if self.graphs:
            return next(iter(self.graphs.values()))
        raise KeyError("artifact has no graphs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "target": self.target,
            "quant": self.quant,
            "model_path": self.model_path,
            "config": self.config.to_dict(),
            "graphs": {mode: graph.to_dict() for mode, graph in self.graphs.items()},
            "pass_traces": self.pass_traces,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompileArtifact":
        return cls(
            version=data.get("version", "0.1"),
            created_at=data.get("created_at", ""),
            target=data.get("target", "cpu"),
            quant=data.get("quant"),
            model_path=data.get("model_path", ""),
            config=ModelConfig.from_dict(data["config"]),
            graphs={mode: Graph.from_dict(graph) for mode, graph in data.get("graphs", {}).items()},
            pass_traces={mode: list(traces) for mode, traces in data.get("pass_traces", {}).items()},
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    def save_bundle(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.save(out / "artifact.json")
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "version": self.version,
                    "target": self.target,
                    "quant": self.quant,
                    "model_path": self.model_path,
                    "modes": sorted(self.graphs),
                    "files": ["artifact.json", "manifest.json", "README.txt"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out / "README.txt").write_text(
            "CacheIR artifact bundle\n\n"
            "artifact.json: full machine-readable compiler artifact\n"
            "graphs/*.cir: final optimized IR text by mode\n"
            "schedules/*.json: runtime kernel schedule by mode\n"
            "passes/*.diff: pass-by-pass IR diffs\n",
            encoding="utf-8",
        )
        graphs_dir = out / "graphs"
        schedules_dir = out / "schedules"
        passes_dir = out / "passes"
        graphs_dir.mkdir(exist_ok=True)
        schedules_dir.mkdir(exist_ok=True)
        passes_dir.mkdir(exist_ok=True)
        for mode, graph in self.graphs.items():
            (graphs_dir / f"{mode}.cir").write_text(graph.to_text(), encoding="utf-8")
            (schedules_dir / f"{mode}.json").write_text(
                json.dumps(graph.attrs.get("execution_schedule", []), indent=2),
                encoding="utf-8",
            )
            for trace in self.pass_traces.get(mode, []):
                name = str(trace.get("name", "pass")).replace("/", "_")
                diff = str(trace.get("diff") or trace.get("after") or "")
                (passes_dir / f"{mode}.{name}.diff").write_text(diff, encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "CompileArtifact":
        artifact_path = Path(path)
        if artifact_path.is_dir():
            artifact_path = artifact_path / "artifact.json"
        return cls.from_dict(json.loads(artifact_path.read_text(encoding="utf-8")))
