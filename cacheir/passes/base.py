from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from cacheir.ir import Graph, graph_text_diff


@dataclass
class CompilerContext:
    target: str = "cpu"
    mode: str = "prefill"
    quant: str | None = None
    max_batch: int = 1
    max_seq: int = 128
    verbose: bool = False
    hardware_profile: dict[str, object] | None = None

    @property
    def symbols(self) -> dict[str, int]:
        seq = 1 if self.mode == "decode" else self.max_seq
        return {
            "batch": self.max_batch,
            "seq": seq,
            "cache_seq": self.max_seq,
        }


@dataclass
class PassResult:
    changed: bool = False
    remarks: list[str] = field(default_factory=list)


@dataclass
class PassTrace:
    name: str
    changed: bool
    remarks: list[str]
    before: str
    after: str
    diff: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "changed": self.changed,
            "remarks": self.remarks,
            "before": self.before,
            "after": self.after,
            "diff": self.diff,
        }


class Pass(Protocol):
    name: str

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        ...


class PassManager:
    def __init__(self, passes: list[Pass]):
        self.passes = passes

    def run(self, graph: Graph, context: CompilerContext) -> tuple[Graph, list[PassTrace]]:
        traces: list[PassTrace] = []
        for compiler_pass in self.passes:
            before = graph.clone()
            before_text = before.to_text()
            result = compiler_pass.run(graph, context)
            after_text = graph.to_text()
            traces.append(
                PassTrace(
                    name=compiler_pass.name,
                    changed=result.changed or before_text != after_text,
                    remarks=result.remarks,
                    before=before_text,
                    after=after_text,
                    diff=graph_text_diff(before, graph),
                )
            )
        graph.validate()
        return graph, traces
