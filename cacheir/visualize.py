from __future__ import annotations

import html
from pathlib import Path

from cacheir.runtime.artifact import CompileArtifact


def graph_to_dot(artifact: CompileArtifact, mode: str = "decode") -> str:
    graph = artifact.graph(mode)
    lines = [f'digraph "{graph.name}" {{', "  rankdir=LR;", '  node [shape=box, fontname="Consolas"];']
    for input_name in graph.inputs:
        lines.append(f'  "{input_name}" [shape=oval, label="%{input_name}\\ninput"];')
    for node in graph.nodes:
        node_id = node.name or ".".join(node.outputs)
        label = f"{node.label()}\\n{node.op}\\n{node.attrs.get('kernel', '')}"
        lines.append(f'  "{node_id}" [label="{_dot_escape(label)}"];')
        for inp in node.inputs:
            source = _producer_id(graph, inp) or inp
            lines.append(f'  "{source}" -> "{node_id}" [label="%{_dot_escape(inp)}"];')
    for output in graph.outputs:
        source = _producer_id(graph, output) or output
        lines.append(f'  "{source}" -> "output:{output}";')
        lines.append(f'  "output:{output}" [shape=oval, label="%{output}\\noutput"];')
    lines.append("}")
    return "\n".join(lines)


def graph_to_html(artifact: CompileArtifact, mode: str = "decode") -> str:
    graph = artifact.graph(mode)
    schedule = graph.attrs.get("execution_schedule", [])
    rows = []
    for item in schedule:
        rows.append(
            "<tr>"
            f"<td>{item.get('step')}</td>"
            f"<td>{html.escape(str(item.get('name')))}</td>"
            f"<td>{html.escape(str(item.get('op')))}</td>"
            f"<td>{html.escape(str(item.get('kernel')))}</td>"
            f"<td>{html.escape(str(item.get('cost_class')))}</td>"
            f"<td>{item.get('estimated_bytes')}</td>"
            f"<td>{item.get('estimated_flops')}</td>"
            "</tr>"
        )
    pass_rows = []
    for trace in artifact.pass_traces.get(mode, []):
        changed = "yes" if trace.get("changed") else "no"
        remarks = "; ".join(str(item) for item in trace.get("remarks", []))
        pass_rows.append(
            "<tr>"
            f"<td>{html.escape(str(trace.get('name')))}</td>"
            f"<td>{changed}</td>"
            f"<td>{html.escape(remarks)}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CacheIR {html.escape(graph.name)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
    pre {{ background: #f5f7fa; padding: 16px; overflow-x: auto; border: 1px solid #d9e2ec; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; font-size: 13px; }}
    th {{ background: #eef2f7; }}
  </style>
</head>
<body>
  <h1>{html.escape(graph.name)}</h1>
  <p>mode={html.escape(graph.mode)} target={html.escape(graph.target)} quant={html.escape(str(artifact.quant))}</p>
  <h2>Execution Schedule</h2>
  <table>
    <thead><tr><th>Step</th><th>Name</th><th>Op</th><th>Kernel</th><th>Cost</th><th>Bytes</th><th>FLOPs</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Passes</h2>
  <table>
    <thead><tr><th>Pass</th><th>Changed</th><th>Remarks</th></tr></thead>
    <tbody>{''.join(pass_rows)}</tbody>
  </table>
  <h2>IR</h2>
  <pre>{html.escape(graph.to_text())}</pre>
</body>
</html>
"""


def export_graph(artifact: CompileArtifact, output: str | Path, mode: str = "decode", fmt: str | None = None) -> Path:
    out = Path(output)
    fmt = fmt or out.suffix.lstrip(".").lower() or "html"
    if fmt == "dot":
        content = graph_to_dot(artifact, mode)
    elif fmt == "html":
        content = graph_to_html(artifact, mode)
    elif fmt in {"cir", "txt"}:
        content = artifact.graph(mode).to_text()
    else:
        raise ValueError(f"Unsupported graph export format {fmt!r}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out


def _producer_id(graph, value: str) -> str | None:
    producer = graph.producer_map().get(value)
    return producer.name if producer else None


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
