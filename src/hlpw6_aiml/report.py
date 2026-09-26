from __future__ import annotations

from html import escape
from pathlib import Path

from .contracts import scoring
from .jsonio import read_json
from .presentation import metric_label


def write_report(root: Path, output: Path) -> dict:
    result = read_json(root / "result.json")
    cases = read_json(root / "metrics/cases.json")["cases"]
    definitions = {item["id"]: item for item in scoring()["metrics"]}

    def table(values: dict) -> str:
        rows = []
        for key, value in values.items():
            definition = definitions.get(key, {})
            unit = definition.get("unit", "")
            label = metric_label(key) + (f" ({unit})" if unit else "")
            direction = definition.get("direction")
            better = f"{direction.capitalize()} is better" if direction else "—"
            display = "Not submitted" if value is None else format(value, ".8g")
            rows.append(
                f"<tr><th scope='row'>{escape(label)}</th><td>{escape(display)}</td>"
                f"<td>{escape(better)}</td></tr>"
            )
        return (
            "<table><thead><tr><th>Metric</th><th>Value</th><th>Direction</th></tr>"
            f"</thead><tbody>{''.join(rows)}</tbody></table>"
        )

    details = "".join(
        f"<details><summary>{escape(case['case_id'])}</summary>{table(case['metric_values'])}</details>"
        for case in cases
    )
    keys = dict.fromkeys(
        [*result["metric_values"], *(key for case in cases for key in case["metric_values"])]
    )
    identifiers = "".join(
        f"<tr><th scope='row'>{escape(metric_label(key))}</th>"
        f"<td><code>{escape(key)}</code></td>"
        f"<td>{escape(definitions.get(key, {}).get('weighting', '—').replace('_', ' '))}</td>"
        f"<td>{escape(definitions.get(key, {}).get('aggregation', '—').replace('_', ' '))}</td></tr>"
        for key in keys
    )
    title = escape(result["submission"]["method_name"])
    scope = {"surface_only": "Surface only", "surface_and_volume": "Surface + volume"}.get(
        result["prediction_scope"], result["prediction_scope"]
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HLPW6 — {title}</title><style>body{{font:16px/1.5 system-ui;margin:3rem auto;padding:0 1rem;max-width:960px;color:#14283f}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{padding:.6rem;border-bottom:1px solid #dce3eb;text-align:left}}td{{font-variant-numeric:tabular-nums}}code{{overflow-wrap:anywhere}}summary{{cursor:pointer;padding:1rem;background:#f0f5fa}}p{{max-width:75ch}}</style>
<h1>HLPW6 AI/ML · {title}</h1><p>{escape(result["split"]["label"])} · {escape(scope)} · {len(cases)} cases</p>
<p>Evaluation: <strong>{escape(result["evaluation_kind"])}</strong>. Maximum score {result["scoring"]["maximum_score"]}/100; weights are fixed. Missing volume predictions receive no points.</p>
<h2>Split results</h2>{table(result["metric_values"])}<h2>Case results</h2>{details}
<details><summary>Metric IDs and weighting</summary><p>Primary surface errors use nodal dual-area weights; equal-entity errors weight each surface point equally. Volume errors weight valid points equally. MAE means mean absolute error; RMSE means root mean squared error.</p><table><thead><tr><th>Metric</th><th>ID</th><th>Weighting</th><th>Aggregation</th></tr></thead><tbody>{identifiers}</tbody></table></details>
<p>Evaluator: {escape(result["evaluator"]["version"])}. This report stays on your computer.</p></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(document)
    return {"report": str(output)}
