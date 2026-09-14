from __future__ import annotations

from html import escape
from pathlib import Path

from .jsonio import read_json


def write_report(root: Path, output: Path) -> dict:
    result = read_json(root / "result.json")
    cases = read_json(root / "metrics/cases.json")["cases"]

    def table(values: dict) -> str:
        rows = "".join(
            f"<tr><th>{escape(key)}</th><td>{'Not submitted' if value is None else escape(format(value, '.8g'))}</td></tr>"
            for key, value in values.items()
        )
        return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"

    details = "".join(
        f"<details><summary>{escape(case['case_id'])}</summary>{table(case['metric_values'])}</details>"
        for case in cases
    )
    title = escape(result["submission"]["method_name"])
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HLPW6 — {title}</title><style>body{{font:16px/1.5 system-ui;margin:3rem auto;padding:0 1rem;max-width:960px;color:#14283f}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{padding:.6rem;border-bottom:1px solid #dce3eb;text-align:left}}td{{font-variant-numeric:tabular-nums}}summary{{cursor:pointer;padding:1rem;background:#f0f5fa}}p{{max-width:75ch}}</style>
<h1>HLPW6 AI/ML · {title}</h1><p>{escape(result["split"]["label"])} · {escape(result["prediction_scope"])} · {len(cases)} cases</p>
<p>Evaluation: <strong>{escape(result["evaluation_kind"])}</strong>. Maximum score {result["scoring"]["maximum_score"]}/100; weights are fixed. Missing volume predictions receive no points.</p>
<h2>Split results</h2>{table(result["metric_values"])}<h2>Case results</h2>{details}
<p>Evaluator: {escape(result["evaluator"]["version"])}. This report stays on your computer.</p></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(document)
    return {"report": str(output)}
