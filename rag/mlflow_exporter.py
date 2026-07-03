"""
MLflow Exporter Module for RAG Knowledge Base Construction (Task E1).

Exports MLflow tracking runs (parameters, metrics, tags, artifact summaries)
to structured Markdown files inside `mlflow_export/` so they can be section-chunked.
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import json
from pathlib import Path
from typing import List, Dict, Any
import mlflow
from mlflow.tracking import MlflowClient


def export_mlflow_runs(export_dir: str = "mlflow_export", tracking_uri: str = None) -> List[str]:
    """
    Export all active MLflow runs from the tracking server to structured markdown files.

    Args:
        export_dir: Destination directory for exported text/markdown files.
        tracking_uri: Optional MLflow tracking URI. Defaults to current URI or ./mlruns.

    Returns:
        List of filepaths created in export_dir.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    runs_dir = Path(export_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    summaries_dir = Path(export_dir) / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    exported_files = []

    try:
        experiments = client.search_experiments()
    except Exception as e:
        print(f"[WARNING] Could not search MLflow experiments: {e}")
        return exported_files

    if not experiments:
        print("[WARNING] No MLflow experiments found.")
        return exported_files

    all_runs_summary = []

    for exp in experiments:
        try:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
        except Exception as e:
            print(f"[WARNING] Could not fetch runs for experiment {exp.name}: {e}")
            continue

        for run in runs:
            run_id = run.info.run_id
            run_name = run.data.tags.get("mlflow.runName", run_id)
            status = run.info.status

            params = run.data.params
            metrics = run.data.metrics
            tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.log-model")}

            # Create structured markdown for section chunking
            lines = [
                f"# Experiment Run: {run_name} ({exp.name})",
                "",
                f"**Run ID**: {run_id}  ",
                f"**Experiment**: {exp.name}  ",
                f"**Status**: {status}  ",
                "",
                "## Run Parameters",
                ""
            ]

            if params:
                for k in sorted(params.keys()):
                    lines.append(f"- **{k}**: {params[k]}")
            else:
                lines.append("No parameters recorded.")

            lines.append("")
            lines.append("## Run Metrics")
            lines.append("")

            if metrics:
                for k in sorted(metrics.keys()):
                    lines.append(f"- **{k}**: {metrics[k]:.6f}")
            else:
                lines.append("No metrics recorded.")

            lines.append("")
            lines.append("## Run Tags")
            lines.append("")
            if tags:
                for k in sorted(tags.keys()):
                    lines.append(f"- **{k}**: {tags[k]}")
            else:
                lines.append("No custom tags recorded.")

            file_path = runs_dir / f"run_{run_id}.md"
            file_path.write_text("\n".join(lines), encoding="utf-8")
            exported_files.append(str(file_path))

            all_runs_summary.append({
                "run_id": run_id,
                "run_name": run_name,
                "experiment_name": exp.name,
                "params": params,
                "metrics": metrics
            })

    # Create global summary file
    summary_path = summaries_dir / "all_runs_summary.md"
    summary_lines = [
        "# MLflow Experiments Overview",
        "",
        "Summary of all tracked runs across adversarial ML experiments.",
        ""
    ]
    for r in all_runs_summary:
        summary_lines.append(f"## {r['experiment_name']} - {r['run_name']}")
        summary_lines.append(f"- **Run ID**: {r['run_id']}")
        for m_name, m_val in r.get("metrics", {}).items():
            summary_lines.append(f"- **{m_name}**: {m_val:.4f}")
        summary_lines.append("")

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    exported_files.append(str(summary_path))

    return exported_files


if __name__ == "__main__":
    files = export_mlflow_runs()
    print(f"Exported {len(files)} MLflow summary files.")
