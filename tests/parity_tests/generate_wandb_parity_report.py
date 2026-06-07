"""Generate a W&B Report for recent vLLM-Hook parity runs.

The report focuses on the most recent Metal/MLX and GPU/non-Metal run from
each selected W&B project and highlights tensor/list L2 metrics plus attention
invariance risks.

Example:
  python tests/parity_tests/generate_wandb_parity_report.py \
    --wandb-entity tm8ctgzqj8-georgia-institute-of-technology \
    --report-project vllm-hook-platform-parity \
    --project attntracker --project corereranker --project steering
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PARITY_DIR = Path(__file__).resolve().parent
LOCAL_WANDB_CONFIG = PARITY_DIR / "local_wandb_config.py"
DEFAULT_PROJECTS = ("hiddenstates", "attntracker", "corereranker", "steering")
L2_KEYS = (
    "q_l2_mean",
    "k_l2_mean",
    "hidden_state_l2_mean",
    "score_l2",
    "steering_vector_l2",
)
PARITY_KEYS = (
    "attn_score_mean",
    "score_margin_top1_top2",
    "text_changed_by_steering",
)
ATTENTION_PROJECT_HINTS = ("attn", "attention")
NUMERIC_TOLERANCE = 1e-5


@dataclass(frozen=True)
class RunSnapshot:
    project: str
    name: str
    run_id: str
    url: str
    created_at: str
    group: str
    job_type: str
    tags: tuple[str, ...]
    config: dict[str, Any]
    summary: dict[str, Any]


def load_local_wandb_config() -> dict[str, str]:
    if not LOCAL_WANDB_CONFIG.exists():
        return {}
    spec = importlib.util.spec_from_file_location("local_wandb_config", LOCAL_WANDB_CONFIG)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {}
    placeholders = {"paste_your_wandb_key_here", "your_key_here", ""}
    for name in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"):
        value = str(getattr(module, name, "")).strip()
        if value and value not in placeholders and "paste_your" not in value:
            values[name] = value
    return values


def parse_args() -> argparse.Namespace:
    local_config = load_local_wandb_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        action="append",
        dest="projects",
        help=(
            "W&B project to include. Repeat for multiple projects. Defaults "
            "to hiddenstates, attntracker, corereranker, and steering."
        ),
    )
    parser.add_argument(
        "--discover-tag",
        default="",
        help=(
            "Also include any accessible W&B project with at least one run "
            "tagged with this value, for example minimal-parity."
        ),
    )
    parser.add_argument(
        "--runs-per-project",
        type=int,
        default=500,
        help=(
            "Number of recent runs to scan per project when selecting the "
            "latest Metal/MLX and GPU/non-Metal run."
        ),
    )
    parser.add_argument(
        "--report-project",
        default=local_config.get(
            "WANDB_PROJECT", os.environ.get("WANDB_REPORT_PROJECT", "vllm-hook-platform-parity")
        ),
        help="W&B project where the generated report is saved.",
    )
    parser.add_argument("--report-title", default="vLLM-Hook Platform Parity Findings")
    parser.add_argument(
        "--report-width",
        choices=("readable", "fixed", "fluid"),
        default=os.environ.get("WANDB_REPORT_WIDTH", "fluid"),
        help="W&B report page width. Use fluid for less narrow margins.",
    )
    parser.add_argument("--wandb-entity", default=local_config.get("WANDB_ENTITY", os.environ.get("WANDB_ENTITY", "")))
    parser.add_argument("--wandb-api-key", default=local_config.get("WANDB_API_KEY", os.environ.get("WANDB_API_KEY", "")))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Markdown report body without creating a W&B report.",
    )
    args = parser.parse_args()
    if args.runs_per_project < 1:
        raise SystemExit("--runs-per-project must be >= 1")
    return args


def require_wandb(args: argparse.Namespace):
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is required. Install with: pip install wandb") from exc

    os.environ["WANDB_MODE"] = args.wandb_mode
    if args.wandb_api_key and args.wandb_mode == "online":
        wandb.login(key=args.wandb_api_key)
    return wandb


def require_reports_api():
    try:
        import wandb_workspaces.reports.v2 as wr
    except ImportError as exc:
        raise SystemExit(
            "wandb-workspaces is required for the Reports API. Install with: "
            "pip install wandb-workspaces"
        ) from exc
    return wr


def run_summary(run: Any) -> dict[str, Any]:
    summary = getattr(run, "summary", {}) or {}
    if hasattr(summary, "_json_dict"):
        summary = summary._json_dict
    return {
        key: value
        for key, value in dict(summary).items()
        if not str(key).startswith("_")
    }


def run_config(run: Any) -> dict[str, Any]:
    config = getattr(run, "config", {}) or {}
    return {
        key: value
        for key, value in dict(config).items()
        if not str(key).startswith("_")
    }


def snapshot(project: str, run: Any) -> RunSnapshot:
    return RunSnapshot(
        project=project,
        name=str(getattr(run, "name", "") or getattr(run, "display_name", "") or getattr(run, "id", "")),
        run_id=str(getattr(run, "id", "")),
        url=str(getattr(run, "url", "")),
        created_at=str(getattr(run, "created_at", "")),
        group=str(getattr(run, "group", "")),
        job_type=str(getattr(run, "job_type", "")),
        tags=tuple(str(tag) for tag in (getattr(run, "tags", []) or [])),
        config=run_config(run),
        summary=run_summary(run),
    )


def platform_key(run: RunSnapshot) -> str:
    backend = str(metric_value(run, "backend") or run.config.get("backend", "")).lower()
    hardware_kind = str(
        metric_value(run, "hardware_kind") or run.config.get("hardware_kind", "")
    ).lower()
    hardware_label = str(
        metric_value(run, "hardware_label") or run.config.get("hardware_label", "")
    ).lower()
    tags = {tag.lower() for tag in run.tags}
    name = run.name.lower()

    values = {backend, hardware_kind, hardware_label, *tags, name}
    joined = " ".join(value for value in values if value)
    if any(token in joined for token in ("metal", "mlx", "apple-metal")):
        return "metal"
    if any(token in joined for token in ("gpu", "cuda", "non-metal", "colab", "t4")):
        return "gpu"
    return "unknown"


PLATFORM_FILTERS = {
    "metal": (
        {"config.backend": "metal"},
        {"config.hardware_kind": {"$in": ["metal", "mlx"]}},
        {"tags": {"$in": ["metal", "mlx", "apple-metal"]}},
    ),
    "gpu": (
        {"config.backend": "non-metal"},
        {"config.hardware_kind": {"$in": ["gpu", "cuda"]}},
        {"tags": {"$in": ["non-metal", "gpu", "cuda", "colab", "t4"]}},
    ),
}


def fetch_latest_filtered_platform_run(
    api: Any, path: str, project: str, platform: str
) -> RunSnapshot | None:
    candidates: list[RunSnapshot] = []
    for filters in PLATFORM_FILTERS[platform]:
        try:
            runs = api.runs(path, filters=filters, order="-created_at", per_page=1)
            for run in runs:
                snap = snapshot(project, run)
                if platform_key(snap) == platform:
                    return snap
                candidates.append(snap)
                break
        except Exception:
            continue
    return candidates[0] if candidates else None


def fetch_latest_platform_runs(
    api: Any, entity: str, project: str, scan_limit: int
) -> list[RunSnapshot]:
    path = f"{entity}/{project}" if entity else project
    latest: dict[str, RunSnapshot] = {}
    unknown: list[RunSnapshot] = []
    for key in ("metal", "gpu"):
        run = fetch_latest_filtered_platform_run(api, path, project, key)
        if run is not None:
            latest[key] = run
    if "metal" in latest and "gpu" in latest:
        return [latest["metal"], latest["gpu"]]

    try:
        for index, run in enumerate(api.runs(path, order="-created_at", per_page=100)):
            if index >= scan_limit:
                break
            snap = snapshot(project, run)
            key = platform_key(snap)
            if key in {"metal", "gpu"} and key not in latest:
                latest[key] = snap
            elif key == "unknown":
                unknown.append(snap)
            if "metal" in latest and "gpu" in latest:
                break
    except Exception as exc:
        print(f"Skipping {project}: {type(exc).__name__}: {exc}", flush=True)
        return []

    selected = []
    for key in ("metal", "gpu"):
        if key in latest:
            selected.append(latest[key])
    if not selected and unknown:
        selected.append(unknown[0])
    return selected


def discover_projects(api: Any, entity: str, tag: str) -> list[str]:
    if not tag:
        return []
    try:
        projects = list(api.projects(entity=entity)) if entity else list(api.projects())
    except Exception as exc:
        try:
            projects = list(api.projects(entity)) if entity else list(api.projects())
        except Exception:
            print(f"Project discovery skipped: {type(exc).__name__}: {exc}", flush=True)
            return []

    discovered: list[str] = []
    for project in projects:
        name = str(getattr(project, "name", ""))
        if not name:
            continue
        path = f"{entity}/{name}" if entity else name
        try:
            tagged = list(api.runs(path, filters={"tags": {"$in": [tag]}}, per_page=1))
        except Exception:
            tagged = []
        if tagged:
            discovered.append(name)
    return discovered


def clean_scalar(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    return json.dumps(value, sort_keys=True, default=str)


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def metric_value(run: RunSnapshot, key: str) -> Any:
    if key in run.summary:
        return run.summary[key]
    return run.config.get(key)


def metric_label(key: str) -> str:
    labels = {
        "q_l2_mean": "Q L2 mean",
        "k_l2_mean": "K L2 mean",
        "hidden_state_l2_mean": "Hidden-state L2 mean",
        "score_l2": "Score L2",
        "steering_vector_l2": "Steering-vector L2",
        "attn_score_mean": "Attention score mean",
        "score_margin_top1_top2": "Top-1 / top-2 score margin",
        "text_changed_by_steering": "Text changed by steering",
    }
    return labels.get(key, key.replace("_", " ").strip().title())


def display_backend_label(run: RunSnapshot) -> str:
    backend = str(metric_value(run, "backend") or run.config.get("backend", "")).strip().lower()
    if backend in {"metal", "mlx", "apple-metal"}:
        return "Metal"
    if backend in {"non-metal", "gpu", "cuda", "colab", "t4"}:
        return "GPU"
    key = platform_key(run)
    if key == "metal":
        return "Metal"
    if key == "gpu":
        return "GPU"
    return "Run"


def metric_keys(runs: list[RunSnapshot]) -> list[str]:
    keys = set()
    for run in runs:
        keys.update(key for key in run.summary if key in L2_KEYS or key in PARITY_KEYS)
        keys.update(key for key in run.config if key in L2_KEYS or key in PARITY_KEYS)
    ordered = [key for key in (*L2_KEYS, *PARITY_KEYS) if key in keys]
    return ordered


def run_display_label(run: RunSnapshot, index: int | None = None) -> str:
    backend = display_backend_label(run)
    if backend != "Run":
        return backend
    if index is not None:
        return f"Run {index + 1}"
    return "Run"


def format_number(value: Any) -> str:
    value = clean_scalar(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def escape_table_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def render_card(text: str) -> str:
    return (
        '<div style="margin: 18px 0; padding: 18px 20px; border: 1px solid #e5e7eb; '
        'border-radius: 16px; background: #ffffff;">\n'
        f"{text}\n"
        "</div>"
    )


def render_title_block(title: str, subtitle: str, generated_at: str) -> str:
    content = (
        '<div style="text-align:center; margin: 18px 0 28px; padding: 22px 20px;">\n'
        f'<h1 style="margin: 0 0 8px;">{html.escape(title)}</h1>\n'
        f'<div style="margin: 0 0 6px; font-size: 1.05em; color: #4b5563;">{html.escape(subtitle)}</div>\n'
        f'<div style="font-size: 0.9em; color: #6b7280;">Generated: {html.escape(generated_at)}</div>\n'
        "</div>"
    )
    return content


def render_section_header(project: str) -> str:
    return render_card(
        f'<h2 style="margin: 0 0 6px;">{html.escape(project)}</h2>\n'
        '<div style="height: 1px; background: #e5e7eb; margin: 12px 0 0;"></div>'
    )


def render_metric_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    head_cells = "".join(
        f'<th style="text-align: {("left" if idx == 0 else "right")}; padding: 8px 10px; border-bottom: 1px solid #d1d5db; white-space: nowrap;">{html.escape(header)}</th>'
        for idx, header in enumerate(headers)
    )
    body_rows = []
    for row in rows:
        row_cells = "".join(
            f'<td style="text-align: {("left" if idx == 0 else "right")}; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; white-space: nowrap;">{html.escape(value)}</td>'
            for idx, value in enumerate(row)
        )
        body_rows.append(f"<tr>{row_cells}</tr>")
    table = (
        '<table style="display: inline-table; width: auto; max-width: 100%; border-collapse: collapse; font-size: 0.95em;">'
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )
    return render_card(
        f'<h3 style="margin: 0 0 12px;">{html.escape(title)}</h3>\n'
        f"{table}"
    )


def render_chart_container(title: str) -> str:
    return render_card(
        f'<h3 style="text-align:center; margin: 0 0 12px;">{html.escape(title)}</h3>'
    )


def project_summary_markdown(project: str, runs: list[RunSnapshot]) -> str:
    keys = metric_keys(runs)
    if not runs:
        return render_card(
            f'<h3 style="margin: 0 0 12px;">Run Summary</h3>\n'
            '<div style="color: #6b7280;">No recent runs were found.</div>'
        )

    headers = ["Backend", *[metric_label(key) for key in keys]]
    rows: list[list[str]] = []
    for index, run in enumerate(runs):
        backend = run_display_label(run, index)
        row = [backend, *[format_number(metric_value(run, key)) for key in keys]]
        rows.append(row)
    return render_metric_table("Run Summary", headers, rows)


def project_delta_markdown(project: str, runs: list[RunSnapshot]) -> str:
    if len(runs) < 2:
        return ""
    left, right = runs[0], runs[1]
    keys = metric_keys(runs)
    left_label = display_backend_label(left)
    right_label = display_backend_label(right)
    rows = []
    for key in keys:
        left_value = numeric(metric_value(left, key))
        right_value = numeric(metric_value(right, key))
        if left_value is None or right_value is None:
            continue
        absolute = abs(left_value - right_value)
        relative = absolute / max(abs(right_value), NUMERIC_TOLERANCE)
        rows.append(
            [
                metric_label(key),
                format_number(left_value),
                format_number(right_value),
                format_number(absolute),
                format_number(relative),
            ]
        )
    if not rows:
        return ""
    return render_metric_table(
        "Metrics",
        ["Metric", left_label, right_label, "Abs Delta", "Relative Delta"],
        rows,
    )


def executive_summary_markdown(project_runs: dict[str, list[RunSnapshot]]) -> str:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    project_list = ", ".join(project_runs)
    subtitle = f"Projects included: {project_list}"
    return render_title_block("vLLM-Hook Platform Parity Findings", subtitle, created)


def markdown_report(project_runs: dict[str, list[RunSnapshot]]) -> str:
    sections = [executive_summary_markdown(project_runs)]
    for project, runs in project_runs.items():
        sections.append(render_section_header(project))
        sections.append(project_summary_markdown(project, runs))
        delta = project_delta_markdown(project, runs)
        if delta:
            sections.append(delta)
            sections.append(render_chart_container("Visualizations"))
        sections.append('<div style="margin: 18px 0;"><hr style="border: 0; border-top: 1px solid #e5e7eb;" /></div>')
    return "\n".join(sections)


def project_panel_grid(wr: Any, project: str, runs: list[RunSnapshot], entity: str) -> Any | None:
    runset_cls = getattr(wr, "Runset", getattr(wr, "RunSet", None))
    if runset_cls is None or not hasattr(wr, "PanelGrid"):
        return None
    names = [run.name for run in runs if run.name]
    if not names:
        return None
    line_plot_cls = getattr(wr, "LinePlot", None)
    if line_plot_cls is None:
        return None
    keys = [key for key in metric_keys(runs) if key in L2_KEYS or key in PARITY_KEYS]
    panels = [line_plot_cls(title=metric_label(key), x="Step", y=[key]) for key in keys[:6]]
    if not panels:
        return None
    filters = f"Metric('displayName') in {json.dumps(names)}"
    return wr.PanelGrid(
        runsets=[runset_cls(project=project, entity=entity or None, filters=filters)],
        panels=panels,
    )


def project_delta_block(wr: Any, project: str, runs: list[RunSnapshot]) -> Any:
    text = project_delta_markdown(project, runs)
    if not text:
        return None
    return wr.MarkdownBlock(text=text)


def executive_summary_block(wr: Any, project_runs: dict[str, list[RunSnapshot]]) -> Any:
    return wr.MarkdownBlock(text=executive_summary_markdown(project_runs))


def render_project_card(wr: Any, project: str, runs: list[RunSnapshot], entity: str) -> list[Any]:
    blocks: list[Any] = []
    blocks.append(wr.MarkdownBlock(text=render_section_header(project)))
    blocks.append(wr.MarkdownBlock(text=project_summary_markdown(project, runs)))
    delta_block = project_delta_block(wr, project, runs)
    if delta_block is not None:
        blocks.append(delta_block)
    panel_grid = project_panel_grid(wr, project, runs, entity)
    if panel_grid is not None:
        blocks.append(wr.MarkdownBlock(text=render_chart_container("Visualizations")))
        blocks.append(panel_grid)
    blocks.append(wr.MarkdownBlock(text='<div style="margin: 18px 0;"><hr style="border: 0; border-top: 1px solid #e5e7eb;" /></div>'))
    return blocks


def report_blocks(
    wr: Any,
    project_runs: dict[str, list[RunSnapshot]],
    entity: str,
) -> list[Any]:
    blocks = [executive_summary_block(wr, project_runs)]
    for project, runs in project_runs.items():
        blocks.extend(render_project_card(wr, project, runs, entity))
    return blocks


def create_report(args: argparse.Namespace, project_runs: dict[str, list[RunSnapshot]]) -> str:
    wr = require_reports_api()
    kwargs: dict[str, Any] = {
        "project": args.report_project,
        "title": args.report_title,
        "description": "Automated parity summary for recent vLLM-Hook runs.",
        "width": "fluid",
    }
    if args.wandb_entity:
        kwargs["entity"] = args.wandb_entity
    report = wr.Report(**kwargs)
    report.blocks = report_blocks(wr, project_runs, args.wandb_entity)
    saved = report.save()
    url = str(getattr(report, "url", "") or getattr(saved, "url", "") or "")
    return url


def selected_projects(args: argparse.Namespace, api: Any) -> list[str]:
    projects = list(args.projects or DEFAULT_PROJECTS)
    projects.extend(discover_projects(api, args.wandb_entity, args.discover_tag))
    seen = set()
    unique = []
    for project in projects:
        if project not in seen:
            seen.add(project)
            unique.append(project)
    return unique


def main() -> int:
    args = parse_args()
    if args.wandb_mode == "disabled" and not args.dry_run:
        raise SystemExit("Report creation requires --wandb-mode online or offline.")
    wandb = require_wandb(args)
    api = wandb.Api()
    projects = selected_projects(args, api)
    project_runs = {
        project: fetch_latest_platform_runs(
            api, args.wandb_entity, project, args.runs_per_project
        )
        for project in projects
    }
    markdown = markdown_report(project_runs)
    if args.dry_run:
        print(markdown)
        return 0
    url = create_report(args, project_runs)
    if url:
        print(f"Created W&B report: {url}")
    else:
        print("Created W&B report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
