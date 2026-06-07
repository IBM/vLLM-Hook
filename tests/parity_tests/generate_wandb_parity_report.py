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
    "hidden_state_l2_mean_from_qkv_x",
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
        default=50,
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


def fetch_latest_platform_runs(
    api: Any, entity: str, project: str, scan_limit: int
) -> list[RunSnapshot]:
    path = f"{entity}/{project}" if entity else project
    try:
        runs = [
            snapshot(project, run)
            for run in list(
                api.runs(path, order="-created_at", per_page=scan_limit)
            )[:scan_limit]
        ]
    except Exception as exc:
        print(f"Skipping {project}: {type(exc).__name__}: {exc}", flush=True)
        return []

    latest: dict[str, RunSnapshot] = {}
    unknown: list[RunSnapshot] = []
    for run in runs:
        key = platform_key(run)
        if key in {"metal", "gpu"} and key not in latest:
            latest[key] = run
        elif key == "unknown":
            unknown.append(run)
        if "metal" in latest and "gpu" in latest:
            break

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


def metric_keys(runs: list[RunSnapshot]) -> list[str]:
    keys = set()
    for run in runs:
        keys.update(key for key in run.summary if key in L2_KEYS or key in PARITY_KEYS)
        keys.update(key for key in run.config if key in L2_KEYS or key in PARITY_KEYS)
    ordered = [key for key in (*L2_KEYS, *PARITY_KEYS) if key in keys]
    return ordered


def project_table(project: str, runs: list[RunSnapshot]) -> str:
    keys = metric_keys(runs)
    if not runs:
        return f"### {project}\n\nNo recent runs were found.\n"
    header = ["Run", "Created", "Backend", "Group", *keys]
    rows = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for run in runs:
        backend = clean_scalar(metric_value(run, "backend") or run.config.get("backend", ""))
        run_label = f"[{run.name}]({run.url})" if run.url else run.name
        values = [
            run_label,
            run.created_at,
            str(backend),
            run.group,
            *[format_value(metric_value(run, key)) for key in keys],
        ]
        rows.append("| " + " | ".join(escape_table_cell(value) for value in values) + " |")
    return f"### {project}\n\n" + "\n".join(rows) + "\n"


def escape_table_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def format_value(value: Any) -> str:
    value = clean_scalar(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else str(value)


def delta_table(project: str, runs: list[RunSnapshot]) -> str:
    if len(runs) < 2:
        return ""
    left, right = runs[0], runs[1]
    keys = metric_keys(runs)
    left_label = str(
        metric_value(left, "backend") or left.config.get("backend", left.name)
    )
    right_label = str(
        metric_value(right, "backend") or right.config.get("backend", right.name)
    )
    rows = [
        f"| Metric | {escape_table_cell(left_label)} | "
        f"{escape_table_cell(right_label)} | Abs Delta | Relative Delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in keys:
        left_value = numeric(metric_value(left, key))
        right_value = numeric(metric_value(right, key))
        if left_value is None or right_value is None:
            continue
        absolute = abs(left_value - right_value)
        relative = absolute / max(abs(right_value), NUMERIC_TOLERANCE)
        rows.append(
            "| "
            + " | ".join(
                [
                    key,
                    f"{left_value:.6g}",
                    f"{right_value:.6g}",
                    f"{absolute:.6g}",
                    f"{relative:.6g}",
                ]
            )
            + " |"
        )
    if len(rows) == 2:
        return ""
    return f"#### {project} Deltas\n\n" + "\n".join(rows) + "\n"


def validation_notes(project_runs: dict[str, list[RunSnapshot]]) -> str:
    notes = [
        "## Statistical Invariance Checks",
        "",
        "The report intentionally separates raw L2 magnitudes from invariance claims. A small raw L2 delta is evidence for parity, but attention calculations can vary across Metal/CUDA kernels, precision modes, and softmax implementations. Treat attention score changes as requiring statistical invariance validation when both backends are present.",
        "",
        "| Project | Attention-like | Backends Observed | Recommendation |",
        "| --- | --- | --- | --- |",
    ]
    for project, runs in project_runs.items():
        backends = sorted(
            {
                str(metric_value(run, "backend") or run.config.get("backend", ""))
                for run in runs
                if metric_value(run, "backend") or run.config.get("backend", "")
            }
        )
        keys = set(metric_keys(runs))
        attention_like = any(hint in project.lower() for hint in ATTENTION_PROJECT_HINTS) or "attn_score_mean" in keys
        if attention_like and len(backends) >= 2:
            recommendation = "Validate distributional invariance across repeated seeds/prompts before claiming parity."
        elif attention_like:
            recommendation = "Add the missing backend or repeated runs before making a statistical claim."
        else:
            recommendation = "Use L2 and task-output deltas as primary parity evidence."
        notes.append(
            "| "
            + " | ".join(
                [
                    escape_table_cell(project),
                    "yes" if attention_like else "no",
                    escape_table_cell(", ".join(backends) or "unknown"),
                    escape_table_cell(recommendation),
                ]
            )
            + " |"
        )
    return "\n".join(notes) + "\n"


def markdown_report(project_runs: dict[str, list[RunSnapshot]]) -> str:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    project_list = ", ".join(project_runs)
    sections = [
        f"# vLLM-Hook Platform Parity Findings",
        "",
        f"Generated: {created}",
        "",
        f"Projects included: {project_list}",
        "",
        "This report captures the latest Metal/MLX run and the latest GPU/non-Metal run per project by default. It prioritizes tensor/list L2 magnitudes, task score deltas, and attention calculations that require invariance validation because backend math can differ across platforms.",
        "",
        "## Recent Runs",
        "",
    ]
    for project, runs in project_runs.items():
        sections.append(project_table(project, runs))
        delta = delta_table(project, runs)
        if delta:
            sections.append(delta)
    sections.append(validation_notes(project_runs))
    return "\n".join(sections)


def report_blocks(
    wr: Any,
    markdown: str,
    project_runs: dict[str, list[RunSnapshot]],
    entity: str,
) -> list[Any]:
    blocks = [wr.MarkdownBlock(text=markdown)]
    runset_cls = getattr(wr, "Runset", getattr(wr, "RunSet", None))
    if runset_cls is None or not hasattr(wr, "PanelGrid"):
        return blocks

    line_plot_cls = getattr(wr, "LinePlot", None)
    for project, runs in project_runs.items():
        if not runs:
            continue
        names = [run.name for run in runs if run.name]
        if not names:
            continue
        filters = f"Metric('displayName') in {json.dumps(names)}"
        panels = []
        if line_plot_cls is not None:
            keys = [key for key in metric_keys(runs) if key in L2_KEYS or key in PARITY_KEYS]
            for key in keys[:6]:
                panels.append(line_plot_cls(title=key, x="Step", y=[key]))
        blocks.append(
            wr.PanelGrid(
                runsets=[runset_cls(project=project, entity=entity or None, filters=filters)],
                panels=panels,
            )
        )
    return blocks


def create_report(args: argparse.Namespace, markdown: str, project_runs: dict[str, list[RunSnapshot]]) -> str:
    wr = require_reports_api()
    kwargs: dict[str, Any] = {
        "project": args.report_project,
        "title": args.report_title,
        "description": "Automated parity summary for recent vLLM-Hook runs.",
        "width": args.report_width,
    }
    if args.wandb_entity:
        kwargs["entity"] = args.wandb_entity
    report = wr.Report(**kwargs)
    report.blocks = report_blocks(wr, markdown, project_runs, args.wandb_entity)
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
    url = create_report(args, markdown, project_runs)
    if url:
        print(f"Created W&B report: {url}")
    else:
        print("Created W&B report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
