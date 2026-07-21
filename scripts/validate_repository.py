from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "pyproject.toml",
    "requirements.txt",
    "notebooks/01_mandatory_start.ipynb",
    "notebooks/02_zero_start.ipynb",
    "notebooks/03_paper_comparison.ipynb",
    "results/full_benchmark_ranking.csv",
    "results/run_manifest.json",
]


def check_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"Missing required repository files: {missing}")


def check_markdown_links() -> None:
    broken: list[str] = []
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for markdown_path in ROOT.rglob("*.md"):
        text = markdown_path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative_target = target.split("#", maxsplit=1)[0]
            if not relative_target:
                continue
            resolved = (markdown_path.parent / relative_target).resolve()
            if not resolved.exists():
                broken.append(f"{markdown_path.relative_to(ROOT)} -> {target}")
    if broken:
        raise RuntimeError("Broken relative Markdown links:\n" + "\n".join(broken))


def check_notebooks() -> None:
    for notebook_path in ROOT.rglob("*.ipynb"):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            raise RuntimeError(f"Unsupported notebook format: {notebook_path}")
        missing_ids = [
            index
            for index, cell in enumerate(notebook.get("cells", []))
            if not cell.get("id")
        ]
        if missing_ids:
            raise RuntimeError(
                f"Notebook cells missing IDs in {notebook_path}: {missing_ids}"
            )


def check_result_files() -> None:
    for csv_path in (ROOT / "results").glob("*.csv"):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            raise RuntimeError(f"Result file is empty: {csv_path}")

    manifest = json.loads(
        (ROOT / "results" / "run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("timesnet_latent_demand_recovery_reproduced") is not False:
        raise RuntimeError(
            "run_manifest.json must explicitly record that TimesNet recovery was not reproduced."
        )


def check_repository_metadata() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    expected_url = "https://github.com/saeedimi/dynamic-ffs-demand-forecasting"
    if expected_url not in citation:
        raise RuntimeError("CITATION.cff does not contain the current repository URL.")
    if "YOUR_USERNAME" in citation:
        raise RuntimeError("CITATION.cff still contains a placeholder username.")


if __name__ == "__main__":
    check_required_files()
    check_markdown_links()
    check_notebooks()
    check_result_files()
    check_repository_metadata()
    print("Repository validation passed.")
