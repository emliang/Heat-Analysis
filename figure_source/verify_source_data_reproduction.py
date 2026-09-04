#!/usr/bin/env python3
"""Run the public data-to-figure commands against submitted Source Data ZIPs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


FIGURE_JOBS = (
    (
        "Source_Data_Fig_1.zip",
        "code/plotting/plot_main_figure_01_panel_d_sources.py",
        ("--source-dir", "{data}", "--output-dir", "{out}/Fig_1d_components"),
        ("Fig_1d_components/Fig1d_network_map.pdf", "Fig_1d_components/Fig1d_selected_line.pdf"),
    ),
    (
        "Source_Data_Fig_3.zip",
        "code/plotting/plot_main_figure_03.py",
        ("--source-dir", "{data}", "--pdf", "{out}/Fig_3.pdf", "--png", "{out}/Fig_3.png"),
        ("Fig_3.pdf", "Fig_3.png"),
    ),
    (
        "Source_Data_Fig_4.zip",
        "code/plotting/plot_main_figure_04.py",
        ("--source-dir", "{data}", "--pdf", "{out}/Fig_4.pdf", "--png", "{out}/Fig_4.png"),
        ("Fig_4.pdf", "Fig_4.png"),
    ),
    (
        "Source_Data_Fig_5.zip",
        "code/plotting/plot_main_figure_05.py",
        ("--source-dir", "{data}", "--pdf", "{out}/Fig_5.pdf", "--png", "{out}/Fig_5.png"),
        ("Fig_5.pdf", "Fig_5.png"),
    ),
    (
        "Source_Data_Fig_6.zip",
        "code/plotting/plot_main_figure_06.py",
        ("--source-dir", "{data}", "--pdf", "{out}/Fig_6.pdf", "--png", "{out}/Fig_6.png"),
        ("Fig_6.pdf", "Fig_6.png"),
    ),
    (
        "Source_Data_Fig_7.zip",
        "code/plotting/plot_main_figure_07.py",
        ("--source-dir", "{data}", "--pdf", "{out}/Fig_7.pdf", "--png", "{out}/Fig_7.png"),
        ("Fig_7.pdf", "Fig_7.png"),
    ),
    (
        "Source_Data_Fig_8.zip",
        "code/plotting/plot_main_figure_08.py",
        (
            "--project-root", "{root}", "--source-dir", "{data}",
            "--pdf", "{out}/Fig_8.pdf", "--png", "{out}/Fig_8.png",
        ),
        ("Fig_8.pdf", "Fig_8.png"),
    ),
    (
        "Source_Data_Extended_Data_Fig_1.zip",
        "code/plotting/plot_extended_data_figures.py",
        (
            "--source-dir", "{data}", "--figure", "1",
            "--output-pdf", "{out}/Extended_Data_Fig_1.pdf",
            "--output-png", "{out}/Extended_Data_Fig_1.png",
        ),
        ("Extended_Data_Fig_1.pdf", "Extended_Data_Fig_1.png"),
    ),
    (
        "Source_Data_Extended_Data_Fig_2.zip",
        "code/plotting/plot_extended_data_figures.py",
        (
            "--source-dir", "{data}", "--figure", "2",
            "--output-pdf", "{out}/Extended_Data_Fig_2.pdf",
            "--output-png", "{out}/Extended_Data_Fig_2.png",
        ),
        ("Extended_Data_Fig_2.pdf", "Extended_Data_Fig_2.png"),
    ),
)


def expand(arguments: tuple[str, ...], *, root: Path, data: Path, out: Path) -> list[str]:
    values = {"root": str(root), "data": str(data), "out": str(out)}
    return [argument.format(**values) for argument in arguments]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    source_data_dir = args.source_data_dir.resolve()
    missing = [name for name, *_ in FIGURE_JOBS if not (source_data_dir / name).is_file()]
    if missing:
        parser.error("missing Source Data archives: " + ", ".join(missing))

    temporary = None
    if args.output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="heat-analysis-figure-check-")
        work = Path(temporary.name)
    else:
        work = args.output_dir.resolve()
        work.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(work / ".matplotlib"))
    environment.setdefault("SOURCE_DATE_EPOCH", "0")

    try:
        for archive_name, script_name, arguments, expected in FIGURE_JOBS:
            job = work / archive_name.removesuffix(".zip")
            if job.exists():
                shutil.rmtree(job)
            job.mkdir(parents=True)
            with zipfile.ZipFile(source_data_dir / archive_name) as archive:
                archive.extractall(job)
            output = job / "reproduced"
            output.mkdir()
            command = [
                sys.executable,
                str(root / script_name),
                *expand(arguments, root=root, data=job / "data", out=output),
            ]
            print(f"RUN  {archive_name}", flush=True)
            subprocess.run(command, check=True, cwd=root, env=environment)
            absent = [name for name in expected if not (output / name).is_file()]
            if absent:
                raise RuntimeError(
                    f"{archive_name} did not create: {', '.join(absent)}"
                )
            print(f"PASS {archive_name}", flush=True)
    finally:
        if temporary is not None:
            temporary.cleanup()

    print(f"PASS all {len(FIGURE_JOBS)} Source Data figure jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
