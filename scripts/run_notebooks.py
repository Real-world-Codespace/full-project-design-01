"""Execute project notebooks in their canonical order.

The first notebook generates local synthetic raw data when S3 ingestion is not
enabled and no raw file is present. Executed copies are written to reports so
source notebooks remain clean and reviewable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError("No pipeline notebooks were found.")
    output = root / "reports" / "executed_notebooks"
    output.mkdir(parents=True, exist_ok=True)
    kernel_root = root / ".runtime" / "jupyter"
    kernel_dir = kernel_root / "kernels" / "cooling-load"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "Cooling Load Project",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root), environment.get("PYTHONPATH", "")]
    )
    environment.setdefault("MPLBACKEND", "Agg")
    environment["JUPYTER_PATH"] = os.pathsep.join(
        [str(kernel_root), environment.get("JUPYTER_PATH", "")]
    )
    for notebook in notebooks:
        print(f"Executing {notebook.name}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                str(notebook),
                "--output-dir",
                str(output),
                "--ExecutePreprocessor.timeout=300",
                "--ExecutePreprocessor.kernel_name=cooling-load",
            ],
            cwd=root,
            env=environment,
            check=True,
        )
    print(f"Executed {len(notebooks)} notebooks. Outputs: {output}")


if __name__ == "__main__":
    main()
