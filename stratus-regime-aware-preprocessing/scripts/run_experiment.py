from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
notebook = ROOT / "notebooks" / "01_stratus_eye_tracking_case_study.ipynb"
output = ROOT / "notebooks" / "executed" / "01_stratus_eye_tracking_case_study_local_run.ipynb"
output.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    "-m",
    "jupyter",
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    "--ExecutePreprocessor.timeout=-1",
    "--output",
    str(output),
    str(notebook),
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
print(f"Executed notebook written to {output}")
