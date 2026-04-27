import subprocess
import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent


def run_notebook(name: str) -> None:
    print(f"Executing {name}...")
    result = subprocess.run(
        ["jupyter", "nbconvert", "--execute", "--to", "notebook", "--inplace", name],
        cwd=NOTEBOOK_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error executing {name}:")
        print(result.stderr)
    else:
        print(f"{name} executed successfully.")


for notebook in [
    "est_gibbs_ces.ipynb",
    "est_gibbs_dynamic.ipynb",
    "est_gibbs_steady.ipynb",
]:
    run_notebook(notebook)

print("Exporting Gibbs TeX outputs...")
result = subprocess.run(
    [sys.executable, "export_gibbs_tex.py"],
    cwd=NOTEBOOK_DIR,
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print("Error exporting Gibbs TeX outputs:")
    print(result.stderr)
else:
    print(result.stdout)
