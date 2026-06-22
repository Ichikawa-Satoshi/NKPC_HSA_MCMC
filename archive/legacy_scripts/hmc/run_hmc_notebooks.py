import subprocess
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent

# Execute est_hmc_ces.ipynb
print("Executing est_hmc_ces.ipynb...")
result1 = subprocess.run(['jupyter', 'nbconvert', '--execute', '--to', 'notebook', '--inplace', 'est_hmc_ces.ipynb'], cwd=NOTEBOOK_DIR, capture_output=True, text=True)
if result1.returncode != 0:
    print("Error executing est_hmc_ces.ipynb:")
    print(result1.stderr)
else:
    print("est_hmc_ces.ipynb executed successfully.")

# Execute est_hmc_dynamic.ipynb
print("Executing est_hmc_dynamic.ipynb...")
result2 = subprocess.run(['jupyter', 'nbconvert', '--execute', '--to', 'notebook', '--inplace', 'est_hmc_dynamic.ipynb'], cwd=NOTEBOOK_DIR, capture_output=True, text=True)
if result2.returncode != 0:
    print("Error executing est_hmc_dynamic.ipynb:")
    print(result2.stderr)
else:
    print("est_hmc_dynamic.ipynb executed successfully.")

# Execute est_hmc_steady.ipynb
print("Executing est_hmc_steady.ipynb...")
result3 = subprocess.run(['jupyter', 'nbconvert', '--execute', '--to', 'notebook', '--inplace', 'est_hmc_steady.ipynb'], cwd=NOTEBOOK_DIR, capture_output=True, text=True)
if result3.returncode != 0:
    print("Error executing est_hmc_steady.ipynb:")
    print(result3.stderr)
else:
    print("est_hmc_steady.ipynb executed successfully.")

# Execute est_hmc_full.ipynb
print("Executing est_hmc_full.ipynb...")
result4 = subprocess.run(['jupyter', 'nbconvert', '--execute', '--to', 'notebook', '--inplace', 'est_hmc_full.ipynb'], cwd=NOTEBOOK_DIR, capture_output=True, text=True)
if result4.returncode != 0:
    print("Error executing est_hmc_full.ipynb:")
    print(result4.stderr)
else:
    print("est_hmc_full.ipynb executed successfully.")

# Execute exp_hmc_tex.ipynb
print("Executing exp_hmc_tex.ipynb...")
result5 = subprocess.run(['jupyter', 'nbconvert', '--execute', '--to', 'notebook', '--inplace', 'exp_hmc_tex.ipynb'], cwd=NOTEBOOK_DIR, capture_output=True, text=True)
if result5.returncode != 0:
    print("Error executing exp_hmc_tex.ipynb:")
    print(result5.stderr)
else:
    print("exp_hmc_tex.ipynb executed successfully.")
