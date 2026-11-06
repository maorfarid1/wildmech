"""Agreement between the LLM-assisted labels and an independent mechanical-engineer annotator.

Usage: python scripts/compute_agreement.py --sheet validation_sheet_completed.xlsx --labels labels_claude.csv --out paper/numbers_agreement.tex
The sheet has columns sid, request, your_l1, your_complexity. Leave a cell empty to skip an item.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wildmech.agreement import kappa_ci  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sheet", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sh = pd.read_excel(a.sheet) if a.sheet.endswith("xlsx") else pd.read_csv(a.sheet)
lab = pd.read_csv(a.labels)
m = sh.merge(lab, on="sid").dropna(subset=["your_l1"])
m["your_l1"] = m["your_l1"].str.strip().str.upper()
k = kappa_ci(m["l1"].to_numpy(), m["your_l1"].to_numpy(), n_boot=2000)
kn = m[~m["l1"].isin(["UNK"]) & ~m["your_l1"].isin(["UNK"])]
kk = kappa_ci(kn["l1"].to_numpy(), kn["your_l1"].to_numpy(), n_boot=2000)
mc = m.dropna(subset=["your_complexity"])
from sklearn.metrics import cohen_kappa_score  # noqa: E402
kc = cohen_kappa_score(mc["c"].astype(int), mc["your_complexity"].astype(int), weights="quadratic") if len(mc) else float("nan")
lines = [
    f"\\renewcommand{{\\NhumanN}}{{{len(m)}}}",
    f"\\renewcommand{{\\NkappaHuman}}{{{k['kappa']:.2f}}}",
    f"\\renewcommand{{\\NkappaHumanLo}}{{{k['ci_lo']:.2f}}}",
    f"\\renewcommand{{\\NkappaHumanHi}}{{{k['ci_hi']:.2f}}}",
    f"\\renewcommand{{\\NaccHuman}}{{{100 * k['accuracy']:.0f}\\%}}",
    f"\\renewcommand{{\\NkappaHumanNoUNK}}{{{kk['kappa']:.2f}}}",
    f"\\renewcommand{{\\NkappaComplexity}}{{{kc:.2f}}}",
]
Path(a.out).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
