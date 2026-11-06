"""Build LaTeX tables for the paper from computed outputs (no numbers typed by hand)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wildmech.taxonomy import load_taxonomy  # noqa: E402

Q = Path("outputs/queries/tables")
S = Path("outputs/survey/tables")
OUT = Path(sys.argv[1])
tax = load_taxonomy()

# Paraphrased, de-identified examples written from annotated requests (no names, companies or part numbers).
EX = {
    "STD": "Which standard applies to a residential balustrade if we normally design industrial handrails to another code?",
    "MEC": "Propose a mechanism that folds solar panels open like a fan after landing, with no motors.",
    "GEN": "Create a parametric CAD model of a helical gear, 40 teeth, module 2, 20 degree pressure angle.",
    "CAL": "How do I calculate the pressure loss in a pipe?",
    "CMP": "Find a torque limiter that disengages at 30 N m and fits a 20 mm shaft.",
    "MAT": "Suggest lighter alternatives to aluminum for this part.",
    "JNT": "What preload and tightening torque per bolt keep a small circuit board from separating?",
    "SIM": "How should I model the heat-affected zone in a nonlinear crash analysis of a rollover structure?",
    "GDT": "What tolerances can I hold on two concentric cylinders of an injection-molded part?",
    "CAD": "How do I make an exploded view of my current assembly?",
    "SRCH": "How many screws are in this assembly file?",
    "DOC": "Turn this sketch into a dimensioned drawing.",
    "DFM": "Is this fluoropolymer suitable for injection molding?",
    "SUP": "Find the price of this catalog part number.",
    "TST": "Does this motor meet our internal validation criteria in this mounting configuration?",
    "FAIL": "Why do the parts not align even though the tolerances look right?",
    "EDU": "Walk me through solving the one-dimensional heat equation.",
    "OOS": "Draw a wiring diagram for a microcontroller and a stepper driver.",
}


def pct(x):
    return f"{100 * x:.1f}"


d = pd.read_csv(Q / "dist_l1.csv").set_index("l1")
order = sorted(d.index, key=lambda k: (tax.phases[tax.phase_of(k)]["order"], -d.loc[k, "share"]))
rows = []
for k in order:
    r = d.loc[k]
    ph = tax.phases[tax.phase_of(k)]["name"]
    rows.append(f"{k} & {tax.name_of(k).replace('&', '\\&')} & {ph} & {int(r.n)} & {pct(r.share)} [{pct(r.ci_lo)}, {pct(r.ci_hi)}] & {EX[k]} \\\\")
tab1 = r"""\begin{table*}[t]
\centering
\caption{The WildMech taxonomy (v1.0) and the share of each category among classifiable opening requests ($n=\Nclassifiable$). Shares are weighted by monthly message volume; brackets give 95\% stratified bootstrap intervals. Examples are paraphrased from real requests with identifying details removed.}
\label{tab:taxonomy}
\small
\begin{tabular}{@{}l p{3.3cm} p{2.6cm} r l p{6.0cm}@{}}
\toprule
Code & Category & Design phase & $n$ & Share, \% [95\% CI] & Example request (paraphrased) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
(OUT / "table_taxonomy.tex").write_text(tab1)

tt = pd.read_csv(S / "2024_task_time.csv").set_index("task")
names = {"answer_engineering_question": "Find the answer to an engineering question",
         "convert_mfg_method": "Convert a part to another manufacturing process",
         "component_search_catalog": "Find a component in supplier catalogs",
         "model_catalog_part": "Model a catalog part in CAD",
         "part_search_internal": "Find an existing part in company data"}
rows = []
for k, nm in names.items():
    r = tt.loc[k]
    rows.append(f"{nm} & {pct(r.relevance_rate)} & {r.median_times_per_project:.0f} & {r.median_minutes_each:.0f} & "
                f"{r.median_hours_per_project:.1f} [{r.median_hours_ci_lo:.1f}, {r.median_hours_ci_hi:.1f}] & {r.pain_mean:.2f} & {pct(r.pain_high_share)} \\\\")
tab2 = r"""\begin{table}[t]
\centering
\caption{Time and experience per design task, 2024 survey ($n=\svTwoFourN$). Frequency, duration and hours are medians among engineers who do the task; brackets give 95\% bootstrap intervals. Pain is the mean of a 1 to 5 item (1 = enjoy very much, 5 = suffer very much); the last column is the share answering 4 or 5.}
\label{tab:survey_time}
\small
\begin{tabular}{@{}p{4.2cm} r r r l r r@{}}
\toprule
Task & Do it, \% & Times per project & Minutes each & Hours per project & Pain & Suffer, \% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "table_survey_time.tex").write_text(tab2)
print("ok")
