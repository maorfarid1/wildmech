"""Write LaTeX macros for every survey number used in the paper (from outputs/survey)."""
import json
import sys
from pathlib import Path

import pandas as pd

S = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/survey")
T = S / "tables"
log = json.loads((S / "survey_log.json").read_text())
M = {}
M["svTwoFourRaw"] = log["2024"]["rows_raw"]
M["svTwoFourN"] = log["2024"]["rows_clean"]
M["svTwoFourDup"] = log["2024"]["duplicate_respondents"]
M["svTwoFiveRaw"] = log["2025"]["rows_raw"]
M["svTwoFiveN"] = log["2025"]["rows_clean"]
M["svPanel"] = log["panel_overlap_2024_2025"]
M["svHardText"] = log["hardq"]["responses_with_text"]
M["svHardLabeled"] = log["hardq"]["substantive_labeled"]
g = log["generic_ai_use"]
M.update(svAIuse=g["rate"], svAIuseLo=g["ci_lo"], svAIuseHi=g["ci_hi"], svAIusers=g["k"])
sat = log["generic_ai_satisfaction"]
M.update(svSatMean=sat["mean"], svSatHigh=sat["share_4_5"], svSatLow=sat["share_1_2"])
p = log["pays_for_ai"]
M.update(svPayAnswered=p["answered"], svPayYes=p["yes"])

tt = pd.read_csv(T / "2024_task_time.csv").set_index("task")
short = {"answer_engineering_question": "EQ", "convert_mfg_method": "Conv", "component_search_catalog": "Cat",
         "model_catalog_part": "Model", "part_search_internal": "Int"}
for k, s in short.items():
    r = tt.loc[k]
    M[f"sv{s}Rel"] = r.relevance_rate
    M[f"sv{s}Times"] = r.median_times_per_project
    M[f"sv{s}MinEach"] = r.median_minutes_each
    M[f"sv{s}Hours"] = r.median_hours_per_project
    M[f"sv{s}HoursLo"] = r.median_hours_ci_lo
    M[f"sv{s}HoursHi"] = r.median_hours_ci_hi
    M[f"sv{s}Pain"] = r.pain_mean
    M[f"sv{s}Suffer"] = r.pain_high_share
    M[f"sv{s}Cens"] = r.censored_share
bt = pd.read_csv(T / "2024_binary_tasks.csv").set_index("task")
for k, s in {"design_to_org_rules": "Rules", "design_to_standards": "Std", "elegant_design_emphasis": "Eleg"}.items():
    M[f"sv{s}Applies"] = bt.loc[k, "applies_rate"]
    M[f"sv{s}Pain"] = bt.loc[k, "pain_mean_if_applies"]
    M[f"sv{s}Suffer"] = bt.loc[k, "pain_high_share_if_applies"]

hq = pd.read_csv(T / "2024_hardq_l1.csv").set_index("task_l1")
for k in hq.index:
    M[f"svHard{k}"] = hq.loc[k, "share"]
cx = pd.read_csv(T / "2024_hardq_complexity.csv").set_index("category")
for k in cx.index:
    M[f"svHard{k}"] = cx.loc[k, "share"]
vf = pd.read_csv(T / "2024_hardq_verifiable.csv").set_index("category")
M["svHardVerif"] = vf.loc["yes", "share"]
l2 = pd.read_csv(T / "2024_hardq_l2_top.csv")
M["svHardCoatN"] = int(l2[(l2.primary == "MAT") & (l2.l2 == "coating_finish")]["n"].sum())
M["svHardSealN"] = int(l2[(l2.primary == "JNT") & (l2.l2 == "seals_orings")]["n"].sum())

uses = pd.read_csv(T / "2025_ai_uses.csv").set_index("option")
for k, s in {"Technical / engineering questions": "EngQ", "Explaining technical concepts": "Explain",
             "Writing documents": "Docs", "Writing emails": "Email", "Translation": "Transl", "Calculations": "Calc",
             "Brainstorming new concepts": "Brain", "Improving existing concepts": "Improve",
             "Component and part selection": "Comp", "Quality checks": "QC", "Simulation": "Sim"}.items():
    M[f"svUse{s}"] = uses.loc[k, "share"]
bar = pd.read_csv(T / "2025_ai_barriers.csv").set_index("option")
for k, s in {"Low trust in the results": "Trust", "Does not meet my needs": "Needs",
             "Does not fit existing workflows": "Workflow", "Data security / IP concerns": "IP",
             "Company restrictions": "Company", "Too complex to use": "Complex"}.items():
    M[f"svBar{s}"] = bar.loc[k, "share"]
rs = pd.read_csv(T / "2025_ai_reasons.csv").set_index("option")
for k, s in {"Save time": "Time", "Personal curiosity": "Curious", "Replace manual work": "Manual",
             "Improve accuracy": "Accuracy", "Management directive": "Mgmt"}.items():
    M[f"svWhy{s}"] = rs.loc[k, "share"]
tl = pd.read_csv(T / "2025_ai_tools.csv").set_index("option")
for k in ["ChatGPT", "Gemini", "Copilot", "Claude", "Perplexity"]:
    M[f"svTool{k}"] = tl.loc[k, "share"]
fq = pd.read_csv(T / "2025_ai_frequency.csv").set_index("category")
M["svDailyPlus"] = fq.loc[["Several times a day", "Daily"], "share"].sum()
sen = pd.read_csv(T / "2025_seniority.csv").set_index("category")
M["svStudent"] = sen.loc["Student", "share"]
M["svJunior"] = sen.loc["Junior", "share"]
ind5 = pd.read_csv(T / "2025_industry.csv").set_index("category")
M["svDefenseTwoFive"] = ind5.loc["Defense/Military", "share"]
ind4 = pd.read_csv(T / "2024_industry.csv").set_index("category")
M["svDefenseTwoFour"] = ind4.loc["Defense/Military", "share"]
org = pd.read_csv(T / "2025_org_size.csv").set_index("category")
M["svBigOrg"] = org.loc["1000+", "share"]
cad = pd.read_csv(T / "2025_cad.csv").set_index("category")
M["svSolidworks"] = cad.loc["Solidworks", "share"]
plm = pd.read_csv(T / "2025_plm.csv").set_index("category")
M["svNoPLM"] = plm.loc["No PLM", "share"]
M["svExpMedian"] = log["2025_years_experience"]["50%"]
bys = pd.read_csv(T / "2025_ai_use_by_seniority.csv").dropna(subset=["rate"])
M["svUseMinSen"] = bys["rate"].min()
M["svUseMaxSen"] = bys["rate"].max()

lines = ["% Auto-generated by scripts/survey_macros.py. Do not edit by hand."]
pct_skip = ("svSatMean", "svExpMedian", "svTwo", "svPanel", "svHardText", "svHardLabeled", "svAIusers",
            "svPayAnswered", "svPayYes", "svHardCoatN", "svHardSealN")
digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five"}
for k, v in M.items():
    k_out = "".join(digits.get(ch, ch) for ch in k)
    if isinstance(v, float) and not k.startswith(pct_skip) and not k.endswith(("Times", "MinEach", "Hours", "HoursLo", "HoursHi", "Pain")) and 0 <= v <= 1:
        t = f"{100 * v:.0f}\\%"
    elif isinstance(v, float):
        t = f"{v:.2f}" if k.endswith("Pain") or k == "svSatMean" else (f"{v:.0f}" if v >= 10 or float(v).is_integer() else f"{v:.1f}")
    else:
        t = f"{int(v):,}"
    lines.append(f"\\newcommand{{\\{k_out}}}{{{t}}}")
(S / "numbers_survey.tex").write_text("\n".join(lines) + "\n")
print(len(M), "macros")
