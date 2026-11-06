import numpy as np
import pandas as pd

from wildmech.agreement import kappa_ci, krippendorff_alpha_nominal
from wildmech.facets import validate_facets
from wildmech.privacy import safe_counts
from wildmech.scrub import Scrubber
from wildmech.signals import compute_signals
from wildmech.survey import multi_select, share_table, wilson
from wildmech.taxonomy import load_taxonomy


def test_scrub_removes_identifiers_keeps_engineering_terms():
    s = Scrubber(deny_terms=["Acme Robotics", "Project Falcon"])
    r = s.scrub("Hi, I'm at Acme Robotics on Project Falcon. Part 7741-0032-01 rev B, file bracket_v3.sldprt, "
                "mail joe@acme.com, call +1 617 555 0199. Need ISO 2768-mK, M8x1.25 thread, 6061-T6, 6205-2RS bearing, "
                "17-4PH, ASME Y14.5 and MIL-STD-810.")
    t = r.text
    for bad in ["Acme", "Falcon", "7741-0032-01", "bracket_v3", "joe@", "555"]:
        assert bad not in t, bad
    for good in ["ISO 2768-mK", "M8x1.25", "6061-T6", "6205-2RS", "17-4PH", "ASME Y14.5", "MIL-STD-810"]:
        assert good in t, good
    assert r.counts["ORG"] == 2


def test_validate_facets_repairs_bad_output():
    tax = load_taxonomy()
    out = validate_facets({"task_l1": "XYZ", "complexity": "C9", "needs_geometry": "True"}, tax)
    assert out["task_l1"] == "OOS" and out["complexity"] == "C2" and out["needs_geometry"] == "yes"
    assert out["phase"] == "P0"


def test_every_l1_has_phase_and_l2():
    tax = load_taxonomy()
    for c in tax.l1_codes:
        assert tax.phase_of(c) in tax.phases
        assert len(tax.l2_of(c)) >= 2


def test_safe_counts_suppresses_small_cells():
    df = pd.DataFrame({"user_id": [f"u{i}" for i in range(30)] + ["x", "y"],
                       "org_id": [f"o{i % 5}" for i in range(30)] + ["ox", "ox"],
                       "task_l1": ["CAL"] * 30 + ["STD"] * 2})
    t = safe_counts(df, "task_l1", min_users=10, min_orgs=3).set_index("task_l1")
    assert not t.loc["CAL", "suppressed"] and t.loc["STD", "suppressed"]
    assert pd.isna(t.loc["STD", "share"])


def test_kappa_and_alpha():
    a = np.array(list("AABBCCAABB"))
    assert kappa_ci(a, a, n_boot=50)["kappa"] == 1.0
    r = pd.DataFrame({"r1": list("AABB"), "r2": list("AABB"), "r3": list("AABB")})
    assert abs(krippendorff_alpha_nominal(r) - 1.0) < 1e-9
    r2 = pd.DataFrame({"r1": list("ABAB"), "r2": list("BABA")})
    assert krippendorff_alpha_nominal(r2) < 0


def test_signals_detect_rephrase_and_correction():
    t = pd.Timestamp("2026-01-01", tz="UTC")
    m = pd.DataFrame([
        ("c1", 0, "user", "What torque for an M8 bolt in aluminum?", t),
        ("c1", 1, "assistant", "x", t + pd.Timedelta(seconds=10)),
        ("c1", 2, "user", "What torque for an M8 bolt into aluminum please", t + pd.Timedelta(minutes=2)),
        ("c2", 0, "user", "Bend radius for 3 mm 5052?", t),
        ("c2", 1, "assistant", "x", t + pd.Timedelta(seconds=10)),
        ("c2", 2, "user", "That's wrong, check again", t + pd.Timedelta(minutes=1)),
        ("c3", 0, "user", "Draft angle for ABS?", t),
        ("c3", 1, "assistant", "x", t + pd.Timedelta(seconds=10)),
    ], columns=["conversation_id", "turn_index", "role", "text_scrubbed", "created_at"])
    s = compute_signals(m).set_index("conversation_id")
    assert s.loc["c1", "rephrase"] == 1 and s.loc["c2", "correction"] == 1
    assert s.loc["c3", "failure_any"] == 0 and s.loc["c3", "abandoned"] == 1


def test_survey_helpers():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    t = share_table(pd.Series(["a"] * 20 + ["b"] * 3), k_min=5).set_index("category")
    assert t.loc["b", "suppressed"] and pd.isna(t.loc["b", "share"])
    ms = multi_select(pd.Series(["חיסכון בזמן, סקרנות אישית", None, "משהו אחר"]),
                      {"חיסכון בזמן": "Save time", "סקרנות אישית": "Curiosity"})
    assert ms.loc[0, "Save time"] == 1 and ms.loc[2, "Other text"] == 1 and ms.loc[1].isna().all()
