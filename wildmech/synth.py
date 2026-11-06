"""Synthetic placeholder export in the exact schema Leo will deliver.
Used only to test the pipeline end to end. Nothing produced from it is a finding."""
from __future__ import annotations

import numpy as np
import pandas as pd

TEMPLATES = {
    "MAT": ["Which coating should I use on a 6061 aluminum part that slides against steel?",
            "What is the difference between 17-4PH H900 and H1150 for a shaft?",
            "How do I avoid galvanic corrosion between a copper pipe and a stainless fitting?",
            "Best material for a snap hook that sees -40C to 85C?"],
    "CAL": ["Calculate the pressure drop in 20 m of 1 inch pipe at 40 L/min water.",
            "How much heat can a 100 x 60 mm aluminum heat sink with 12 fins dissipate at natural convection?",
            "What is the first natural frequency of a 300 mm steel cantilever, 20 x 5 mm section?",
            "Buckling load for a hollow 25 mm tube, 2 mm wall, 1.2 m long, pinned ends?"],
    "JNT": ["What torque for an M8 class 10.9 bolt into a tapped aluminum hole?",
            "O-ring groove dimensions for a static face seal using AS568-214?",
            "How many M6 bolts do I need to hold 5 bar on a 150 mm cover?",
            "Fillet weld size for a 2 kN cantilever bracket?"],
    "GDT": ["What fit should I use to press a 6205 bearing into an aluminum housing?",
            "Tolerance stack-up for three parts in a locking mechanism, how do I do RSS?",
            "How do I call out position for a bolt pattern per ASME Y14.5?"],
    "CMP": ["Size a motor for a lead screw axis moving 30 kg at 0.2 m/s.",
            "Which linear guide for a vibrating pick-and-place head?",
            "Compression spring for 50 N preload in a 20 mm bore?"],
    "STD": ["Which ISO standard covers general tolerances for machined parts?",
            "What does MIL-STD-810 require for vibration testing of a ground vehicle box?",
            "Is there a standard for ultrasonic cleaning of stainless parts?"],
    "DFM": ["Minimum bend radius for 3 mm 5052 sheet?",
            "Draft angle for a textured ABS injection molded housing?",
            "Can I machine a 4 m screw on a 1 m lathe?"],
    "CAD": ["How do I make a sweep along a helix in SolidWorks without self intersection?",
            "Why does my fillet fail in Creo on this edge loop?",
            "How do I write a macro to export all drawings to PDF?"],
    "SRCH": ["Do we already have a bracket like this in our vault?",
            "Find our last test report for the pump housing.",
            "Which of our existing parts use a 25 mm bearing?"],
    "FAIL": ["Why did my aluminum bracket crack at the bolt hole after 10k cycles?",
            "We see 12 C difference between test and simulation, what could cause it?",
            "Parts do not align even though tolerances look fine, why?"],
    "TST": ["Which mechanical tests should I run to qualify a plastic latch?",
            "How do I run a fine leak test per MIL-STD-883?"],
    "DOC": ["Write a design review summary for the new gearbox cover.",
            "Check this BOM for missing fasteners."],
    "SIM": ["What boundary conditions should I use for a bolted flange in FEA?",
            "My CFD does not converge in a narrow gap, what mesh settings help?"],
    "MEC": ["Suggest mechanisms to fold a drone arm with one motion.",
            "Concept for holding a load with rubber pads and a hydraulic piston."],
    "SUP": ["Find an alternative to an obsolete 24 V solenoid valve.",
            "Rough cost of anodizing 200 small parts?"],
    "OOS": ["Write a Python script that renames files.", "Pick a lens for a machine vision camera."],
}
WEIGHTS = {"MAT": .14, "CAL": .14, "JNT": .09, "GDT": .07, "CMP": .1, "STD": .07, "DFM": .07, "CAD": .08,
           "SRCH": .06, "FAIL": .04, "TST": .03, "DOC": .03, "SIM": .02, "MEC": .02, "SUP": .02, "OOS": .02}
FAIL_P = {"CAL": .3, "FAIL": .35, "GDT": .28, "STD": .25, "SRCH": .2, "MAT": .18}
INDUSTRIES = ["industrial_machinery", "oil_gas", "automotive", "aerospace_defense", "energy", "consumer_products",
              "semiconductor_equipment", "other"]
SIZES = ["1-49", "50-249", "250-999", "1000-9999", "10000+"]
SENIORITY = ["student", "junior", "engineer", "senior", "lead", "unknown"]


def make(n_conv: int = 2000, n_users: int = 300, n_orgs: int = 40, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    orgs = pd.DataFrame({"org_id": [f"o{i:03d}" for i in range(n_orgs)],
                         "industry": rng.choice(INDUSTRIES, n_orgs), "size_band": rng.choice(SIZES, n_orgs),
                         "cad_system": rng.choice(["solidworks", "nx", "creo", "catia", "onshape"], n_orgs),
                         "plm_system": rng.choice(["none", "teamcenter", "windchill", "pdm"], n_orgs)})
    users = pd.DataFrame({"user_id": [f"u{i:04d}" for i in range(n_users)],
                          "org_id": rng.choice(orgs["org_id"], n_users),
                          "seniority": rng.choice(SENIORITY, n_users), "role": rng.choice(["design", "analysis", "manufacturing", "unknown"], n_users),
                          "signup_month": rng.choice(["2025-06", "2025-09", "2026-01", "2026-04"], n_users),
                          "plan": rng.choice(["enterprise", "self_serve"], n_users, p=[.6, .4]),
                          "country": rng.choice(["US", "DE", "GB", "IL", "SE"], n_users)})
    cats, w = list(WEIGHTS), np.array(list(WEIGHTS.values()))
    w = w / w.sum()
    convs, msgs = [], []
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(n_conv):
        cid = f"c{i:05d}"
        u = users.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        cat = rng.choice(cats, p=w)
        q = rng.choice(TEMPLATES[cat])
        start = t0 + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 300)))
        convs.append({"conversation_id": cid, "user_id": u.user_id, "org_id": u.org_id, "started_at": start,
                      "surface": rng.choice(["web_chat", "cad_addin"]), "model_version": rng.choice(["v3", "v4"]),
                      "_true_l1": cat})
        failed = rng.random() < FAIL_P.get(cat, .12)
        turn, t = 0, start
        msgs.append(_m(cid, turn, "user", q, t)); turn += 1
        t += pd.Timedelta(seconds=int(rng.integers(5, 40)))
        msgs.append(_m(cid, turn, "assistant", f"[synthetic answer about {cat}]", t,
                       feedback="down" if failed and rng.random() < .3 else ("up" if not failed and rng.random() < .15 else "none")))
        turn += 1
        if failed:
            kind = rng.choice(["rephrase", "correct", "regen"])
            t += pd.Timedelta(minutes=int(rng.integers(1, 6)))
            if kind == "rephrase":
                msgs.append(_m(cid, turn, "user", q.replace("?", "") + " - I mean exactly this, please", t))
            elif kind == "correct":
                msgs.append(_m(cid, turn, "user", "That's wrong, check again the units.", t))
            else:
                msgs[-1]["regenerated"] = True
                msgs.append(_m(cid, turn, "user", "ok thanks", t))
            turn += 1
            msgs.append(_m(cid, turn, "assistant", "[synthetic retry]", t + pd.Timedelta(seconds=20)))
        elif rng.random() < .35:
            t += pd.Timedelta(minutes=int(rng.integers(1, 20)))
            msgs.append(_m(cid, turn, "user", "Thanks. And what about a stainless version?", t)); turn += 1
            msgs.append(_m(cid, turn, "assistant", "[synthetic follow-up]", t + pd.Timedelta(seconds=20)))
    m = pd.DataFrame(msgs)
    m["message_id"] = [f"m{i:06d}" for i in range(len(m))]
    return {"conversations": pd.DataFrame(convs), "messages": m, "users": users, "orgs": orgs}


def _m(cid, turn, role, text, t, feedback="none"):
    return {"conversation_id": cid, "turn_index": turn, "role": role, "text": text, "created_at": t,
            "has_attachment": False, "attachment_types": "none", "feedback": feedback, "regenerated": False,
            "copied": False}
