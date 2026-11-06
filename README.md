# WildMech

Code and docs for "WildMech: What Mechanical Engineers Ask AI" (paper 2) and the data pipeline paper 3 reuses.

## Layout
- `config/taxonomy.yaml` - WildMech taxonomy v1.0: 18 task codes plus UNK, 6 design phases plus learning, facets
- `config/survey_columns.yaml` - column maps, PII list, Hebrew to English labels for the 2024/2025 surveys
- `config/config.yaml` - run config (paths via env vars, nothing private in the repo)
- `wildmech/` - scrub, facets (LLM), signals, clustering, privacy thresholds, agreement, benchmark seeds, survey, weighting, figures, synthetic data
- `scripts/run_survey.py` - survey aggregates and figures
- `docs/` - analysis plan, literature map + refs.bib, data request, legal checklist, venues
- `outputs/survey/` - real survey aggregates and figures (no row-level data)
- `outputs/synthetic_run/` - pipeline dry run on synthetic placeholders. Not findings.

## Run
```bash
pip install pandas scikit-learn matplotlib pyyaml openpyxl pytest   # + anthropic for the LLM backend
python -m pytest -q tests

# surveys (raw files stay outside the repo)
python scripts/run_survey.py --s2024 PATH --s2025 PATH --labels PATH --out outputs/survey

# real Leo logs (private inputs; WILDMECH_INTERNAL_TENANTS points to a private list of internal workspaces)
export WILDMECH_INTERNAL_TENANTS=/path/to/internal_tenants.txt
python scripts/run_queries.py --mixpanel DIR --sample label_sample.pkl --labels labels.csv --aggregates mixpanel_aggregates.json
python scripts/survey_macros.py outputs/survey
# after the human validation sheet is filled in
python scripts/compute_agreement.py --sheet validation_sheet_200.xlsx --labels labels.csv --out numbers_agreement.tex
# rebuild the asmejour version from main.tex
python scripts/make_asme.py path/to/manuscript

# query pipeline, offline dry run
python -m wildmech.pipeline --synthetic --out outputs/synthetic_run

# query pipeline, real export (inside Leo's environment)
export WILDMECH_EXPORT_DIR=... WILDMECH_DENY_TERMS=... ANTHROPIC_API_KEY=...
python -m wildmech.pipeline --export $WILDMECH_EXPORT_DIR --backend anthropic --out runs/v1
```

## Privacy rules built into the code
- Scrubber runs before any model sees text; customer names come from a private deny list.
- The LLM returns facets and a de-identified summary only. Raw and scrubbed text are never written out.
- Any reported cell needs at least 10 users and 3 orgs (queries) or 5 respondents (surveys).
- Survey PII columns are dropped on load; emails are hashed only to dedupe, then dropped.
