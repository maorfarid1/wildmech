# WildMech

Taxonomy, codebook, analysis code and aggregate tables for:

> Farid, M. *What Do Mechanical Engineers Ask an AI Design Assistant? Evidence From Two Years of Real-World Use and Two Practitioner Surveys.* Manuscript, 2026.

WildMech is a taxonomy of 18 mechanical engineering task categories mapped to the phases of systematic design, with facets for complexity (C1 to C4), required context and verifiability. The paper applies it to opening requests sent to a commercial AI assistant for mechanical engineering and links the results to two practitioner surveys.

## Contents
- `config/taxonomy.yaml` - WildMech v1.0: 18 task codes plus UNK, design phases, facets
- `docs/codebook.pdf` - full codebook with level-2 subcategories and decision rules
- `config/survey_columns.yaml` - survey column maps and English labels
- `wildmech/` - scrubbing, facets, failure signals, weighting, agreement statistics, figures
- `scripts/` - scripts that produce every table, figure and number in the paper
- `outputs/queries/`, `outputs/survey/` - aggregate tables behind the paper (no row-level data)
- `tests/` - unit tests

## Run
```bash
pip install pandas scikit-learn matplotlib pyyaml openpyxl pytest
python -m pytest -q tests
python -m wildmech.pipeline --synthetic --out outputs/synthetic_run   # dry run on synthetic data
```
The real logs and individual survey responses are not public because they contain information about users and their employers. All reported cells meet minimum-count thresholds (5 respondents for surveys).

## Citation
See `CITATION.cff`. Code: MIT. Aggregate tables: CC BY 4.0.
