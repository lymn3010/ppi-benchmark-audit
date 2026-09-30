# Paper reproduction code

`experiment.json` is the authoritative ordered workflow. Run it through the public CLI:

```bash
python -m src check
python -m src reproduce statistics --output DIR
```

The files are named by the artifact or analysis they produce:

| Files | Purpose |
| --- | --- |
| `runner.py`, `config.py`, `materials.py` | Load the manifest, verify inputs and run the workflow |
| `check_numbers.py` | Verify saved numerical claims and figure inputs |
| `cross_corpus.py`, `within_corpus.py`, `within_corpus_null.py` | Primary cross- and within-corpus comparisons |
| `residual_corpus.py`, `vocabulary_control.py`, `residual_bootstrap.py` | Residual corpus-effect model and controls |
| `construction_families.py`, `matched_gap_nulls.py`, `scope_sensitivity.py` | Construction-family robustness and null analyses |
| `within_corpus_physical.py`, `coverage.py` | Physical-candidate summary and evidence-tier coverage |
| `validate_human_ratings.py`, `score_human_ratings.py` | Validate the released ratings or score a returned workbook |
| `build_review_workbook.py` | Rebuild the supplementary review workbook |
| `plot_residual.py` | Rebuild the figure used in the paper |

Analysis scripts write only to the isolated output selected by `runner.py`; the fixed files
under `supplements/` are inputs and are not silently overwritten.
