# PPI benchmark audit

Code and data for *Auditing Protein–Protein Interaction Benchmarks by Relation Pattern
and Entity Boundary* (APBC 2026).

## Setup

```bash
conda env create -f environments/PPI-stanza.yml
conda activate PPI-stanza
python -m src check
```

## Reproduce

Run all cells of `reproduce.ipynb` with the PPI-stanza kernel, or:

```bash
python -m src reproduce statistics --output DIR
python -m src reproduce full --output DIR
```

`full` needs the parser models. Seeds are in `src/reproduction/experiment.json`.

## Parser models

```bash
python -c "import stanza; stanza.download('en', package='craft', processors='tokenize,pos,lemma,depparse')"
python -c "import stanza; stanza.download('en', package=None, processors={'coref': 'default'})"
python -m src sentence "PROTEIN0 binds PROTEIN1."
```

Versions are in `environments/VERSIONS.md`.

## Files

| Location | Contents |
| --- | --- |
| `reproduce.ipynb` | Reproduction notebook |
| `src/` | Reader; paper analyses in `src/reproduction/` |
| `data/` | Corpora, lexicons and rules |
| `supplements/` | Supplement PDF, readable result tables, ledgers and anonymous ratings |
| `environments/` | Conda environment and versions |

## Citation

[Supplement](supplements/appendix/paper-supplement.pdf) ·
[Corpus citations](data/CITATIONS.md)

```bibtex
@inproceedings{kan2026auditing,
  title     = {Auditing Protein--Protein Interaction Benchmarks by Relation Pattern and Entity Boundary},
  author    = {Kan, Li-Yu and Hsu, Wen-Lian},
  booktitle = {Asia Pacific Bioinformatics Conference (APBC 2026)},
  year      = {2026}
}
```

## License

The codebase is released under the MIT License ([LICENSE](LICENSE)). The corpora in
`data/datasets/` are not covered by this license and remain subject to their original
terms. Please refer to each dataset's README for available licensing information.
