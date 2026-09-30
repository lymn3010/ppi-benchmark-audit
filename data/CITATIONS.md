# Citations

Cite the five-corpus comparison and the original paper of each corpus used.
Each `data/datasets/<Corpus>/README.md` repeats its citation and license.
The JSON files are rebuilt from `data/datasets/<Corpus>/source_xml/`; check with
`python -m src.tools.verify_ppi_corpus_provenance`.

## Five-corpus comparison

Pyysalo S, Airola A, Heimonen J, Bjorne J, Ginter F, Salakoski T. "Comparative
analysis of five protein-protein interaction corpora." BMC Bioinformatics
9(Suppl 3):S6, 2008. DOI: 10.1186/1471-2105-9-S3-S6.

## Corpora

| Corpus | Citation |
| --- | --- |
| AIMed | Bunescu R, Ge R, Kate RJ, Marcotte EM, Mooney RJ, Ramani AK, Wong YW. "Comparative experiments on learning information extractors for proteins and their interactions." Artificial Intelligence in Medicine 33(2):139-155, 2005. DOI: 10.1016/j.artmed.2004.07.016. |
| BioInfer | Pyysalo S, Ginter F, Heimonen J, Bjorne J, Boberg J, Jarvinen J, Salakoski T. "BioInfer: a corpus for information extraction in the biomedical domain." BMC Bioinformatics 8:50, 2007. DOI: 10.1186/1471-2105-8-50. |
| HPRD50 | Fundel K, Kuffner R, Zimmer R. "RelEx: relation extraction using dependency parse trees." Bioinformatics 23(3):365-371, 2007. DOI: 10.1093/bioinformatics/btl616. |
| IEPA | Ding J, Berleant D, Nettleton D, Wurtele E. "Mining MEDLINE: abstracts, sentences, or phrases?" Pacific Symposium on Biocomputing, 326-337, 2002. DOI: 10.1142/9789812799623_0031. |
| LLL | Nedellec C. "Learning language in logic: genic interaction extraction challenge." Proceedings of the Fourth Learning Language in Logic Workshop, 31-37, 2005. |

## Models

[Stanza](https://aclanthology.org/2020.acl-demos.14/) with
[CRAFT biomedical models](https://doi.org/10.1093/jamia/ocab090);
[QANom](https://aclanthology.org/2020.coling-main.274/); coreference:
[Word-Level](https://aclanthology.org/2021.emnlp-main.605/),
[CAW](https://aclanthology.org/2023.crac-main.2/),
[MSCAW](https://aclanthology.org/2024.crac-1.4/).
