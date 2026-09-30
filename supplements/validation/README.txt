HUMAN READER-VALIDATION DATA

One independent annotator rated the fixed 163-item sample on 2026-09-12.
Join validation-ratings-2026-09-12.csv and validation-items.csv on item_id.
In the sentence field, <<...>> marks the two rated proteins.

RATING FIELDS

relation_asserted
  Q1. The sentence asserts a relation for the rated pair.

trigger_correct
  Q2. The proposed trigger is correct.

carrier_correct
  Q3. The proposed carrier attachment is correct and complete.

endpoints_correct
  Q4. The marked proteins themselves are the relation endpoints.

Answers are yes, no or uncertain. Fractions are yes / (yes + no), excluding
uncertain answers. Definite-response counts are 162, 163, 163 and 129 for Q1-Q4.
These are unweighted sample estimates, not corpus precision. Q3 includes the same
annotator's final reassessment under clarified wording. Q4 had no not-applicable
option. Participant comments are excluded.

FILES

rating-form.xlsx
  Blank English instrument with instructions, two examples and 163 study items.

validation-items.csv
  Item context and the sampling fields needed by the released checks. The fields
  carrier_class and cell_significant replace the former separate JSONL metadata file.

validation-ratings-2026-09-12.csv
  Anonymous, comment-free ratings from the independent annotator.

CHECKS

python -m src.reproduction.validate_human_ratings --check
python -m src.reproduction.check_numbers

To score a newly returned workbook:

python -m src.reproduction.score_human_ratings rating-form.xlsx --out score.json
