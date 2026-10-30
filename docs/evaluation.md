# Evaluation and known limitations

[Back to the project overview](../README.md)

## Spatial baseline

The **count-based spatial baseline**, not XGBoost, captured **83.8% (477/569)** of mapped Oct–Dec 2024 accidents in grids with at least five Jan–Sep accidents. It flagged 112 of the 260 grids observed during Jan–Sep (43.1%), giving **1.95× lift** relative to uniformly selecting the same share of those grids. Records without valid dates or coordinates inside the configured Bangkok bounds are excluded.

Reproduce this result from the repository root without retraining or writing files:

```bash
python -c 'from src.evidence_metrics import holdout_validation; print(holdout_validation())'
```

## Historical concentration

The separate **17.5×** figure comes from ranking grids by accident count over the **full 2024 dataset**: the top approximately 5% contain 87.4% of mapped accidents. This describes historical concentration, not held-out prediction performance, and should not be quoted as XGBoost performance.

## XGBoost

The checkout does not include `models/xgb_results.json`, the test split (`X_test.csv`, `y_test.csv`), or `split_info.json`. XGBoost accuracy, recall, and improvement over the spatial baseline are therefore not verified here. Its training script also uses the test split for early stopping; an independent final test set is needed for an unbiased performance claim. Grid-classification recall and the fraction of accidents captured measure different things and cannot establish an improvement by direct subtraction.

## Thai NLP

The NLP labels are derived from `มูลเหตุสันนิษฐาน` (recorded cause), which is also included in the input text. The saved 100% test accuracy and 99.47% cross-validation accuracy describe categorization with the cause supplied; they do not establish prediction of unknown causes or performance on free-form user reports. Cross-validation also reuses a pre-fitted TF-IDF vectorizer rather than fitting it within each fold. A predictive evaluation would need cause-free inputs, checks for duplicate reports, and preprocessing fitted within each training fold.

## Grid boundaries

Cells use a 0.009° latitude/longitude step, approximately 1.00 km north–south and 0.97 km east–west near central Bangkok. Partial cells exist at the northern and eastern bounds.

Runtime coordinate lookup includes the partial northern and eastern cells. A regression test checks all 2,418 saved grid centres against their recorded IDs, with additional checks for the outer bounds and points just outside them. This fixes the earlier mismatch affecting 100 cell centres. The spatial holdout calculation uses its own assignment with partial cells included; the lookup fix does not establish predictive model accuracy.

## Scope of the results

The data is a historical 2024 snapshot. Counts are not normalized by traffic volume or distance driven, so they describe recorded accident concentration rather than a driver's probability of an accident. Route scores are estimates for exploration, not validated safety guarantees.
