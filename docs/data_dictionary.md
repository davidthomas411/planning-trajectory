# Data Dictionary (CSV exports)

The CSV exports are generated from `data/derived/*.jsonl` and written to `data/derived/exports/`.

## plan_attempts.csv

One row per plan attempt (iteration).

- `protocol_name` - protocol label from the evaluation record.
- `patient_id` - patient identifier (as stored in MongoDB).
- `plan_id` - plan identifier (as stored in MongoDB).
- `attempt_number` - attempt number from the source record.
- `attempt_index` - 1-based index within the ordered attempts for a plan.
- `attempt_count` - total number of attempts for that plan.
- `attempt_progress` - normalized progress from 0.0 to 1.0 across attempts.
- `created_at` - timestamp of the evaluation.
- `constraints_total` - total constraints for the protocol.
- `constraints_matched` - constraints matched to the plan structures.
- `coverage_pct` - matched / total constraints.
- `constraints_pass` - count of passed constraints.
- `constraints_fail` - count of failed constraints.
- `constraints_unknown` - count with no numeric achieved value.
- `near_limit_count` - count near the constraint threshold.
- `worst_margin` - worst (smallest) margin to threshold.
- `worst_normalized_margin` - worst margin normalized by threshold.
- `min_percentile` - minimum directional percentile across constraints.
- `p10_percentile` - 10th percentile across constraints.
- `p50_percentile` - median percentile across constraints.
- `p90_percentile` - 90th percentile across constraints.
- `mean_percentile` - mean percentile across constraints.
- `plan_score` - protocol-specific plan score (built from approved plans only).
- `label_stop` - stop/continue label for modeling.
- `future_best_delta` - improvement delta to best future attempt.

## constraint_evaluations.csv

One row per constraint evaluation within a plan attempt.

- `protocol_name` - protocol label from the evaluation record.
- `patient_id` - patient identifier (as stored in MongoDB).
- `plan_id` - plan identifier (as stored in MongoDB).
- `attempt_number` - attempt number from the source record.
- `structure` - structure name as stored in the plan.
- `structure_tg263` - TG-263 normalized structure name, if present.
- `metric_display` - display label for the metric.
- `metric_type` - metric type (Dose, Volume, etc).
- `metric_subtype` - metric subtype (Mean, Max, D0.03cc, etc).
- `priority` - protocol priority level (1-2 where available).
- `goal_operator` - threshold operator (<=, >=, etc).
- `goal_value` - threshold value.
- `achieved_value` - achieved value from the evaluation.
- `pass` - whether the constraint was met.
- `margin` - achieved minus threshold (signed).
- `normalized_margin` - margin normalized by threshold.
- `near_limit` - whether the constraint is close to the limit.
- `percentile` - directional percentile within the protocol distribution.

## Notes

- Columns may evolve if the Phase 2 dataset builder changes.
- If you add new fields, re-run `scripts/export_csv.py`.
