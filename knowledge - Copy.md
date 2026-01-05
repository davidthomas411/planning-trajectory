# MongoDB Knowledge (planeval)

This is a field-level summary of the MongoDB schema and practical behaviors observed while building the planning-trajectory pipeline. It is intended to help a new agent avoid a cold start.

## Database / Collections

- Database: `planeval`
- Collections used:
  - `evaluations` (core DVH evaluation documents)
  - `standard_protocols` (canonical protocol constraints)
  - `protocols` (template protocols with standard refs; used for alias mapping + fallback constraints)
  - `custom_structure_aliases` (canonical structure name + aliases)

## Collection: `evaluations`

Primary document represents one evaluation attempt for a plan.

Common top-level fields (used by pipeline):
- `patient.patient_id` (patient identifier; string/number)
- `plan_id` (plan identifier; string)
- `attempt_number` (iteration number; integer)
- `created_at` (datetime)
- `updated_at` (datetime; sometimes used as fallback sort)
- `standard_protocol` (string; may be empty or inconsistent)
- `protocol.protocol_name` (string; template protocol name)
- `approval.is_approved` (bool; used to qualify "final approved" plans)
- `approval.status` (string; used only for reporting)
- `summary.score` (numeric; exists but not used for model labels)

Results array:
- `results` is a list of constraint evaluations. Each `result` may contain:
  - `structure` (string; often the raw name)
  - `structure_tg263` (string or null; standardized when available)
  - `metric` (object; may be empty):
    - `display` (string; e.g., "D0.03cc[Gy]")
    - `type`, `subtype`
    - `dose_gy`, `dose_pct`, `volume_cc`, `volume_pct` (nullable)
  - `objective` (string; fallback when `metric.display` missing)
  - `goal` (object with `operator`, `value`)
  - `variation` (object with `operator`, `value`)
  - `priority` (numeric; used for weighting)
  - `achieved` (object with `value`, `unit`, `display`)

Operational notes:
- Ordering of attempts uses `attempt_number` when present, else `created_at`, else `updated_at`.
- Some `results.achieved.value` are non-numeric or missing; those are skipped.
- `structure_tg263` is not always present; fallback to `structure`.
- `standard_protocol` may be missing; fallback to `protocol.protocol_name`.

## Collection: `standard_protocols`

Canonical protocol constraints.

Fields used:
- `protocol_name` (string)
- `constraints` (list of constraints). Each constraint typically has:
  - `structure` (string)
  - `objective` (string)
  - `priority` (numeric)
  - `goal_operator`, `goal_value`
  - `variation_operator`, `variation_value`

Notes:
- No `metric` object here; `objective` is used as the display name.

## Collection: `protocols`

Template protocols that reference a standard protocol.

Fields used:
- `protocol_name` (template name)
- `constraints` (list of constraints). Each constraint typically has:
  - `structure` (string)
  - `metric.display` or `objective`
  - `priority`
  - `goal` (object with `operator`, `value`)
  - `variation` (object with `operator`, `value`)
- `standard_ref` (object):
  - `standard_id`
  - `standard_name`
  - `is_primary` (bool)

Notes:
- `standard_ref` is used to map template names to a canonical protocol and to group “related templates”.
- `protocols` is used as a fallback constraint source when `standard_protocols` is missing.

## Collection: `custom_structure_aliases`

Canonical structure name mapping.

Fields used:
- `canonical` (string)
- `aliases` (list of strings)

Mapping behavior:
- Alias map is built from `canonical` + `aliases`.
- If alias not found, a token overlap heuristic is used to match to a canonical structure in the protocol constraint set.

## Protocol Name Resolution

Protocol names appear in multiple forms:
- `standard_protocol` in `evaluations`
- `protocol.protocol_name` in `evaluations`
- `protocol_name` in `standard_protocols` and `protocols`

Resolution steps used:
1) Use `standard_protocol` if present.
2) Fallback to `protocol.protocol_name`.
3) Map template names to canonical standard names using `protocols.standard_ref`.
4) Normalize whitespace / casing for lookup.

## Constraint Matching Logic (from evaluations -> protocol constraints)

Constraints are matched by a composite key:
- canonical structure name
- metric display (from `metric.display` or `objective`)
- goal and variation
- priority
- (when sourced from `protocols`, thresholds are included in the key)

Duplicate results in an evaluation are de-duplicated by this composite key.

## Plan Qualification Criteria Used

Plans were considered "qualified" when:
- at least 2 attempts, and
- final attempt is approved (`approval.is_approved` True)

Attempt order uses `attempt_number` > `created_at` > `updated_at`.

## Coverage + Plan Score Notes

Coverage:
- Matched constraints / total protocol constraints.
- Coverage threshold used in Phase 2: 0.65 (65%).

Plan score:
- Computed from percentiles against protocol-specific CPDs.
- CPDs are built from final approved evaluations only (meeting coverage threshold).
- Direction is derived from goal operator (`<=` lower is better, `>=` higher is better).
- Priority weights: priority 1 weight 2, priority 2 weight 1.
- Used only as a reference for labels and analysis, not as a model input feature.

## Data Scale (from derived dataset)

From qualified data in Phase 2:
- 5,377 approved plans
- 9,398 evaluation attempts
- 402,922 constraint evaluations
- 150 unique structures
- 51 protocols total, 23 protocols with >=20 plans
- Only 2 protocols have >=100 qualified plans

## Known Limitations / Pitfalls

- Some results are missing numeric achieved values; those rows are ignored.
- Structure names are inconsistent; alias mapping helps but is incomplete.
- `standard_protocol` is not always populated; mapping to canonical names requires `protocols.standard_ref`.
- Many protocols have small sample sizes; strong protocol-to-protocol variability in model performance.
- The current “next iteration better” label is a proxy using chronological order, not a direct outcome label.

## Read-Only Access

All data access in the pipeline is read-only. No writes to MongoDB are performed; derived artifacts are written locally to `data/derived/`.
