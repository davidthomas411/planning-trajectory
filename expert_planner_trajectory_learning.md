# Expert Planner Trajectory Learning from PlanEval MongoDB

## Project Goal (High Level)

Learn **expert planner judgment and sequencing** from real clinical treatment planning trajectories using
PlanEval scorecards stored in MongoDB.
This project does **not** attempt to predict TPS optimization parameters or generate treatment plans.

Instead, it learns:

1. How expert planners **rank plan states** over iterative improvements
2. How close a plan is to **final approval**
3. Which **constraint families improve next** during planning

All learning is based on **DVH-derived scorecard data only**, with variable numbers of iterations per plan.

---

## Key Constraints

- No structure sets or spatial dose data
- No TPS optimization parameters
- Variable number of iterations per plan (typically 5–15)
- MongoDB is the single source of truth
- Analysis must be **plan-level** to avoid data leakage

---

## Data Sources

### MongoDB (Primary)

- **Read-only DB:** `planeval`
- **Writes:** disabled (all outputs are stored locally)

Connection string (read-only):

```text
mongodb://analyst:.hLeUaWqd2xu3ath3K.B~Uej@10.187.138.252:27017/planeval?authSource=admin
```

Collections of Interest
planeval.evaluations

Contains multiple attempts per plan

Each attempt corresponds to a planner iteration (Attempt #1 … Attempt #N)

CSV sources (PatientDataFiles, PatientDataFiles_Results) were used upstream,
but MongoDB is authoritative for this project.

Inclusion Criteria (Critical)
A plan is included only if:

It has ≥ 2 evaluation attempts

It has a final approved state

Currently determined using the CSV Approval field

May later be cross-validated with Mosaiq/MIM

All attempts belong to the same:

Patient

Plan

Protocol

Filtering must occur before feature extraction.

Step-by-Step Pipeline (Implementation Order)
Step 1 — Query and Filter Plans
Group evaluations by (patient_id, plan_id)

Keep only groups with:

count >= 2

at least one approved / final evaluation

Sort attempts by:

Attempt # if present

otherwise Created timestamp

Resulting trajectories:

text
Copy code
Plan A: S1 → S2 → S3 → … → SN
Plan B: S1 → S2 → … → S5
Variable-length trajectories are expected and correct.

Step 2 — Extract Constraint-Level Features (Per Attempt)
From each evaluation attempt, extract constraint-level information:

For each constraint:

Constraint type (Dmax, Dmean, Vx, CI, GI, etc.)

Priority (H / M / L)

Structure name

Achieved value

Limit value

Pass / Fail

Compute:

Normalized margin-to-limit

Signed consistently for ≤ and ≥ constraints

Near-limit flag (e.g., |margin| < threshold)

Step 3 — Aggregate Plan-State Features
For each attempt (plan state), compute:

Total number of constraints

Number of constraints met

Number of failed constraints

Worst normalized margin

Worst margin per structure group (PTV, Lung, Cord, Heart, Chest Wall, etc.)

Count of near-limit constraints

Overall score (optional; exclude for some tasks to avoid leakage)

This yields a fixed-length feature vector independent of protocol size.

Learning Tasks (Train / Test)
Train/Test Split
Split unit = plan, not evaluation attempt

All attempts from a plan must stay in the same split

Recommended split:

70% plans → training

10% plans → validation

20% plans → testing

Optional: temporal split using Created timestamp

Task 1 — Preference / Ranking (Core Task)
Objective: Learn expert preference ordering of plan states

Training labels:

For each plan: (S_t, S_{t+1}) → S_{t+1} preferred

Optional: wider gaps (S_t, S_{t+k})

Evaluation metrics:

Pairwise ranking accuracy

Spearman or Kendall correlation with true attempt order

Task 2 — Progress Estimation
Objective: Predict proximity to final approval

Labels:

remaining_steps = N − t

or progress = t / N

Evaluation metrics:

Mean absolute error (iterations)

Calibration curve

Task 3 — Next-Improvement Prediction
Objective: Learn planner sequencing strategy

Labels derived from (S_t → S_{t+1}):

Which constraint family improved most:

PTV coverage

Hotspots

Lung

Cord

Heart

Chest wall

Other OARs

Evaluation metrics:

Top-1 accuracy

Top-3 accuracy

Confusion matrix

Models (Initial Recommendation)
Gradient boosting models (XGBoost / LightGBM)

Ranking loss for Task 1

Regression for Task 2

Classification for Task 3

No LLM is required for the first phase.

Outputs (Local Files)
All analysis results are written locally (no MongoDB writes), including:

Train/test split metadata

Feature schema hash

Model versioning

Evaluation metrics

Per-plan predictions

This ensures reproducibility and auditability.

Deliverables
MongoDB filtering and trajectory builder

Feature extraction pipeline

Train/test-safe dataset constructor

Models for Tasks 1–3

Visualizations:

Ranking accuracy

Progress calibration

Next-improvement confusion matrix

Explicit Non-Goals
No TPS integration

No plan generation

No anatomy-aware modeling

No claim of replacing human planners

One-Sentence Project Summary
This project learns expert planner judgment and decision sequencing from real iterative planning scorecards, enabling reproducible evaluation and guidance of treatment planning workflows without access to optimizer parameters or spatial dose data.
```
