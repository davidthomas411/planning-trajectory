# Planning Trajectory Learning (Student Handoff)

This repo analyzes DVH-based plan evaluations from the institutional planning system to study how plan quality changes over iterative planning. It builds a local dataset, trains protocol-specific models, and serves a dashboard for reviewing results and draft abstract figures.

## Quickstart (student)

1) Install Python 3.10+ and create a virtual environment.
2) Install dependencies:

```
pip install -r requirements.txt
```

3) Copy the environment template and add the read-only MongoDB URI:

```
cp .env.template .env
```

4) Run the full pipeline (reads MongoDB, writes local files only):

```
./scripts/run_pipeline.sh
```

5) Start the dashboard:

```
python3 src/dashboard/app.py
```

6) Edit the abstract text in `draft_abstract.md`.

## Data safety

- The code only reads from MongoDB. All outputs are written locally under `data/derived/`.
- The `.gitignore` blocks any derived data exports from being committed.

## Docs

- `docs/student_guide.md` - step-by-step workflow for new students
- `docs/data_dictionary.md` - CSV column definitions
- `docs/dashboard.md` - how to run and use the dashboard
- `docs/troubleshooting.md` - common issues and fixes
- `docs/handoff_checklist.md` - checklist for supervising handoffs
- `docs/sample_data/` - non-PHI sample exports

## Key scripts

- `scripts/run_pipeline.sh` - run Phase 2/3 pipeline locally
- `scripts/export_csv.py` - export local CSV snapshots for review
