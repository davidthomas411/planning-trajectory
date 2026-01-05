# Student Guide

This guide is for students with minimal coding experience. Follow the steps in order.

## What you are working on (plain English)

- We collect DVH evaluation records from the planning system.
- We build a local dataset of plan attempts and constraint evaluations.
- We train simple models per protocol to predict planning decisions.
- We review results in a local dashboard and update the abstract text.

## Setup (one time)

1) Create a virtual environment and install dependencies:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Copy the environment template and fill in the read-only MongoDB URI:

```
cp .env.template .env
```

Ask the supervisor for the read-only URI (do not share it publicly).

## Run the pipeline (data -> models -> exports)

```
./scripts/run_pipeline.sh
```

This command:
- Reads from MongoDB (no writes)
- Writes local outputs to `data/derived/`
- Exports CSV snapshots to `data/derived/exports/`

## View the dashboard

```
python3 src/dashboard/app.py
```

Open `http://localhost:8000` in your browser.

## Build the project page (optional)

The project page lives in `docs/index.html` and can be published with GitHub Pages.

```
python3 scripts/render_figures.py
python3 scripts/build_site.py
```

## Export CSV for review

If you only need CSVs:

```
python3 scripts/export_csv.py
```

CSV outputs are written to `data/derived/exports/`.

## Edit the abstract

Open `draft_abstract.md` and update the text based on the results shown in the dashboard.

## If something breaks

Start with `docs/troubleshooting.md`.
