# Dashboard

The dashboard is a lightweight local web app that reads from `data/derived/`.

## Run

```
python3 src/dashboard/app.py
```

Open `http://localhost:8000`.

## Common tasks

- Refresh data after reruns: append `?refresh=1` to the URL or restart the app.
- Update abstract text: edit `draft_abstract.md` and reload the page.
- Review protocol performance: use the Abstract Figures tab.

## Data inputs

The dashboard reads the following files when available:
- `data/derived/phase2_summary.json`
- `data/derived/phase3_metrics.json`
- `data/derived/phase3_analysis.json`
- `data/derived/phase3_baselines.json`
- `data/derived/phase3_alternatives.json`
