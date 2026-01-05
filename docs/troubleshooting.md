# Troubleshooting

## ModuleNotFoundError: No module named 'src'

Run modules from the repo root:

```
python3 -m src.data.phase2_dataset
```

Or set `PYTHONPATH=.` before running.

## MongoDB connection errors

- Verify `.env` exists and contains `PLANEVAL_MONGODB_URI`.
- Confirm you are on the institutional network or VPN.
- Ask the supervisor for a read-only URI if needed.

## Empty outputs in data/derived

- Confirm Phase 2 ran successfully.
- Check the console for errors during `phase2_dataset.py`.

## Dashboard shows stale data

- Restart the dashboard or load `http://localhost:8000/?refresh=1`.

## Permission denied when running scripts/run_pipeline.sh

Run it with `bash scripts/run_pipeline.sh` or make it executable with `chmod +x scripts/run_pipeline.sh`.
