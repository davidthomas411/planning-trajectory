# Radiation Plan Scoring Dashboard

Interactive dashboard prototype that visualizes radiation treatment plan quality using the population-based scoring approach described in `main.pdf`:
"A Visualization and Radiation Treatment Plan Quality Scoring Method for Triage in a Population-Based Context" (Leone et al., 2024).

## Paper Summary (main.pdf)
- Dataset: 111 head and neck cancer plans (70 Gy in 35 fx) used to build population distributions for 13 DVH objectives.
- Method: For each objective, build an empirical CDF and convert each plan's DVH value to a percentile rank (0 = worst, 100 = best).
- Plan score: Compute the geometric mean of objective percentiles and map it to a population percentile.
- Visualization: "Daisy plot" (radar-like) where each petal shows objective percentile and dots indicate requested vs achieved values; overall plan score is centered.
- Validation: 6 physicians scored 60 plans on a 5-point Likert scale. Plan percentile correlated with physician rating (Spearman r = 0.53, P < .001). ROC AUC was 0.76 for separating acceptable vs unacceptable plans.
- Results: Median plan percentiles were 89th (acceptable), 62.6th (minor edits), and 35.6th (unacceptable), with reported spread values 14.5, 25.1, and 25.7.
- Conclusion: Percentile-based scoring with visual feedback can help triage plan quality and highlight improvable objectives.

## How This Repo Maps to the Paper
- Implements a daisy-plot style scorecard using D3 and Next.js.
- Displays per-objective percentile ranks and a central plan score.
- Links protocol table rows to plot sectors for quick inspection.

## Data
- Demo data lives in `data/` and `lib/sample-data.ts`.
- Replace with your own population DVH/plan data to compute percentiles and scores.

## Run Locally
```
pnpm install
pnpm dev
```
Open `http://localhost:3000`.

## Citation
Leone AO, Mohamed ASR, Fuller CD, et al. A Visualization and Radiation Treatment Plan Quality Scoring Method for Triage in a Population-Based Context. Advances in Radiation Oncology. 2024;9:101533. DOI: 10.1016/j.adro.2024.101533

## Disclaimer
This project is a research-inspired visualization prototype and is not a clinical decision support tool. Do not use for treatment decisions without appropriate validation and regulatory clearance.
