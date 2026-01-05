# Project Handoff Playbook (Local -> GitHub + Pages)

Use this checklist to turn a local VSCode/Codex project into a GitHub repo with a GitHub Pages project page, ready for a student to take over.

## 1) Prep the repo (local)

- Create a clear folder structure (`src/`, `scripts/`, `docs/`, `data/derived/`).
- Add a `.gitignore` to block any local data outputs and secrets.
- Add `.env.template` with placeholders (never commit `.env`).
- Create `requirements.txt` or equivalent dependency lock file.

## 2) Make student-friendly docs

- `README.md` (plain English overview, quickstart, safety notes)
- `docs/student_guide.md` (step-by-step, minimal jargon)
- `docs/data_dictionary.md` (CSV column definitions)
- `docs/troubleshooting.md`
- `docs/handoff_checklist.md`
- `docs/sample_data/` (non-PHI examples)

## 3) Add reproducible scripts

- `scripts/run_pipeline.sh` (runs the pipeline locally)
- `scripts/export_csv.py` (local CSV export; no PHI on GitHub)
- `scripts/update_readme.py` (injects summary + abstract into README)
- `scripts/render_figures.py` (generates figures for README/PAGES)

## 4) Build a project page (GitHub Pages)

- Create `docs/index.html` + `docs/site.css`.
- Keep the page static and render figures from `docs/figures/`.
- Add a small logo in the header (avoid large wordmarks).
- Include:
  - Project summary + key metrics
  - Figures (auto-generated)
  - Draft abstract (auto-generated)
  - Refresh instructions

## 5) Initialize Git and push

```
git init
git branch -M main
git remote add origin <repo-url>
git add .
git commit -m "Initial project handoff"
git push -u origin main
```

## 6) Enable GitHub Pages

- Repo Settings -> Pages
- Source: `main` / Folder: `/docs`
- Add the project page link to the top of `README.md`.

## 7) Student runbook (what they do)

1) Create virtualenv + install dependencies
2) Copy `.env.template` -> `.env` and add read-only URI
3) Run pipeline script
4) Refresh figures + README + project page
5) Use dashboard and edit `draft_abstract.md`

## 8) Safety reminders

- Never commit PHI or raw derived data.
- Keep `.gitignore` updated to block exports.
- Use aggregated metrics only for public pages.

## Codex prompt template

Copy/paste this into a new project to automate the handoff work:

```
You are a coding agent working inside this repo.
Goal: prepare this project for student handoff with GitHub Pages.
Constraints:
- Keep all data local; do not write to production databases.
- Do not commit PHI or raw derived data.
- Provide student-friendly docs and scripts.

Tasks:
1) Add .gitignore, .env.template, requirements.
2) Add docs (student_guide, data_dictionary, troubleshooting, handoff_checklist).
3) Add scripts: run_pipeline, export_csv, update_readme, render_figures.
4) Build a static project page in docs/ (index.html + site.css).
5) Add project page link to README and update README snapshot sections.
6) Provide instructions for enabling GitHub Pages.
```
