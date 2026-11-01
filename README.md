# SkillPulse

A routed web app for exploring Indian tech hiring and estimating pay, built on real data.

| Route | What it does |
|---|---|
| `/` | Market overview: pay by role family, most requested skills, cities, employers |
| `/jobs`, `/jobs/<id>` | Filter 5,100 postings by text, role family, level, skills (all/any, exclude), company, city, experience, salary, agencies; live insights for the filtered set |
| `/companies`, `/companies/<slug>` | Rank employers by postings or pay; filter by role family, skill, city, size, agency |
| `/skills`, `/skills/<slug>` | Skill demand and pay premium, filterable by role, level, city; co-occurring skills |
| `/roles`, `/roles/<slug>` | Role families: pay range, experience curve, defining skills, hirers |
| `/tools/salary` | Quantile gradient-boosting salary range (Naukri) |
| `/tools/success` | Probability of out-earning peers (Stack Overflow survey 2026), with a grouped tool picker and a pay lookup for other skills |
| `/tools/career` | Skill-to-occupation fit (U.S. O*NET) |
| `/about` | Data, method and licences |

Light and dark themes are available from the toggle at the bottom of the sidebar. Every filter lives in the URL, so any view can be bookmarked or shared. JavaScript only adds live updating and the token pickers; every page works without it.

## Run locally
    pip install -r requirements.txt
    python app.py          # http://localhost:7860

Code layout: `app.py` (routes), `market.py` (data + filters), `tools.py` (models), `charts.py` (SVG), `templates/`, `static/`.

## Retrain / rebuild data (raw files are too big for git; keep them in raw_data/)
    python prepare_data.py raw_data/naukri_data_scientist.jsonl raw_data/naukri_software_engineer.jsonl
    python prepare_so.py   raw_data/so_results.csv
    python prepare_onet.py
    python train.py

## Deploy on Render
Build `pip install -r requirements.txt`, start `python app.py` (see render.yaml). Models are committed. Free instances sleep when idle.

## Data & licences
- Naukri job postings, Dec 2024 (Hugging Face `muhammetakkurt/naukri-jobs-dataset`, CC BY-NC 4.0)
- Stack Overflow Developer Survey 2026 (ODbL 1.0)
- O*NET 31.x database, U.S. Department of Labor, Employment and Training Administration (CC BY 4.0)
