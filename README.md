# SkillPulse – Job Success Predictor

Four tabs, all built on real data:
1. **Salary estimate (India)** – gradient-boosting quantile models on ~5,100 Naukri postings that disclose a salary range.
2. **Success predictor (global)** – probability of out-earning peers (same country, similar experience), trained on the Stack Overflow Developer Survey 2026.
3. **Career fit** – match of your skills to 40 data/analytics/IT occupations using U.S. Dept. of Labor O*NET ratings.
4. **Who is hiring** – companies hiring for each role family on Naukri, with the experience and skills they ask for.

## Run locally
    pip install -r requirements.txt
    python train.py        # (re)writes models/ - already committed, only needed if you change data
    python app.py          # http://localhost:7860

## Rebuild the processed data (raw files are too big for GitHub; keep them in raw_data/)
    python prepare_data.py raw_data/naukri_data_scientist.jsonl raw_data/naukri_software_engineer.jsonl
    python prepare_so.py   raw_data/so_results.csv
    python prepare_onet.py                      # needs raw_data/onet_*.csv
    python train.py

## Deploy on Render
Build `pip install -r requirements.txt`, start `python app.py` (see render.yaml). Models are committed, so there is no training at build time.
Free instances sleep when idle; the first visit after a pause takes ~1 minute. For a live demo, open the site a few minutes beforehand or ping it every 5-10 minutes (e.g. UptimeRobot).

## Data & licences
- Naukri job postings, Dec 2024 (Hugging Face `muhammetakkurt/naukri-jobs-dataset`, CC BY-NC 4.0)
- Stack Overflow Developer Survey 2026 (ODbL 1.0)
- O*NET 31.x database, U.S. Department of Labor, Employment and Training Administration (CC BY 4.0)
