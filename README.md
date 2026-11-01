# SkillPulse – Job Success Predictor

Salary-range estimator (Naukri postings) + career-fit demo (CareerMap) + hiring table, served with Gradio.

## Run locally
    pip install -r requirements.txt
    python train.py        # writes models/
    python app.py          # http://localhost:7860

## Rebuild the processed data (only if you change the raw Naukri files)
Put the big raw files in raw_data/ (not committed; too large for GitHub), then:
    python prepare_data.py raw_data/naukri_data_scientist.jsonl raw_data/naukri_software_engineer.jsonl

## Deploy on Render
Build: `pip install -r requirements.txt && python train.py`   Start: `python app.py`   (render.yaml has both)

## Data
- Naukri job postings, Dec 2024 (Hugging Face: muhammetakkurt/naukri-jobs-dataset, CC BY-NC 4.0)
- CareerMap (Mendeley Data, CC BY 4.0) - synthetic, used for demonstration only
