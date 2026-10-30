# SkillPulse – Job Success Predictor

1. `pip install -r requirements.txt`
2. `python train.py`   (writes models/)
3. `python app.py`     (Gradio on http://localhost:7860)

## Deploy on Render
Push this folder to a GitHub repo -> Render dashboard -> New -> Blueprint (reads render.yaml)
or New -> Web Service with Build `pip install -r requirements.txt && python train.py`, Start `python app.py`.
Free plan sleeps after inactivity, so the first request can take ~1 min.
