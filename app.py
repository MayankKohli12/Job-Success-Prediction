import json, os, re
from pathlib import Path
import joblib, pandas as pd, gradio as gr

M = Path("models")
if not (M / "salary_band.joblib").exists():
    import train  # noqa: F401  (trains on first start if models are missing)
rep = json.load(open(M / "report.json"))
salary_model = joblib.load(M / "salary_band.joblib")
skills_model = joblib.load(M / "skills_hike.joblib")
pers_model = joblib.load(M / "personality_success.joblib")
companies = pd.read_csv(M / "company_salaries.csv")

BAND_LABEL = {"0to3": "0–3 LPA", "3to6": "3–6 LPA", "6to10": "6–10 LPA",
              "10to15": "10–15 LPA", "15to25": "15–25 LPA", "25to50": "25–50 LPA"}
norm = lambda s: re.sub(r"\s*,\s*", ",", (s or "").lower()).strip()

def predict_salary(role, skills, location, exp_min, exp_max):
    if exp_max < exp_min:
        raise gr.Error("Max experience must be >= min experience.")
    row = pd.DataFrame([{
        "role_text": (role or "").lower(), "skills_text": norm(skills), "loc_text": norm(location),
        "exp_min": exp_min, "exp_max": exp_max, "exp_mid": (exp_min + exp_max) / 2,
        "n_locations": norm(location).count(",") + 1, "n_skills": norm(skills).count(",") + 1}])
    p = salary_model.predict_proba(row)[0]
    return {BAND_LABEL.get(c, c): float(v) for c, v in zip(salary_model.classes_, p)}

def predict_skills(*vals):
    f = rep["skills_hike"]["features"]
    p = skills_model.predict_proba(pd.DataFrame([vals], columns=f))[0][1]
    return {"High salary hike": float(p), "Low salary hike": float(1 - p)}

def predict_pers(*vals):
    f = rep["personality_success"]["features"]
    p = pers_model.predict_proba(pd.DataFrame([vals], columns=f))[0][1]
    return {"High success": float(p), "Low success": float(1 - p)}

def company_lookup(title):
    d = companies[companies.job_title == title].sort_values("num_of_jobs", ascending=False).head(10)
    return d[["company_name", "min_experience", "min_salary", "avg_salary", "max_salary", "num_of_jobs"]].rename(
        columns={"min_salary": "min (LPA)", "avg_salary": "avg (LPA)", "max_salary": "max (LPA)"})

def sliders(key, step):
    return [gr.Slider(lo, hi, value=round((lo + hi) / 2, 1), step=step, label=f.replace("_", " ").title())
            for f, (lo, hi) in rep[key]["ranges"].items()]

with gr.Blocks(title="SkillPulse – Job Success Predictor") as demo:
    gr.Markdown("# SkillPulse – Job Success Predictor\nThree models trained on Indian analytics / data-science job data.")
    with gr.Tab("Salary band for a job posting"):
        sb = rep["salary_band"]
        gr.Markdown(f"Text (role, skills, locations) + experience → salary band. "
                    f"Test accuracy {sb['results'][sb['best']]['accuracy']:.0%} (6 classes), "
                    f"within one band {sb['results'][sb['best']]['within_one_band']:.0%}.")
        role = gr.Textbox(label="Job designation", value="Senior Data Scientist")
        skills = gr.Textbox(label="Key skills (comma separated)", value="python, machine learning, sql, statistics")
        loc = gr.Textbox(label="Location(s) (comma separated)", value="Bengaluru")
        with gr.Row():
            emin = gr.Slider(0, 30, value=4, step=1, label="Min experience (yrs)")
            emax = gr.Slider(0, 40, value=8, step=1, label="Max experience (yrs)")
        out = gr.Label(label="Predicted salary band", num_top_classes=6)
        gr.Button("Predict", variant="primary").click(predict_salary, [role, skills, loc, emin, emax], out)
    with gr.Tab("Skills → salary hike"):
        gr.Markdown(f"Rate your skills 0–5. CV accuracy {rep['skills_hike']['results'][rep['skills_hike']['best']]['cv_accuracy']:.0%} on {rep['skills_hike']['n']} people.")
        sk = sliders("skills_hike", 0.1); o2 = gr.Label()
        gr.Button("Predict", variant="primary").click(predict_skills, sk, o2)
    with gr.Tab("Personality → success"):
        gr.Markdown(f"Big-Five scores. CV accuracy {rep['personality_success']['results'][rep['personality_success']['best']]['cv_accuracy']:.0%} on {rep['personality_success']['n']} people.")
        pe = sliders("personality_success", 1); o3 = gr.Label()
        gr.Button("Predict", variant="primary").click(predict_pers, pe, o3)
    with gr.Tab("Top hiring companies"):
        t = gr.Dropdown(sorted(companies.job_title.unique()), value="Data Scientist", label="Job title")
        tb = gr.Dataframe(value=company_lookup("Data Scientist"))
        t.change(company_lookup, t, tb)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
