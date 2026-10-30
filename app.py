import json, os, re
from pathlib import Path
import joblib, numpy as np, pandas as pd, gradio as gr
from salary_lib import (BANDS, BAND_LABEL, EDGES, featurize, cum_to_band_probs, norm_list, novelty)

M = Path("models")
if not (M / "salary_band.joblib").exists():
    import train  # noqa: F401  (trains on first start if models are missing)
rep = json.load(open(M / "report.json"))
art = joblib.load(M / "salary_band.joblib")
skills_model = joblib.load(M / "skills_hike.joblib")
pers_model = joblib.load(M / "personality_success.joblib")
companies = pd.read_csv(M / "company_salaries.csv")

def _quantile_lpa(p, q):
    """Salary (LPA) at quantile q, interpolating linearly inside the band that contains it."""
    c = np.cumsum(p)
    k = int(np.searchsorted(c, q))
    k = min(k, 5)
    prev = c[k - 1] if k else 0.0
    frac = (q - prev) / max(p[k], 1e-9)
    return EDGES[k] + min(max(frac, 0), 1) * (EDGES[k + 1] - EDGES[k])

def predict_salary(role, skills, location, exp_min, exp_max):
    if exp_max < exp_min:
        raise gr.Error("Max experience must be >= min experience.")
    sk, lc = norm_list(skills), norm_list(location)
    row = pd.DataFrame([{"role_text": (role or "").lower().strip(), "skills_text": sk, "loc_text": lc,
                         "exp_min": exp_min, "exp_max": exp_max}])
    X = featurize(row, art["vocab"])[art["columns"]]
    p = cum_to_band_probs(np.column_stack([m.predict_proba(X)[:, 1] for m in art["models"]]))[0]
    probs = {BAND_LABEL[b]: float(v) for b, v in zip(BANDS, p)}
    lo, mid, hi = (_quantile_lpa(p, q) for q in (0.1, 0.5, 0.9))
    notes = []
    nv = novelty((role or "").lower(), sk, art["vocab"])
    if nv["role_total"] and nv["role_known"] == 0:
        notes.append("⚠️ **Unfamiliar job title.** None of the words in it appear in the analytics/IT job postings "
                     "this model was trained on, so the estimate relies almost only on experience.")
    if nv["skills_total"] and nv["skills_known"] == 0:
        notes.append("⚠️ **Unfamiliar skills.** None of these skills appear often in the training data.")
    if exp_max > rep["salary_band"]["exp_max_trained"] or exp_min > rep["salary_band"]["exp_min_trained"]:
        notes.append("⚠️ **Experience beyond the training data** (max seen: "
                     f"{rep['salary_band']['exp_max_trained']:.0f} yrs). The model holds the top level flat here.")
    sb = rep["salary_band"]["results"]["full_model"]
    summary = (f"### Typical salary: **~{mid:.0f} LPA**  \n"
               f"80% likely range: **{lo:.0f} – {hi:.0f} LPA**\n\n"
               f"_Salary bands overlap heavily for the same profile, so treat this as a range, not a point. "
               f"On unseen postings the model picks the right band {sb['accuracy']:.0%} of the time "
               f"and a band at most one step away {sb['within_one_band']:.0%} of the time._")
    return probs, summary + ("\n\n" + "\n\n".join(notes) if notes else "")

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
        gr.Markdown("Role, skills, location and experience → expected salary band "
                    "(trained on ~15,800 Indian analytics/IT job postings).")
        role = gr.Textbox(label="Job designation", value="Senior Data Scientist")
        skills = gr.Textbox(label="Key skills (comma separated)", value="python, machine learning, sql, statistics")
        loc = gr.Textbox(label="Location(s) (comma separated)", value="Bengaluru")
        with gr.Row():
            emin = gr.Slider(0, 30, value=4, step=1, label="Min experience (yrs)")
            emax = gr.Slider(0, 40, value=8, step=1, label="Max experience (yrs)")
        out = gr.Label(label="Probability by salary band", num_top_classes=6)
        summ = gr.Markdown()
        gr.Button("Predict", variant="primary").click(predict_salary, [role, skills, loc, emin, emax], [out, summ])
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
