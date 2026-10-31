import json, os, re
from pathlib import Path
import joblib, numpy as np, pandas as pd, gradio as gr
from salary_lib import (BANDS, BAND_LABEL, EDGES, featurize, cum_to_band_probs, norm_list, novelty, in_domain)

M = Path("models")
if not (M / "salary_band.joblib").exists():
    import train  # noqa: F401  (trains on first start if models are missing)
rep = json.load(open(M / "report.json"))
art = joblib.load(M / "salary_band.joblib")
skills_model = joblib.load(M / "skills_hike.joblib")
pers_model = joblib.load(M / "personality_success.joblib")
companies = pd.read_csv(M / "company_salaries.csv")

INDIGO, TEAL, GREY, RED = "#4f46e5", "#0d9488", "#94a3b8", "#dc2626"

def _quantiles_lpa(P, qs=(0.1, 0.5, 0.9)):
    """Salary (LPA) at each quantile, interpolating linearly inside the band that holds it. P: (n,6)."""
    c = np.cumsum(P, axis=1)
    out = np.zeros((len(P), len(qs)))
    for r in range(len(P)):
        for j, q in enumerate(qs):
            k = min(int(np.searchsorted(c[r], q)), 5)
            prev = c[r][k - 1] if k else 0.0
            frac = min(max((q - prev) / max(P[r][k], 1e-9), 0), 1)
            out[r, j] = EDGES[k] + frac * (EDGES[k + 1] - EDGES[k])
    return out

def _probs(rows):
    X = featurize(pd.DataFrame(rows), art["vocab"])[art["columns"]]
    return cum_to_band_probs(np.column_stack([m.predict_proba(X)[:, 1] for m in art["models"]]))

def _card(label, value, sub="", color=INDIGO):
    cls = " t" if color == TEAL else ""
    return (f'<div class="kpi"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v{cls}">{value}</div><div class="kpi-s">{sub}</div></div>')

PAL = {  # chart colours for light / dark pages (PNG background is transparent, so the card colour shows through)
    "light": dict(text="#334155", title="#0f172a", axis="#cbd5e1", bar="#c7d2fe", top="#4f46e5", line="#4f46e5", band="#c7d2fe"),
    "dark":  dict(text="#cbd5e1", title="#f1f5f9", axis="#475569", bar="#3730a3", top="#818cf8", line="#a5b4fc", band="#4338ca"),
}

def _chart_png_b64(mode, p, top, xs, curve, exp_min, exp_max, mid):
    import io, base64
    from matplotlib.figure import Figure
    c = PAL[mode]
    fig = Figure(figsize=(11, 4.2))
    a1, a2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
    labels = [BAND_LABEL[b] for b in BANDS]
    a1.barh(labels, p * 100, color=[c["top"] if i == top else c["bar"] for i in range(6)], height=0.62)
    for i, v in enumerate(p * 100):
        a1.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=10, color=c["text"])
    a1.set_xlim(0, max(60, p.max() * 100 + 12)); a1.invert_yaxis()
    a1.set_title("Probability of each salary band", fontsize=12, loc="left", fontweight="bold", color=c["title"])
    a2.fill_between(xs, curve[:, 0], curve[:, 2], color=c["band"], alpha=0.55, label="80% likely range")
    a2.plot(xs, curve[:, 1], color=c["line"], lw=2.4, label="Typical (median)")
    a2.axvspan(exp_min, exp_max, color=TEAL, alpha=0.2, label="Your experience")
    a2.scatter([exp_min], [mid], color="#f43f5e", zorder=5, s=55, label="This prediction")
    a2.set_xlabel("Min. experience required (years)", fontsize=10, color=c["text"])
    a2.set_ylabel("Salary (LPA)", fontsize=10, color=c["text"])
    a2.set_title("How salary moves with experience", fontsize=12, loc="left", fontweight="bold", color=c["title"])
    leg = a2.legend(fontsize=8.5, frameon=False, loc="upper left")
    for t in leg.get_texts(): t.set_color(c["text"])
    for ax in (a1, a2):
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(c["axis"])
        ax.tick_params(colors=c["text"], labelsize=9.5)
        ax.set_facecolor("none")
    fig.tight_layout(w_pad=3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", transparent=True)
    return base64.b64encode(buf.getvalue()).decode()

def _chart_html(p, top, xs, curve, exp_min, exp_max, mid):
    l = _chart_png_b64("light", p, top, xs, curve, exp_min, exp_max, mid)
    d = _chart_png_b64("dark", p, top, xs, curve, exp_min, exp_max, mid)
    return ('<div class="chartcard">'
            f'<img class="ch-light" alt="Salary charts" src="data:image/png;base64,{l}">'
            f'<img class="ch-dark" alt="Salary charts" src="data:image/png;base64,{d}"></div>')

def predict_salary(role, skills, location, exp_min, exp_max):
    if exp_max < exp_min:
        raise gr.Error("Max experience must be >= min experience.")
    role_l, sk, lc = (role or "").lower().strip(), norm_list(skills), norm_list(location)
    if not in_domain(role_l, art["vocab"]):
        msg = ("Enter a job title to get an estimate." if not role_l else
               f"<b>“{role.strip()}” isn't a role this model knows.</b> It was trained only on analytics, data-science "
               "and IT job postings, and this title doesn't match any of them, so no estimate is shown. "
               "Try a title like <i>Data Analyst</i>, <i>Business Analyst</i>, <i>Data Engineer</i> or <i>Machine Learning Engineer</i>.")
        return f'<div class="warn blocked">🚫 {msg}</div>'
    base = {"role_text": role_l, "skills_text": sk, "loc_text": lc}
    p = _probs([{**base, "exp_min": exp_min, "exp_max": exp_max}])[0]
    lo, mid, hi = _quantiles_lpa(p[None, :])[0]
    top = int(p.argmax())

    # how the estimate moves with experience for THIS role / skills / location
    span = exp_max - exp_min
    xs = np.arange(0, 26)
    curve = _quantiles_lpa(_probs([{**base, "exp_min": float(x), "exp_max": float(x + span)} for x in xs]))

    chart = _chart_html(p, top, xs, curve, exp_min, exp_max, mid)

    sb = rep["salary_band"]["results"]["full_model"]
    cards = ('<div class="kpis">' + _card("Typical salary", f"~{mid:.0f} LPA", "median estimate") +
             _card("80% likely range", f"{lo:.0f}–{hi:.0f} LPA", "bands overlap a lot, so use the range", TEAL) +
             _card("Most likely band", BAND_LABEL[BANDS[top]], f"{p[top]:.0%} probability", INDIGO) + "</div>")
    notes = []
    nv = novelty(role_l, sk, art["vocab"])
    if nv["role_ratio"] < 1.0:
        notes.append("<b>Part of this job title is unfamiliar</b> to the model, so it is mostly judged from the familiar words.")
    if nv["skills_total"] and nv["skills_known"] == 0:
        notes.append("<b>Unfamiliar skills.</b> None of these skills are common in the training data.")
    if exp_max > rep["salary_band"]["exp_max_trained"] or exp_min > rep["salary_band"]["exp_min_trained"]:
        notes.append(f"<b>Experience beyond the training data</b> (max seen: {rep['salary_band']['exp_max_trained']:.0f} yrs); "
                     "the model holds the top level flat here.")
    warn = "".join(f'<div class="warn">⚠️ {n}</div>' for n in notes)
    foot = (f'<div class="foot">Validated on unseen postings: right band {sb["accuracy"]:.0%} of the time, '
            f'within one band {sb["within_one_band"]:.0%} (always guessing the most common band: '
            f'{rep["salary_band"]["results"]["majority_baseline"]["accuracy"]:.0%}).</div>')
    return cards + warn + chart + foot

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


CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
.hero {background: linear-gradient(120deg,#4f46e5,#0f766e); padding:26px 30px; border-radius:16px; margin-bottom:12px}
.hero, .hero * {color:#fff !important}
.hero h1 {margin:0 0 6px 0; font-size:28px; font-weight:700} .hero p {margin:0; opacity:.95; font-size:15px}
.chips {margin-top:12px} .chip {display:inline-block; background:rgba(255,255,255,.2); padding:4px 12px; border-radius:999px; margin:0 8px 6px 0; font-size:13px}
.kpis {display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:10px}
.kpi, .chartcard {background:var(--block-background-fill); border:1px solid var(--border-color-primary); border-radius:14px; padding:14px 16px}
.kpi-l {font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--body-text-color-subdued)}
.kpi-v {font-size:26px; font-weight:700; margin:2px 0; color:#4f46e5} .kpi-v.t {color:#0f766e}
.kpi-s {font-size:12.5px; color:var(--body-text-color-subdued)}
.dark .kpi-v {color:#a5b4fc} .dark .kpi-v.t {color:#5eead4}
.warn {background:#fef3c7; border:1px solid #f59e0b; border-radius:10px; padding:9px 13px; margin:6px 0; font-size:13.5px}
.warn, .warn * {color:#78350f !important}
.dark .warn {background:#422006; border-color:#b45309} .dark .warn, .dark .warn * {color:#fde68a !important}
.blocked {padding:18px 20px; font-size:15px}
.chartcard {margin-top:4px; padding:8px} .chartcard img {width:100%; height:auto; display:block}
.ch-dark {display:none !important} .dark .ch-light {display:none !important} .dark .ch-dark {display:block !important}
.foot {font-size:12.5px; color:var(--body-text-color-subdued); margin:8px 2px}
@media (max-width:700px){ .kpis{grid-template-columns:1fr} }
"""

def sliders(key, step):
    return [gr.Slider(lo, hi, value=round((lo + hi) / 2, 1), step=step, label=f.replace("_", " ").title())
            for f, (lo, hi) in rep[key]["ranges"].items()]

R = rep["salary_band"]["results"]["full_model"]
sk_r, pe_r = rep["skills_hike"], rep["personality_success"]
HERO = f"""<div class="hero"><h1>SkillPulse · Job Success Predictor</h1>
<p>Estimate salary, salary growth and career success for analytics &amp; data-science roles in India.</p>
<div class="chips"><span class="chip">{rep["salary_band"]["n"]:,} job postings</span>
<span class="chip">Salary band: {R["accuracy"]:.0%} exact · {R["within_one_band"]:.0%} within one band</span>
<span class="chip">Skills → hike: {sk_r["results"][sk_r["best"]]["cv_accuracy"]:.0%}</span>
<span class="chip">Personality → success: {pe_r["results"][pe_r["best"]]["cv_accuracy"]:.0%}</span></div></div>"""

with gr.Blocks(title="SkillPulse – Job Success Predictor", theme=gr.themes.Default(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate", font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]), css=CSS) as demo:
    gr.HTML(HERO)
    with gr.Tab("💰 Salary estimate"):
        with gr.Row():
            with gr.Column(scale=2, min_width=300):
                role = gr.Textbox(label="Job designation", value="Senior Data Scientist")
                skills = gr.Textbox(label="Key skills (comma separated)", value="python, machine learning, sql, statistics")
                loc = gr.Textbox(label="Location(s) (comma separated)", value="Bengaluru")
                emin = gr.Slider(0, 30, value=4, step=1, label="Min experience (yrs)")
                emax = gr.Slider(0, 40, value=8, step=1, label="Max experience (yrs)")
                btn = gr.Button("Predict salary", variant="primary")
                gr.Examples([["Analytics Trainee", "excel, sql", "Pune", 0, 1],
                             ["Data Analyst", "sql, tableau, excel", "Mumbai", 2, 4],
                             ["Senior Data Scientist", "python, machine learning, sql", "Bengaluru", 5, 9],
                             ["Analytics Manager", "sql, tableau, team management", "Gurgaon", 10, 15],
                             ["Head of Data Science", "machine learning, python, strategy", "Bengaluru", 15, 20]],
                            [role, skills, loc, emin, emax], label="Try an example")
            with gr.Column(scale=5):
                kp = gr.HTML()
        btn.click(predict_salary, [role, skills, loc, emin, emax], kp)
        demo.load(predict_salary, [role, skills, loc, emin, emax], kp)
    with gr.Tab("📈 Skills → salary hike"):
        gr.Markdown("Rate yourself 0–5 on each skill area.")
        with gr.Row():
            with gr.Column():
                sk = sliders("skills_hike", 0.1)
                o2 = gr.Label(label="Likely salary hike")
                gr.Button("Predict", variant="primary").click(predict_skills, sk, o2)
    with gr.Tab("🧠 Personality → success"):
        gr.Markdown("Big-Five personality scores.")
        with gr.Row():
            with gr.Column():
                pe = sliders("personality_success", 1)
                o3 = gr.Label(label="Likely success")
                gr.Button("Predict", variant="primary").click(predict_pers, pe, o3)
    with gr.Tab("🏢 Top hiring companies"):
        t = gr.Dropdown(sorted(companies.job_title.unique()), value="Data Scientist", label="Job title")
        tb = gr.Dataframe(value=company_lookup("Data Scientist"), interactive=False)
        t.change(company_lookup, t, tb)
    gr.Markdown("<small>Models are trained on historic postings and small survey samples; treat outputs as estimates, not guarantees.</small>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
