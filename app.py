import io, json, os, re, base64
from pathlib import Path
import joblib, numpy as np, pandas as pd, gradio as gr
from salary_lib import featurize, norm_list, novelty, in_domain

M, D = Path("models"), Path("data")
if not (M / "salary_range.joblib").exists():
    import train  # noqa: F401  (trains on first start if models are missing)
rep = json.load(open(M / "report.json"))
art = joblib.load(M / "salary_range.joblib")
career = joblib.load(M / "career_fit.joblib")
posts = pd.read_csv(D / "naukri_salary.csv")
hiring = pd.read_csv(D / "company_hiring.csv")
S, C = rep["salary"], rep["career"]
R = S["results"]["full_model"]

INDIGO, TEAL = "#4f46e5", "#0f766e"
PAL = {  # chart colours for light / dark pages (PNG background is transparent, so the card colour shows through)
    "light": dict(text="#334155", title="#0f172a", axis="#cbd5e1", bar="#c7d2fe", line="#4f46e5", band="#c7d2fe", shade="#4f46e5"),
    "dark":  dict(text="#cbd5e1", title="#f1f5f9", axis="#475569", bar="#4338ca", line="#a5b4fc", band="#4338ca", shade="#818cf8"),
}

# ------------------------------------------------------------------ salary estimate
def _predict(rows):
    X = featurize(pd.DataFrame(rows), art["vocab"])[art["columns"]]
    q = np.exp(np.column_stack([art["models"][k].predict(X) for k in (0.1, 0.5, 0.9)]))
    q[:, 0] = np.minimum(q[:, 0], q[:, 1]); q[:, 2] = np.maximum(q[:, 2], q[:, 1])
    return q   # columns: p10, median, p90 (LPA)

def _chart_b64(mode, sim, xs, curve, exp_min, exp_max, lo, mid, hi):
    from matplotlib.figure import Figure
    c = PAL[mode]
    fig = Figure(figsize=(11, 4.2))
    a1, a2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
    a1.hist(np.clip(sim, 0, 60), bins=np.arange(0, 62, 3), color=c["bar"], edgecolor="none")
    a1.axvspan(lo, hi, color=c["shade"], alpha=0.18, label="80% likely range")
    a1.axvline(mid, color=c["line"], lw=2.4, label="Typical (median)")
    a1.set_xlabel("Salary (LPA)", fontsize=10, color=c["text"]); a1.set_ylabel("Real postings", fontsize=10, color=c["text"])
    a1.set_title("Disclosed salaries of similar postings", fontsize=12, loc="left", fontweight="bold", color=c["title"])
    a2.fill_between(xs, curve[:, 0], curve[:, 2], color=c["band"], alpha=0.55, label="80% likely range")
    a2.plot(xs, curve[:, 1], color=c["line"], lw=2.4, label="Typical (median)")
    a2.axvspan(exp_min, exp_max, color=TEAL, alpha=0.2, label="Your experience")
    a2.scatter([exp_min], [mid], color="#f43f5e", zorder=5, s=55, label="This prediction")
    a2.set_xlabel("Min. experience required (years)", fontsize=10, color=c["text"]); a2.set_ylabel("Salary (LPA)", fontsize=10, color=c["text"])
    a2.set_title("How salary moves with experience", fontsize=12, loc="left", fontweight="bold", color=c["title"])
    for ax in (a1, a2):
        leg = ax.legend(fontsize=8.5, frameon=False, loc="upper left" if ax is a2 else "upper right")
        for t in leg.get_texts(): t.set_color(c["text"])
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(c["axis"])
        ax.tick_params(colors=c["text"], labelsize=9.5); ax.set_facecolor("none")
    fig.tight_layout(w_pad=3)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", transparent=True)
    return base64.b64encode(buf.getvalue()).decode()

def _card(label, value, sub="", cls=""):
    return (f'<div class="kpi"><div class="kpi-l">{label}</div><div class="kpi-v{cls}">{value}</div><div class="kpi-s">{sub}</div></div>')

def predict_salary(role, skills, location, exp_min, exp_max):
    if exp_max < exp_min:
        raise gr.Error("Max experience must be >= min experience.")
    role_l, sk, lc = (role or "").lower().strip(), norm_list(skills), norm_list(location)
    if not in_domain(role_l, art["vocab"]):
        msg = ("Enter a job title to get an estimate." if not role_l else
               f"<b>“{role.strip()}” isn't a role this model knows.</b> It was trained on Indian tech and data job postings "
               "(Naukri), and this title doesn't match any of them, so no estimate is shown. "
               "Try a title like <i>Data Analyst</i>, <i>Data Scientist</i>, <i>Data Engineer</i> or <i>Machine Learning Engineer</i>.")
        return f'<div class="warn blocked">🚫 {msg}</div>'
    base = {"role_text": role_l, "skills_text": sk, "loc_text": lc}
    lo, mid, hi = _predict([{**base, "exp_min": exp_min, "exp_max": exp_max}])[0]
    span, xs = exp_max - exp_min, np.arange(0, 26)
    curve = _predict([{**base, "exp_min": float(x), "exp_max": float(x + span)} for x in xs])

    em = (exp_min + exp_max) / 2
    pm = (posts.exp_min + posts.exp_max) / 2
    near = posts[(pm - em).abs() <= 1.5]
    if len(near) < 30: near = posts[(pm - em).abs() <= 3]
    imgs = "".join(f'<img class="ch-{m}" alt="Salary charts" src="data:image/png;base64,{_chart_b64(m, near.sal_mid.values, xs, curve, exp_min, exp_max, lo, mid, hi)}">'
                   for m in ("light", "dark"))
    cards = ('<div class="kpis">' + _card("Typical salary", f"~{mid:.0f} LPA", "median estimate") +
             _card("80% likely range", f"{lo:.0f}–{hi:.0f} LPA", "wide on purpose: advertised pay varies a lot", " t") +
             _card("Similar real postings", f"{len(near):,}", f"median {near.sal_mid.median():.0f} LPA at ~{em:.0f} yrs experience") + "</div>")
    notes, nv = [], novelty(role_l, sk, art["vocab"])
    if nv["role_ratio"] < 1.0:
        notes.append("<b>Part of this job title is unfamiliar</b> to the model, so it is mostly judged from the familiar words.")
    if nv["skills_total"] and nv["skills_known"] == 0:
        notes.append("<b>Unfamiliar skills.</b> None of these skills are common in the training data.")
    if exp_max > S["exp_max_trained"] or exp_min > S["exp_min_trained"]:
        notes.append(f"<b>Experience beyond the training data</b> (max seen: {S['exp_max_trained']:.0f} yrs); the model holds the top level flat here.")
    warn = "".join(f'<div class="warn">⚠️ {n}</div>' for n in notes)
    foot = (f'<div class="foot">Validated on unseen postings: average error ₹{R["mae_lpa"]:.1f} LPA (just guessing the median: '
            f'₹{S["results"]["median_baseline_mae_lpa"]:.1f} LPA); the 80% range contained the true salary {R["range80_coverage"]:.0%} of the time. '
            f'Based on {S["n"]:,} postings that disclosed a salary range, so pay at employers who hide salaries may differ.</div>')
    return cards + warn + f'<div class="chartcard">{imgs}</div>' + foot

# ------------------------------------------------------------------ career fit
def predict_career(*vals):
    p = career.predict_proba(pd.DataFrame([vals], columns=C["skills"]))[0]
    return {r: float(v) for r, v in zip(career.classes_, p)}

# ------------------------------------------------------------------ hiring table
def company_lookup(cat):
    d = hiring[hiring.category == cat].sort_values("postings", ascending=False).head(15)
    d = d.assign(**{"Disclosed salary (median LPA)": d.median_salary_lpa.map(lambda v: "—" if pd.isna(v) else f"{v:.0f}")})
    return d[["companyName", "postings", "openings", "Disclosed salary (median LPA)"]].rename(
        columns={"companyName": "Company", "postings": "Job postings", "openings": "Vacancies"})

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

HERO = f"""<div class="hero"><h1>SkillPulse · Job Success Predictor</h1>
<p>Estimate salary and career fit for data, analytics and software roles in India.</p>
<div class="chips"><span class="chip">{S["n"]:,} Naukri postings with salary</span>
<span class="chip">Salary error ±₹{R["mae_lpa"]:.1f} LPA</span>
<span class="chip">80% range hit rate {R["range80_coverage"]:.0%}</span>
<span class="chip">Career fit: {len(C["roles"])} roles</span></div></div>"""

with gr.Blocks(title="SkillPulse – Job Success Predictor",
               theme=gr.themes.Default(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate",
                                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]), css=CSS) as demo:
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
                gr.Examples([["Data Analyst", "sql, excel, power bi, python", "Mumbai", 1, 3],
                             ["Data Scientist", "python, machine learning, sql", "Bengaluru", 3, 6],
                             ["Senior Data Engineer", "spark, python, aws, sql", "Hyderabad", 5, 9],
                             ["Machine Learning Engineer", "deep learning, python, nlp", "Pune", 4, 8],
                             ["Software Engineer", "java, spring boot, sql", "Chennai", 2, 5]],
                            [role, skills, loc, emin, emax], label="Try an example")
            with gr.Column(scale=5):
                kp = gr.HTML()
        btn.click(predict_salary, [role, skills, loc, emin, emax], kp)
        demo.load(predict_salary, [role, skills, loc, emin, emax], kp)
    with gr.Tab("🧭 Career fit"):
        gr.Markdown("Rate yourself from 0 (none) to 6 (expert) on each skill to see which tech roles fit best. "
                    "**Illustrative only:** this uses the CareerMap dataset, which is synthetic (artificially generated), "
                    "so treat it as a demo of the technique, not career advice.")
        with gr.Row():
            with gr.Column():
                sl = [gr.Slider(0, 6, value=3, step=1, label=s) for s in C["skills"]]
                out_c = gr.Label(label="Best-fitting roles", num_top_classes=5)
                gr.Button("Find my fit", variant="primary").click(predict_career, sl, out_c)
    with gr.Tab("🏢 Who is hiring"):
        gr.Markdown("Companies with the most job postings for each role family on Naukri (Dec 2024 snapshot).")
        t = gr.Dropdown(sorted(hiring.category.unique()), value="Data Scientist", label="Role family")
        tb = gr.Dataframe(value=company_lookup("Data Scientist"), interactive=False)
        t.change(company_lookup, t, tb)
    gr.Markdown("<small>Salary data: Naukri job postings (Dec 2024, CC BY-NC 4.0); CareerMap (Mendeley Data, CC BY 4.0). "
                "Models give estimates, not guarantees.</small>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
