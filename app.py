import html, json, os
from pathlib import Path
import joblib, numpy as np, pandas as pd, gradio as gr
from salary_lib import featurize, norm_list, novelty, in_domain

M, D = Path("models"), Path("data")
if not (M / "salary_range.joblib").exists():
    import train  # noqa: F401  (trains on first start if models are missing)
rep = json.load(open(M / "report.json"))
art = joblib.load(M / "salary_range.joblib")
so_art = joblib.load(M / "so_pay.joblib")
posts = pd.read_csv(D / "naukri_salary.csv")
hiring = pd.read_csv(D / "company_hiring.csv")
cat_stats = pd.read_csv(D / "category_stats.csv").set_index("category")
so = pd.read_csv(D / "so_success.csv.gz")
onet = pd.read_csv(D / "onet_skills.csv")
onet_info = pd.read_csv(D / "onet_info.csv").fillna("").set_index("occupation")
S, SU = rep["salary"], rep["success"]
R = S["results"]["full_model"]
esc = html.escape
CRIT = "<b>Part of this job title is unfamiliar</b> to the model, so it is mostly judged from the familiar words."

# ================================================================== small SVG chart helpers (no matplotlib: instant + theme aware)
def _ticks(vmax, n=4):
    step = max(1, int(np.ceil(vmax / n / 10.0) * 10)) if vmax > 40 else max(1, int(np.ceil(vmax / n)))
    return list(range(0, int(vmax) + 1, step))

def svg_hist(vals, lo, mid, hi, w=520, h=260):
    bins = np.arange(0, 63, 3)
    counts, _ = np.histogram(np.clip(vals, 0, 60), bins=bins)
    ml, mr, mt, mb = 44, 12, 12, 40
    pw, ph = w - ml - mr, h - mt - mb
    ymax = max(int(counts.max()), 1) * 1.12
    X = lambda v: ml + v / 60 * pw
    Y = lambda c: mt + ph * (1 - c / ymax)
    o = [f'<svg viewBox="0 0 {w} {h}" class="sv" role="img" aria-label="Distribution of real salaries">']
    for t in _ticks(ymax):
        o.append(f'<line class="sv-grid" x1="{ml}" x2="{w - mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/><text class="sv-t" x="{ml - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    o.append(f'<rect class="sv-range" x="{X(lo):.1f}" y="{mt}" width="{max(X(hi) - X(lo), 1):.1f}" height="{ph}"/>')
    for i, c in enumerate(counts):
        if c: o.append(f'<rect class="sv-bar" x="{X(bins[i]) + 1:.1f}" y="{Y(c):.1f}" width="{pw / 20 - 2:.1f}" height="{ph * c / ymax:.1f}" rx="2"/>')
    o.append(f'<line class="sv-line" x1="{X(mid):.1f}" x2="{X(mid):.1f}" y1="{mt}" y2="{mt + ph}"/>')
    for t in range(0, 61, 10):
        o.append(f'<text class="sv-t" x="{X(t):.1f}" y="{h - 20}" text-anchor="middle">{t}{"+" if t == 60 else ""}</text>')
    o.append(f'<text class="sv-t" x="{ml + pw / 2:.0f}" y="{h - 3}" text-anchor="middle">Salary (LPA)</text></svg>')
    return "".join(o)

def svg_curve(xs, curve, exp_min, exp_max, mid, w=600, h=260):
    ml, mr, mt, mb = 44, 12, 12, 40
    pw, ph = w - ml - mr, h - mt - mb
    ymax = float(np.ceil(min(curve[:, 2].max(), 150) / 10) * 10)
    X = lambda v: ml + v / 25 * pw
    Y = lambda v: mt + ph * (1 - min(v, ymax) / ymax)
    o = [f'<svg viewBox="0 0 {w} {h}" class="sv" role="img" aria-label="Salary by experience">']
    step = 10 if ymax <= 60 else 20 if ymax <= 120 else 30
    for t in range(0, int(ymax) + 1, step):
        o.append(f'<line class="sv-grid" x1="{ml}" x2="{w - mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/><text class="sv-t" x="{ml - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    o.append(f'<rect class="sv-exp" x="{X(exp_min):.1f}" y="{mt}" width="{max(X(exp_max) - X(exp_min), 2):.1f}" height="{ph}"/>')
    up = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, curve[:, 2]))
    dn = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs[::-1], curve[::-1, 0]))
    o.append(f'<polygon class="sv-band" points="{up} {dn}"/>')
    o.append(f'<polyline class="sv-line" fill="none" points="{" ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, curve[:, 1]))}"/>')
    o.append(f'<circle class="sv-dot" cx="{X(exp_min):.1f}" cy="{Y(mid):.1f}" r="5"/>')
    for t in range(0, 26, 5):
        o.append(f'<text class="sv-t" x="{X(t):.1f}" y="{h - 20}" text-anchor="middle">{t}</text>')
    o.append(f'<text class="sv-t" x="{ml + pw / 2:.0f}" y="{h - 3}" text-anchor="middle">Min. experience required (years)</text></svg>')
    return "".join(o)

def _card(label, value, sub="", cls=""):
    return f'<div class="kpi"><div class="kpi-l">{label}</div><div class="kpi-v{cls}">{value}</div><div class="kpi-s">{sub}</div></div>'

def _chart_card(title, svg, legend):
    return f'<div class="chartcard"><div class="ct">{title}</div>{svg}<div class="legend">{legend}</div></div>'

# ================================================================== 1. India salary estimate (Naukri)
def _predict(rows):
    X = featurize(pd.DataFrame(rows), art["vocab"])[art["columns"]]
    q = np.exp(np.column_stack([art["models"][k].predict(X) for k in (0.1, 0.5, 0.9)]))
    q[:, 0] = np.minimum(q[:, 0], q[:, 1]); q[:, 2] = np.maximum(q[:, 2], q[:, 1])
    return q

def predict_salary(role, skills, location, exp_min, exp_max):
    if exp_max < exp_min:
        raise gr.Error("Max experience must be at least the min experience.")
    role_l, sk, lc = (role or "").lower().strip(), norm_list(skills), norm_list(location)
    if not in_domain(role_l, art["vocab"]):
        msg = ("Enter a job title to get an estimate." if not role_l else
               f"<b>“{esc(role.strip())}” isn't a role this model knows.</b> It learned from Indian tech and data job postings (Naukri), "
               "and this title doesn't match any of them, so no estimate is shown. "
               "Try <i>Data Analyst</i>, <i>Data Scientist</i>, <i>Data Engineer</i> or <i>Machine Learning Engineer</i>.")
        return f'<div class="warn blocked">🚫 {msg}</div>'
    base = {"role_text": role_l, "skills_text": sk, "loc_text": lc}
    lo, mid, hi = _predict([{**base, "exp_min": exp_min, "exp_max": exp_max}])[0]
    span, xs = exp_max - exp_min, np.arange(0, 26)
    curve = _predict([{**base, "exp_min": float(x), "exp_max": float(x + span)} for x in xs])
    em = (exp_min + exp_max) / 2
    pm = (posts.exp_min + posts.exp_max) / 2
    near = posts[(pm - em).abs() <= 1.5]
    if len(near) < 30: near = posts[(pm - em).abs() <= 3]
    cards = ('<div class="kpis">' + _card("Typical salary", f"~{mid:.0f} LPA", "median estimate") +
             _card("80% likely range", f"{lo:.0f}–{hi:.0f} LPA", "wide on purpose: advertised pay varies a lot", " t") +
             _card("Similar real postings", f"{len(near):,}", f"median {near.sal_mid.median():.0f} LPA at ~{em:.0f} yrs experience") + "</div>")
    notes, nv = [], novelty(role_l, sk, art["vocab"])
    if nv["role_ratio"] < 1.0: notes.append(CRIT)
    if nv["skills_total"] and nv["skills_known"] == 0:
        notes.append("<b>Unfamiliar skills.</b> None of these skills are common in the training data.")
    if exp_max > S["exp_max_trained"] or exp_min > S["exp_min_trained"]:
        notes.append(f"<b>Experience beyond the training data</b> (max seen: {S['exp_max_trained']:.0f} yrs); the model holds the top level flat here.")
    warn = "".join(f'<div class="warn">⚠️ {n}</div>' for n in notes)
    leg1 = '<span class="k k-bar"></span>Real postings <span class="k k-range"></span>80% range <span class="k k-line"></span>Typical'
    leg2 = '<span class="k k-band"></span>80% range <span class="k k-line"></span>Typical <span class="k k-exp"></span>Your experience <span class="k k-dot"></span>This prediction'
    charts = ('<div class="charts">' + _chart_card("Real salaries of similar postings", svg_hist(near.sal_mid.values, lo, mid, hi), leg1) +
              _chart_card("How salary moves with experience", svg_curve(xs, curve, exp_min, exp_max, mid), leg2) + "</div>")
    foot = (f'<div class="foot">Checked on unseen postings: average error ₹{R["mae_lpa"]:.1f} LPA (just guessing the median: ₹{S["results"]["median_baseline_mae_lpa"]:.1f} LPA); '
            f'the 80% range contained the true salary {R["range80_coverage"]:.0%} of the time. Based on {S["n"]:,} Naukri postings that disclosed a salary range, '
            "so pay at employers who hide it may differ.</div>")
    return cards + warn + charts + foot

# ================================================================== 2. global success predictor (Stack Overflow survey)
SO_CATS, SO_TOOLS = so_art["categories"], [t[5:] for t in so_art["tools"]]
BANDS = [(-1, 2, "0-2"), (2, 5, "3-5"), (5, 9, "6-9"), (9, 14, "10-14"), (14, 24, "15-24"), (24, 99, "25+")]
band_of = lambda e: next(b for lo, hi, b in BANDS if lo < e <= hi)
so["band"] = so.exp.map(band_of)

def _so_frame(country, exp, role, edu, org, mgr, mode, tools, extra=None):
    row = {"country": country, "education": edu, "role": role, "org_size": org, "manager": mgr, "work_mode": mode, "exp": float(exp)}
    row.update({"tool_" + t: int(t in (tools or [])) for t in SO_TOOLS})
    rows = [row] + [{**row, "tool_" + t: 1} for t in (extra or [])]
    return pd.DataFrame(rows)[so_art["cat"] + so_art["num"]]

def predict_success(country, exp, role, edu, org, mgr, mode, tools):
    tools = tools or []
    missing = [t for t in SO_TOOLS if t not in tools]
    p = so_art["model"].predict_proba(_so_frame(country, exp, role, edu, org, mgr, mode, tools, missing))[:, 1]
    p0, lifts = float(p[0]), sorted(zip(missing, p[1:] - p[0]), key=lambda x: -x[1])
    b = band_of(exp)
    peers = so[(so.country == country) & (so.band == b) & so.comp_usd.notna()]
    scope = f"in {esc(country)} with {b} years of experience"
    if len(peers) < 15:
        peers = so[(so.country == country) & so.comp_usd.notna()]; scope = f"in {esc(country)}"
    sat = so[(so.role == role) & (so.band == b) & so.satisfied.notna()]; sscope = f"{esc(role.lower())}s with {b} yrs"
    if len(sat) < 30:
        sat = so[(so.role == role) & so.satisfied.notna()]; sscope = f"{esc(role.lower())}s"
    verdict = ("likely to out-earn peers" if p0 >= 0.6 else "slightly better than a coin flip" if p0 >= 0.52 else
               "about even with peers" if p0 >= 0.45 else "less likely than peers to out-earn")
    gauge = f'<div class="gauge"><div class="gfill" style="width:{p0 * 100:.0f}%"></div><div class="gmid"></div></div>'
    cards = ('<div class="kpis">' +
             _card("Chance of out-earning peers", f"{p0:.0%}", verdict + f"<br>{gauge}") +
             _card("Typical peer pay", f"${peers.comp_usd.median():,.0f}", f"per year, median of {len(peers):,} developers {scope}", " t") +
             _card("Highly satisfied in role", f"{sat.satisfied.mean():.0%}", f"of {len(sat):,} {sscope} rate their job 8 or higher out of 10") + "</div>")
    good = [(t, d) for t, d in lifts[:3] if d >= 0.005]
    tips = ("<div class='tipcard'><div class='ct'>What the survey links to better odds</div>" +
            ("".join(f"<div class='tip'><b>{esc(t)}</b><span>+{d * 100:.1f} pts</span></div>" for t, d in good) +
             "<div class='kpi-s'>Adding each skill to your profile, other things equal. Associations in survey data, not guaranteed causes.</div>"
             if good else "<div class='kpi-s'>Your tool mix already covers the skills most linked to higher pay in this survey.</div>") + "</div>")
    r = SU["results"]["full_model"]
    foot = (f'<div class="foot"><b>Success</b> here means earning more than the typical developer in the same country with similar experience. '
            f'Checked on held-out respondents: {r["acc"]:.0%} correct (guessing: {SU["results"]["majority_baseline_acc"]:.0%}; knowing only country and experience: '
            f'{SU["results"]["country_exp_only"]["acc"]:.0%}). Based on {SU["results"]["n"]:,} survey respondents in {SU["countries"]} countries. '
            'Satisfaction is shown as a survey statistic because profile data predicts it only weakly.</div>')
    return cards + tips + foot

# ================================================================== 3. career fit (O*NET)
SK = {"Mathematics": "Using maths to solve problems", "Programming": "Writing and debugging code",
      "Operations Analysis": "Turning needs and requirements into a design", "Management of Personnel Resources": "Leading and motivating people",
      "Active Learning": "Picking up new ideas and tools quickly", "Systems Evaluation": "Judging how well a system works and what to improve",
      "Complex Problem Solving": "Solving messy, many-part problems", "Systems Analysis": "Understanding how a system works and how changes affect it",
      "Judgment and Decision Making": "Weighing costs and benefits to choose well", "Monitoring": "Tracking performance of people, projects or systems",
      "Instructing": "Teaching others how to do things", "Reading Comprehension": "Understanding written documents quickly",
      "Critical Thinking": "Questioning logic and spotting weak arguments", "Persuasion": "Convincing others to change their minds"}
SKILLS = [s for s in SK if s in set(onet.skill)]
occ_names = sorted(onet.occupation.unique())
LV = onet.pivot(index="occupation", columns="skill", values="level").loc[occ_names, SKILLS]
IM = onet.pivot(index="occupation", columns="skill", values="importance").loc[occ_names, SKILLS]

DEFAULT_RATINGS = {"Mathematics": 4, "Programming": 4, "Critical Thinking": 4, "Complex Problem Solving": 4, "Active Learning": 4, "Systems Analysis": 3}

def predict_career(*ratings):
    if sum(ratings) == 0:
        return '<div class="warn blocked">Rate at least one skill above 0 to see your closest occupations.</div>'
    u = np.array(ratings, float) * 7 / 5                       # self-rating 0-5 -> O*NET level scale 0-7
    w = IM.values / 5
    sw = np.sqrt(w)
    a, b = LV.values * sw, u[None, :] * sw
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.maximum(np.linalg.norm(b, axis=1), 1e-9))        # does the SHAPE of your profile match?
    reach = np.clip((w * np.minimum(u[None, :], LV.values)).sum(1) / (w * LV.values).sum(1), 0, 1)  # do you reach the LEVEL they need?
    fit = np.clip((cos - 0.55) / 0.45, 0, 1) * reach
    order = np.argsort(-fit)
    rows = "".join(f'<div class="fitrow"><span class="fname">{esc(occ_names[i])}</span><div class="fbar"><div class="ffill" style="width:{fit[i] * 100:.0f}%"></div></div>'
                   f'<span class="fpct">{fit[i]:.0%}</span></div>' for i in order[:6])
    best = order[0]; name = occ_names[best]
    gap = (w[best] * (LV.values[best] - u)); gi = [i for i in np.argsort(-gap)[:3] if gap[i] > 0.05]
    gaps = "".join(f"<div class='tip'><b>{esc(SKILLS[i])}</b><span>needs ~{LV.values[best][i] * 5 / 7:.1f}/5, you ~{u[i] * 5 / 7:.1f}/5</span></div>" for i in gi) or \
           "<div class='kpi-s'>You already meet the typical level in all of these skills.</div>"
    info = onet_info.loc[name]
    detail = (f"<div class='chartcard'><div class='ct'>Best match: {esc(name)}</div><div class='kpi-s'>Skills to work on first</div>{gaps}"
              f"<div class='meta'><b>Knowledge areas:</b> {esc(info.knowledge)}</div><div class='meta'><b>Work styles that matter:</b> {esc(info.work_styles)}</div>"
              f"<div class='meta'><b>In-demand tools:</b> {esc(info.tools) or 'n/a'}</div></div>")
    top = f"<div class='chartcard'><div class='ct'>Your closest occupations</div>{rows}</div>"
    foot = ("<div class='foot'>Fit combines two things: how closely the <i>shape</i> of your skills matches what each occupation emphasises, and how much of the "
            "<i>level</i> it typically needs you reach, weighted by how important each skill is. Based on U.S. Department of Labor O*NET expert ratings for 40 data, analytics and IT occupations; it shows fit, not a job offer or salary.</div>")
    return f"<div class='charts'>{top}{detail}</div>{foot}"

# ================================================================== 4. who is hiring (Naukri)
def hiring_view(cat):
    d = hiring[hiring.category == cat].sort_values("postings", ascending=False).head(15)
    s = cat_stats.loc[cat]
    cards = ('<div class="kpis">' + _card("Job postings", f"{int(s.postings):,}", f"from {int(s.companies):,} companies") +
             _card("Typical experience asked", f"{s.exp_med:.0f}+ yrs", "median minimum across postings", " t") +
             _card("Typical disclosed salary", f"{s.median_salary_lpa:.0f} LPA", f"median of {int(s.disclosed):,} postings that state pay") + "</div>")
    body = "".join(f"<tr><td><b>{esc(r.companyName)}</b></td><td class='num'>{int(r.postings)}</td><td>{esc(r.experience)}</td><td>{esc(r.city)}</td><td>{esc(r.skills)}</td></tr>"
                   for r in d.itertuples())
    table = ("<div class='tblwrap'><table class='tbl'><thead><tr><th>Company</th><th>Postings</th><th>Experience asked</th><th>Main cities</th><th>Skills most requested</th></tr></thead>"
             f"<tbody>{body}</tbody></table></div>")
    return cards + table + "<div class='foot'>Naukri.com postings, December 2024 snapshot.</div>"

# ================================================================== page
CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
.hero {background: linear-gradient(120deg,#4f46e5,#0f766e); padding:26px 30px; border-radius:16px; margin-bottom:12px}
.hero, .hero * {color:#fff !important}
.hero h1 {margin:0 0 6px 0; font-size:28px; font-weight:700} .hero p {margin:0; opacity:.95; font-size:15px}
.chips {margin-top:12px} .chip {display:inline-block; background:rgba(255,255,255,.2); padding:4px 12px; border-radius:999px; margin:0 8px 6px 0; font-size:13px}
.kpis {display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:10px}
.kpi, .chartcard, .tipcard {background:var(--block-background-fill); border:1px solid var(--border-color-primary); border-radius:14px; padding:14px 16px}
.kpi-l {font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--body-text-color-subdued)}
.kpi-v {font-size:26px; font-weight:700; margin:2px 0; color:#4f46e5} .kpi-v.t {color:#0f766e}
.kpi-s {font-size:12.5px; color:var(--body-text-color-subdued)}
.dark .kpi-v {color:#a5b4fc} .dark .kpi-v.t {color:#5eead4}
.warn {background:#fef3c7; border:1px solid #f59e0b; border-radius:10px; padding:9px 13px; margin:6px 0; font-size:13.5px}
.warn, .warn * {color:#78350f !important}
.dark .warn {background:#422006; border-color:#b45309} .dark .warn, .dark .warn * {color:#fde68a !important}
.blocked {padding:18px 20px; font-size:15px}
.charts {display:grid; grid-template-columns:1fr 1.15fr; gap:12px; margin-top:4px}
.ct {font-weight:600; margin-bottom:6px; color:var(--body-text-color)}
.sv {width:100%; height:auto; display:block} .sv-t {fill:var(--body-text-color-subdued); font-size:11px}
.sv-grid {stroke:var(--border-color-primary); stroke-width:1}
.sv-bar {fill:#c7d2fe} .sv-range {fill:#4f46e5; fill-opacity:.14} .sv-line {stroke:#4f46e5; stroke-width:2.5}
.sv-band {fill:#4f46e5; fill-opacity:.18} .sv-exp {fill:#0f766e; fill-opacity:.2} .sv-dot {fill:#f43f5e}
.dark .sv-bar {fill:#4338ca} .dark .sv-line {stroke:#a5b4fc} .dark .sv-band {fill:#818cf8; fill-opacity:.25} .dark .sv-range {fill:#818cf8; fill-opacity:.2}
.legend {font-size:12px; color:var(--body-text-color-subdued); margin-top:4px}
.k {display:inline-block; width:12px; height:8px; border-radius:2px; margin:0 4px 0 10px; vertical-align:middle}
.k-bar {background:#c7d2fe} .k-range {background:rgba(79,70,229,.3)} .k-line {background:#4f46e5; height:3px} .k-band {background:rgba(79,70,229,.3)}
.k-exp {background:rgba(15,118,110,.4)} .k-dot {background:#f43f5e; border-radius:50%; width:9px; height:9px}
.foot {font-size:12.5px; color:var(--body-text-color-subdued); margin:10px 2px}
.gauge {position:relative; height:8px; border-radius:6px; background:var(--border-color-primary); margin-top:8px; overflow:hidden}
.gfill {height:100%; background:linear-gradient(90deg,#4f46e5,#0f766e)} .gmid {position:absolute; left:50%; top:0; bottom:0; width:2px; background:var(--body-text-color)}
.kpi b, .tip b, .meta b, .tbl b, .ct, .fname, .fpct, .tip {color:var(--body-text-color)}
.tipcard {margin-top:4px} .tip {display:flex; justify-content:space-between; gap:12px; padding:7px 0; border-bottom:1px solid var(--border-color-primary); font-size:14px}
.tip span {color:var(--body-text-color-subdued)} .tip:last-of-type {border-bottom:none}
.fitrow {display:grid; grid-template-columns:minmax(120px,1.4fr) 2fr 44px; gap:10px; align-items:center; padding:6px 0; font-size:14px}
.fbar {height:10px; background:var(--border-color-primary); border-radius:6px; overflow:hidden} .ffill {height:100%; background:linear-gradient(90deg,#4f46e5,#0f766e)}
.fpct {text-align:right; font-weight:600} .meta {font-size:13.5px; margin-top:8px; color:var(--body-text-color)}
.tblwrap {overflow-x:auto; border:1px solid var(--border-color-primary); border-radius:14px; background:var(--block-background-fill)}
.tbl {width:100%; border-collapse:collapse; font-size:13.5px} .tbl th {text-align:left; padding:10px 12px; font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--body-text-color-subdued); border-bottom:1px solid var(--border-color-primary)}
.tbl td {padding:9px 12px; border-bottom:1px solid var(--border-color-primary); vertical-align:top} .tbl tr:last-child td {border-bottom:none} .tbl .num {text-align:right}
@media (max-width:760px){ .kpis, .charts {grid-template-columns:1fr} }
"""

HERO = f"""<div class="hero"><h1>SkillPulse · Job Success Predictor</h1>
<p>Salary estimates for India, success odds for developers worldwide, and career fit from U.S. Department of Labor data.</p>
<div class="chips"><span class="chip">{S["n"]:,} Indian job postings with salary</span>
<span class="chip">{SU["results"]["n"]:,} developers in {SU["countries"]} countries</span>
<span class="chip">{onet.occupation.nunique()} occupations (O*NET)</span>
<span class="chip">Salary error ±₹{R["mae_lpa"]:.1f} LPA</span></div></div>"""

D_ROLE, D_COUNTRY = ("Data scientist" if "Data scientist" in SO_CATS["role"] else SO_CATS["role"][0]), ("India" if "India" in SO_CATS["country"] else SO_CATS["country"][0])
D_EDU = "Bachelor's degree" if "Bachelor's degree" in SO_CATS["education"] else SO_CATS["education"][0]
D_ORG = "1,000 - 4,999" if "1,000 - 4,999" in SO_CATS["org_size"] else SO_CATS["org_size"][0]
D_MODE = "Hybrid" if "Hybrid" in SO_CATS["work_mode"] else SO_CATS["work_mode"][0]
DEF = {
    "salary": predict_salary("Data Scientist", "python, machine learning, sql, statistics", "Bengaluru", 3, 6),
    "success": predict_success(D_COUNTRY, 3, D_ROLE, D_EDU, D_ORG, "Individual contributor", D_MODE, ["Python", "SQL"]),
    "career": predict_career(*[DEFAULT_RATINGS.get(s, 2) for s in SKILLS]),
    "hiring": hiring_view("Data Scientist"),
}

with gr.Blocks(title="SkillPulse – Job Success Predictor",
               theme=gr.themes.Default(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate",
                                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]), css=CSS) as demo:
    gr.HTML(HERO)
    with gr.Tab("💰 Salary estimate (India)"):
        with gr.Row():
            with gr.Column(scale=2, min_width=300):
                role = gr.Textbox(label="Job designation", value="Data Scientist")
                skills = gr.Textbox(label="Key skills (comma separated)", value="python, machine learning, sql, statistics")
                loc = gr.Textbox(label="Location(s) (comma separated)", value="Bengaluru")
                emin = gr.Slider(0, 30, value=3, step=1, label="Min experience (yrs)")
                emax = gr.Slider(0, 40, value=6, step=1, label="Max experience (yrs)")
                btn = gr.Button("Predict salary", variant="primary")
            with gr.Column(scale=5):
                out_sal = gr.HTML(value=DEF["salary"])
        btn.click(predict_salary, [role, skills, loc, emin, emax], out_sal)
        gr.Examples([["Data Analyst", "sql, excel, power bi, python", "Mumbai", 1, 3],
                     ["Data Scientist", "python, machine learning, sql", "Bengaluru", 3, 6],
                     ["Senior Data Engineer", "spark, python, aws, sql", "Hyderabad", 5, 9],
                     ["Machine Learning Engineer", "deep learning, python, nlp", "Pune", 4, 8],
                     ["Software Engineer", "java, spring boot, sql", "Chennai", 2, 5]],
                    inputs=[role, skills, loc, emin, emax], outputs=out_sal, fn=predict_salary, run_on_click=True, cache_examples=False,
                    label="Try an example")
    with gr.Tab("🌍 Success predictor (global)"):
        with gr.Row():
            with gr.Column(scale=2, min_width=300):
                s_country = gr.Dropdown(SO_CATS["country"], value=D_COUNTRY, label="Country")
                s_exp = gr.Slider(0, 40, value=3, step=1, label="Years of work experience")
                s_role = gr.Dropdown(SO_CATS["role"], value=D_ROLE, label="Role")
                s_edu = gr.Dropdown(SO_CATS["education"], value=D_EDU, label="Education")
                s_org = gr.Dropdown(SO_CATS["org_size"], value=D_ORG, label="Company size (employees)")
                s_mgr = gr.Radio(SO_CATS["manager"], value="Individual contributor", label="Do you manage people?")
                s_mode = gr.Dropdown(SO_CATS["work_mode"], value=D_MODE, label="Work mode")
                s_tools = gr.CheckboxGroup(SO_TOOLS, value=["Python", "SQL"], label="Tools you use")
                s_btn = gr.Button("Predict my success odds", variant="primary")
            with gr.Column(scale=5):
                out_succ = gr.HTML(value=DEF["success"])
        s_btn.click(predict_success, [s_country, s_exp, s_role, s_edu, s_org, s_mgr, s_mode, s_tools], out_succ)
    with gr.Tab("🧭 Career fit"):
        gr.Markdown("Rate yourself from 0 (none) to 5 (expert) to see which data, analytics and IT occupations fit your skills best.")
        with gr.Row():
            with gr.Column(scale=2, min_width=300):
                sl = [gr.Slider(0, 5, value=DEFAULT_RATINGS.get(s, 2), step=1, label=s, info=SK[s]) for s in SKILLS]
                c_btn = gr.Button("Find my fit", variant="primary")
            with gr.Column(scale=5):
                out_car = gr.HTML(value=DEF["career"])
        c_btn.click(predict_career, sl, out_car)
    with gr.Tab("🏢 Who is hiring"):
        h_cat = gr.Dropdown(sorted(hiring.category.unique()), value="Data Scientist", label="Role family")
        out_hire = gr.HTML(value=DEF["hiring"])
        h_cat.change(hiring_view, h_cat, out_hire)
    gr.Markdown("<small>Data: Naukri job postings, Dec 2024 (CC BY-NC 4.0) · Stack Overflow Developer Survey 2026 (ODbL) · "
                "O*NET 31.x database by the U.S. Department of Labor, Employment and Training Administration (CC BY 4.0). "
                "Models give estimates, not guarantees.</small>")

demo.queue(max_size=32, default_concurrency_limit=4)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
