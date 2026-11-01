"""Predictive tools (salary, success odds, career fit). Each returns plain dicts that the templates render."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from salary_lib import featurize, norm_list, novelty, in_domain
from market import posts, fmt_skill, hist, SKILL_COUNT, GENERIC

M, D = Path("models"), Path("data")
if not (M / "salary_range.joblib").exists():
    import train  # noqa: F401  (trains on first start if models are missing)
rep = json.load(open(M / "report.json"))
art = joblib.load(M / "salary_range.joblib")
so_art = joblib.load(M / "so_pay.joblib")
so = pd.read_csv(D / "so_success.csv.gz")
onet = pd.read_csv(D / "onet_skills.csv")
onet_info = pd.read_csv(D / "onet_info.csv").fillna("").set_index("occupation")
S, SU = rep["salary"], rep["success"]
R = S["results"]["full_model"]


# ------------------------------------------------------------------ salary (India, Naukri)
SALARY_ROLES = ["Data Analyst", "Data Scientist", "Data Engineer", "Machine Learning Engineer", "Business Analyst", "Software Engineer",
                "DevOps Engineer", "QA Engineer"]
SALARY_LEVELS = [("", "Mid"), ("Junior", "Junior"), ("Senior", "Senior"), ("Lead", "Lead"), ("Manager", "Manager")]
_vocab = [x for x in art["vocab"]["skills"] if x not in GENERIC]
SALARY_SKILLS = [(x, fmt_skill(x)) for x in _vocab]
SALARY_POPULAR = sorted(SALARY_SKILLS, key=lambda kv: -SKILL_COUNT.get(kv[0], 0))[:30]
SALARY_CITIES = [(c, c.replace(" (all areas)", "").title().replace("Ncr", "NCR")) for c in art["vocab"]["locs"] if c]


def _predict(rows):
    X = featurize(pd.DataFrame(rows), art["vocab"])[art["columns"]]
    q = np.exp(np.column_stack([art["models"][k].predict(X) for k in (0.1, 0.5, 0.9)]))
    q[:, 0] = np.minimum(q[:, 0], q[:, 1]); q[:, 2] = np.maximum(q[:, 2], q[:, 1])
    return q


def salary_estimate(role, skills, cities, exp_min, exp_max):
    role_l, sk, lc = (role or "").lower().strip(), norm_list(",".join(skills)), norm_list(",".join(cities))
    if exp_max < exp_min: return {"error": "Maximum experience must be at least the minimum."}
    if not in_domain(role_l, art["vocab"]):
        why = ("Enter a job title to get an estimate." if not role_l else
               f"“{role.strip()}” isn't a role this model knows. It learned from Indian tech and data postings, and this title matches none of them.")
        return {"error": why, "hint": True}
    base = {"role_text": role_l, "skills_text": sk, "loc_text": lc}
    lo, mid, hi = _predict([{**base, "exp_min": exp_min, "exp_max": exp_max}])[0]
    span, xs = exp_max - exp_min, np.arange(0, 26)
    curve = _predict([{**base, "exp_min": float(x), "exp_max": float(x + span)} for x in xs])
    em = (exp_min + exp_max) / 2
    near = posts[(posts.exp_mid - em).abs() <= 1.5]
    if len(near) < 30: near = posts[(posts.exp_mid - em).abs() <= 3]
    notes, nv = [], novelty(role_l, sk, art["vocab"])
    if nv["role_ratio"] < 1.0: notes.append("Part of this job title is unfamiliar to the model, so it is mostly judged from the familiar words.")
    if nv["skills_total"] and nv["skills_known"] == 0: notes.append("None of these skills are common in the training data, so they add little to the estimate.")
    if exp_max > S["exp_max_trained"] or exp_min > S["exp_min_trained"]:
        notes.append(f"Experience is beyond the training data (maximum seen: {S['exp_max_trained']:.0f} years); the model holds the top level flat here.")
    hc, hb = hist(near.sal_mid.values)
    return {"hcounts": hc, "hbins": hb, "near_n": len(near), "near_med": float(near.sal_mid.median()), "lo": float(lo), "mid": float(mid), "hi": float(hi), "xs": xs, "curve": curve, "em": em, "notes": notes,
            "exp_min": exp_min, "exp_max": exp_max, "mae": R["mae_lpa"], "base_mae": S["results"]["median_baseline_mae_lpa"],
            "coverage": R["range80_coverage"], "n": S["n"]}


# ------------------------------------------------------------------ success odds (Stack Overflow survey)
SO_CATS, SO_TOOLS = so_art["categories"], [t[5:] for t in so_art["tools"]]
BANDS = [(-1, 2, "0-2"), (2, 5, "3-5"), (5, 9, "6-9"), (9, 14, "10-14"), (14, 24, "15-24"), (24, 99, "25+")]
band_of = lambda e: next(b for lo, hi, b in BANDS if lo < e <= hi)
so["band"] = so.exp.map(band_of)


def _so_frame(country, exp, role, edu, org, mgr, mode, tools, extra=None):
    row = {"country": country, "education": edu, "role": role, "org_size": org, "manager": mgr, "work_mode": mode, "exp": float(exp)}
    row.update({"tool_" + t: int(t in tools) for t in SO_TOOLS})
    rows = [row] + [{**row, "tool_" + t: 1} for t in (extra or [])]
    return pd.DataFrame(rows)[so_art["cat"] + so_art["num"]]


def success_odds(country, exp, role, edu, org, mgr, mode, tools):
    missing = [t for t in SO_TOOLS if t not in tools]
    p = so_art["model"].predict_proba(_so_frame(country, exp, role, edu, org, mgr, mode, tools, missing))[:, 1]
    p0, lifts = float(p[0]), sorted(zip(missing, p[1:] - p[0]), key=lambda x: -x[1])
    b = band_of(exp)
    peers = so[(so.country == country) & (so.band == b) & so.comp_usd.notna()]; scope = f"in {country} with {b} years of experience"
    if len(peers) < 15: peers = so[(so.country == country) & so.comp_usd.notna()]; scope = f"in {country}"
    sat = so[(so.role == role) & (so.band == b) & so.satisfied.notna()]; sscope = f"{role.lower()}s with {b} years"
    if len(sat) < 30: sat = so[(so.role == role) & so.satisfied.notna()]; sscope = f"{role.lower()}s"
    verdict = ("Likely to out-earn peers" if p0 >= 0.6 else "Slightly better than a coin flip" if p0 >= 0.52 else
               "About even with peers" if p0 >= 0.45 else "Less likely than peers to out-earn")
    r = SU["results"]["full_model"]
    return {"p": p0, "verdict": verdict, "peer_pay": float(peers.comp_usd.median()), "peer_n": len(peers), "scope": scope,
            "sat": float(sat.satisfied.mean()), "sat_n": len(sat), "sscope": sscope,
            "tips": [(t, d) for t, d in lifts[:5] if d >= 0.005], "acc": r["acc"], "guess": SU["results"]["majority_baseline_acc"],
            "ce": SU["results"]["country_exp_only"]["acc"], "n": SU["results"]["n"], "countries": SU["countries"]}


# ------------------------------------------------------------------ career fit (O*NET)
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


def career_fit(ratings):
    u = np.array(ratings, float) * 7 / 5                       # self-rating 0-5 -> O*NET level scale 0-7
    w = IM.values / 5; sw = np.sqrt(w)
    a, b = LV.values * sw, u[None, :] * sw
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.maximum(np.linalg.norm(b, axis=1), 1e-9))
    reach = np.clip((w * np.minimum(u[None, :], LV.values)).sum(1) / (w * LV.values).sum(1), 0, 1)
    fit = np.clip((cos - 0.55) / 0.45, 0, 1) * reach
    order = np.argsort(-fit)
    ranked = [(occ_names[i], float(fit[i])) for i in order[:8]]
    best = order[0]; name = occ_names[best]
    gap = w[best] * (LV.values[best] - u)
    gaps = [(SKILLS[i], LV.values[best][i] * 5 / 7, u[i] * 5 / 7) for i in np.argsort(-gap)[:3] if gap[i] > 0.05]
    info = onet_info.loc[name]
    imp = IM.values[best]
    compare = [(SKILLS[i], LV.values[best][i] * 5 / 7, u[i] * 5 / 7) for i in np.argsort(-imp)]
    return {"compare": compare, "ranked": ranked, "best": name, "gaps": gaps, "knowledge": info.knowledge, "styles": info.work_styles, "tools": info.tools,
            "n_occ": len(occ_names)}


CAREER_GROUPS = [("Technical", ["Mathematics", "Programming", "Systems Analysis", "Systems Evaluation", "Operations Analysis"]),
                 ("Thinking", ["Complex Problem Solving", "Critical Thinking", "Judgment and Decision Making", "Active Learning", "Reading Comprehension"]),
                 ("People and delivery", ["Management of Personnel Resources", "Monitoring", "Instructing", "Persuasion"])]
CAREER_PRESETS = {
    "Data analyst": {"Mathematics": 4, "Programming": 3, "Critical Thinking": 4, "Complex Problem Solving": 3, "Systems Analysis": 3, "Reading Comprehension": 4},
    "Software engineer": {"Programming": 5, "Mathematics": 3, "Complex Problem Solving": 4, "Systems Analysis": 4, "Active Learning": 4, "Systems Evaluation": 3},
    "Team lead": {"Management of Personnel Resources": 4, "Judgment and Decision Making": 4, "Monitoring": 4, "Persuasion": 4, "Instructing": 3, "Programming": 3},
    "Researcher": {"Mathematics": 5, "Critical Thinking": 5, "Complex Problem Solving": 4, "Reading Comprehension": 5, "Active Learning": 5, "Programming": 3},
}
