"""Train the salary-range model (Naukri postings) and the career-fit model (CareerMap).
Run: python train.py     (needs data/*.csv, produced by prepare_data.py and committed)"""
import json, joblib, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from salary_lib import build_vocab, featurize, mono_vector

warnings.filterwarnings("ignore")
D, OUT, RS = Path("data"), Path("models"), 42
OUT.mkdir(exist_ok=True)
report = {}

# ------------------------------------------------------------------ 1. salary range model
a = pd.read_csv(D / "naukri_salary.csv").fillna({"role_text": "", "skills_text": "", "loc_text": "", "companyName": ""})
y = np.log(a.sal_mid.values)                       # model log-salary: errors are proportional, not absolute

# identical postings (same role/company/location) never straddle the train/test split
grp = a.groupby(["role_text", "companyName", "loc_text"]).ngroup()
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=RS).split(a, groups=grp))
KW = dict(min_skill=15, min_word=15, n_skills=150, n_words=120, n_loc=15)
vocab = build_vocab(a.iloc[tr], **KW)

def fit(F, yv, cols, q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_depth=4, learning_rate=0.05, max_iter=300,
                                         min_samples_leaf=25, l2_regularization=1.0, random_state=RS,
                                         monotonic_cst=mono_vector(cols)).fit(F[cols], yv)

def evaluate(F, cols):
    ms = {q: fit(F.iloc[tr], y[tr], cols, q) for q in (0.1, 0.5, 0.9)}
    lo, md, hi = (ms[q].predict(F.iloc[te][cols]) for q in (0.1, 0.5, 0.9))
    lo, hi = np.minimum(lo, md), np.maximum(hi, md)
    t = a.sal_mid.values[te]; p = np.exp(md)
    r2 = 1 - ((y[te] - md) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    return {"mae_lpa": float(np.abs(p - t).mean()), "r2_log": float(r2), "within_30pct": float(np.mean(np.abs(p - t) / t <= 0.3)),
            "range80_coverage": float(np.mean((y[te] >= lo) & (y[te] <= hi))), "range80_median_width_x": float(np.median(np.exp(hi - lo)))}

F = featurize(a, vocab)
ecols = ["exp_min", "exp_max", "exp_mid", "exp_span"]
res = {"median_baseline_mae_lpa": float(np.abs(np.median(a.sal_mid.values[tr]) - a.sal_mid.values[te]).mean()),
       "experience_only": evaluate(F, ecols), "full_model": evaluate(F, list(F.columns))}
print(json.dumps(res, indent=1))

# final model on all data, vocab from all data; scope vocab from ALL postings (not just the salaried ones)
vocab = build_vocab(a, **KW)
vocab["known_words"] = pd.read_csv(D / "known_words.csv").iloc[:, 0].tolist()
vocab["known_skills"] = pd.read_csv(D / "known_skills.csv").iloc[:, 0].tolist()
F = featurize(a, vocab); cols = list(F.columns)
joblib.dump({"models": {q: fit(F, y, cols, q) for q in (0.1, 0.5, 0.9)}, "vocab": vocab, "columns": cols}, OUT / "salary_range.joblib", compress=3)
report["salary"] = {"n": len(a), "results": res, "exp_max_trained": float(a.exp_max.max()), "exp_min_trained": float(a.exp_min.max()),
                    "locations": vocab["locs"]}

# ------------------------------------------------------------------ 2. career-fit model (synthetic dataset!)
cm = pd.read_csv(D / "careermap.csv")
skills = cm.columns[:17].tolist()
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
cv = cross_val_score(pipe, cm[skills], cm.Role, cv=StratifiedKFold(5, shuffle=True, random_state=RS)).mean()
pipe.fit(cm[skills], cm.Role)
joblib.dump(pipe, OUT / "career_fit.joblib")
report["career"] = {"n": len(cm), "cv_accuracy": float(cv), "skills": skills, "roles": sorted(cm.Role.unique().tolist())}
print("career fit CV acc", round(cv, 4))
json.dump(report, open(OUT / "report.json", "w"), indent=2)
print("saved")
