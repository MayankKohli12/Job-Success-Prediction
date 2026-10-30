"""Train three job-success models and save them to models/.
Run:  python train.py
"""
import re, json, joblib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

D, OUT, RS = Path("data"), Path("models"), 42
OUT.mkdir(exist_ok=True)
report = {}

def clean_cols(df):
    return df.rename(columns=lambda c: re.sub(r"[^0-9a-z]+", "_", c.strip().lower()).strip("_"))

# ---------- 1. Job-posting salary band (text + experience) ----------
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from salary_lib import (BANDS, build_vocab, featurize, mono_vector, cum_to_band_probs,
                        parse_experience, norm_list)

a = pd.read_csv(D / "Analytics_Jobs.csv")
a[["exp_min", "exp_max"]] = [parse_experience(x) for x in a["experience"]]
a = a.dropna(subset=["exp_min", "exp_max"]).reset_index(drop=True)
a["role_text"] = a["job_desig"].fillna("").str.lower().str.strip()
a["skills_text"] = a["key_skills"].fillna("").map(norm_list)
a["loc_text"] = a["location"].fillna("").map(norm_list)
a["band"] = a["salary"].astype(str).map({b: i for i, b in enumerate(BANDS)})

# identical postings (same experience/role/skills/location) must never be split across train and test
grp = a.groupby(["experience", "role_text", "skills_text", "loc_text"]).ngroup()
tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=RS).split(a, groups=grp))
vocab = build_vocab(a.iloc[tr_i])
F = featurize(a, vocab)
mono = mono_vector(F.columns)

def fit_ordinal(Xt, yt, cols_mono):
    ms = []
    for k in range(1, 6):   # one binary model per threshold: P(band >= k)
        m = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=250,
                                           min_samples_leaf=30, l2_regularization=1.0,
                                           monotonic_cst=cols_mono, random_state=RS)
        ms.append(m.fit(Xt, (yt >= k).astype(int)))
    return ms

def predict_ordinal(ms, X):
    return cum_to_band_probs(np.column_stack([m.predict_proba(X)[:, 1] for m in ms]))

def score(P, y):
    pred = P.argmax(1)
    exp_band = (P * np.arange(6)).sum(1)
    return {"accuracy": float((pred == y).mean()),
            "within_one_band": float((np.abs(pred - y) <= 1).mean()),
            "mae_bands": float(np.abs(exp_band - y).mean())}

ytr, yte = a.band.values[tr_i], a.band.values[te_i]
res = {}
maj = np.bincount(ytr).argmax()
res["majority_baseline"] = {"accuracy": float((yte == maj).mean()), "within_one_band": float((np.abs(yte - maj) <= 1).mean()),
                            "mae_bands": float(np.abs(yte - maj).mean())}
ecols = ["exp_min", "exp_max", "exp_mid", "exp_span"]
m_exp = fit_ordinal(F.iloc[tr_i][ecols], ytr, mono_vector(ecols))
res["experience_only"] = score(predict_ordinal(m_exp, F.iloc[te_i][ecols]), yte)
m_full = fit_ordinal(F.iloc[tr_i], ytr, mono)
res["full_model"] = score(predict_ordinal(m_full, F.iloc[te_i]), yte)
for k, v in res.items():
    print("salary band", k, {x: round(y, 3) for x, y in v.items()})
# final model: refit on all rows
vocab = build_vocab(a); F = featurize(a, vocab)
m_final = fit_ordinal(F, a.band.values, mono)
joblib.dump({"models": m_final, "vocab": vocab, "columns": F.columns.tolist()}, OUT / "salary_band.joblib", compress=3)
report["salary_band"] = {"results": res, "n": len(a), "n_train_groups": int(grp.iloc[tr_i].nunique()),
                         "exp_max_trained": float(a.exp_max.max()), "exp_min_trained": float(a.exp_min.max())}
report["skill_hints"] = vocab["skills"][:40]
report["locations"] = vocab["locs"]

# ---------- 2/3. Numeric trait models (already clean apart from column names) ----------
def fit_binary(df, feats, target, name):
    Xn, yn = df[feats], df[target].astype(int)
    out = {}
    for n, m in {"LogReg": LogisticRegression(max_iter=1000),
                 "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RS)}.items():
        p = Pipeline([("sc", StandardScaler()), ("clf", m)])
        cv = cross_val_score(p, Xn, yn, cv=StratifiedKFold(5, shuffle=True, random_state=RS), scoring="accuracy")
        out[n] = {"cv_accuracy": cv.mean(), "cv_std": cv.std()}
        print(name, n, out[n])
    bn = max(out, key=lambda k: out[k]["cv_accuracy"])
    mdl = LogisticRegression(max_iter=1000) if bn == "LogReg" else RandomForestClassifier(n_estimators=300, random_state=RS)
    pipe = Pipeline([("sc", StandardScaler()), ("clf", mdl)]).fit(Xn, yn)
    joblib.dump(pipe, OUT / f"{name}.joblib")
    ranges = {f: [float(Xn[f].min()), float(Xn[f].max())] for f in feats}
    report[name] = {"best": bn, "results": out, "features": feats, "ranges": ranges, "n": len(df),
                    "class_balance": yn.value_counts().to_dict()}

sk = clean_cols(pd.read_excel(D / "JDS_Skill_Traits.xlsx"))
fit_binary(sk, [c for c in sk.columns if c.endswith("_skills")], "salary_hike_high_or_low", "skills_hike")
pe = clean_cols(pd.read_excel(D / "SDS_Personality_Traits.xlsx"))
tgt = [c for c in pe.columns if c.startswith("success")][0]
fit_binary(pe, [c for c in pe.columns if c not in ("id", tgt)], tgt, "personality_success")

# ---------- company salary lookup (strings like "7.8L" -> 7.8 lakhs) ----------
ds = pd.read_csv(D / "DataScience_Jobs.csv")
for c in ["avg_salary", "min_salary", "max_salary"]:
    ds[c] = ds[c].astype(str).str.replace("L", "", regex=False).astype(float)
ds.to_csv(OUT / "company_salaries.csv", index=False)
json.dump(report, open(OUT / "report.json", "w"), indent=2, default=float)
print("saved")
