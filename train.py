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

# ---------- 1. Job-posting salary band: this is where the STRINGS are ----------
a = pd.read_csv(D / "Analytics_Jobs.csv")
# "6-10 yrs" -> numeric min / max years
ex = a["experience"].str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
a["exp_min"], a["exp_max"] = ex[0], ex[1]
a = a.dropna(subset=["exp_min", "exp_max"])
# free text -> one lowercase string; "Bengaluru, Chennai" -> comma separated tokens
a["role_text"] = a["job_desig"].fillna("").str.lower()
# normalise "a , b" -> "a,b" so a plain regex can split on commas (no lambdas => picklable)
a["skills_text"] = a["key_skills"].fillna("").str.lower().str.replace(r"\s*,\s*", ",", regex=True).str.strip()
a["loc_text"] = a["location"].fillna("").str.lower().str.replace(r"\s*,\s*", ",", regex=True).str.strip()
a["n_locations"] = a["loc_text"].str.count(",") + 1
a["n_skills"] = a["skills_text"].str.count(",") + 1
a["exp_mid"] = (a.exp_min + a.exp_max) / 2

pre = ColumnTransformer([
    ("role", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=5000, sublinear_tf=True), "role_text"),
    ("skills", TfidfVectorizer(token_pattern=r"[^,]+", min_df=5, max_features=4000, sublinear_tf=True), "skills_text"),
    ("loc", TfidfVectorizer(token_pattern=r"[^,]+", min_df=5, use_idf=False, norm=None), "loc_text"),
    ("num", StandardScaler(), ["exp_min", "exp_max", "exp_mid", "n_locations", "n_skills"]),
])
cols = ["role_text", "skills_text", "loc_text", "exp_min", "exp_max", "exp_mid", "n_locations", "n_skills"]
X, y = a[cols], a["salary"].astype(str)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
cands = {
    "LogReg": LogisticRegression(max_iter=2000, C=2.0),
    "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RS),
}
best = None
res = {}
for n, m in cands.items():
    p = Pipeline([("pre", pre), ("clf", m)]).fit(Xtr, ytr)
    pr = p.predict(Xte)
    order = {b: i for i, b in enumerate(["0to3", "3to6", "6to10", "10to15", "15to25", "25to50"])}
    within1 = float(np.mean(np.abs(pr.map(order) - yte.map(order)) <= 1)) if hasattr(pr, "map") else float(np.mean(np.abs(pd.Series(pr).map(order).values - yte.map(order).values) <= 1))
    res[n] = {"accuracy": accuracy_score(yte, pr), "macro_f1": f1_score(yte, pr, average="macro"), "within_one_band": within1}
    print("salary band", n, res[n])
    if best is None or res[n]["accuracy"] > best[1]:
        best = (n, res[n]["accuracy"], p)
joblib.dump(best[2], OUT / "salary_band.joblib", compress=3)
bands = ["0to3", "3to6", "6to10", "10to15", "15to25", "25to50"]
report["salary_band"] = {"best": best[0], "results": res, "n": len(a),
                         "classes": [b for b in bands if b in set(y)]}
# vocab for UI hints
report["skill_hints"] = a["skills_text"].str.split(",").explode().str.strip().value_counts().head(40).index.tolist()
report["locations"] = a["loc_text"].str.split(",").explode().str.strip().value_counts().head(25).index.tolist()

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
