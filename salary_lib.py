"""Feature building + ordinal salary-band model (shared by train.py and app.py)."""
import re
import numpy as np, pandas as pd

BANDS = ["0to3", "3to6", "6to10", "10to15", "15to25", "25to50"]
BAND_LABEL = {"0to3": "0–3 LPA", "3to6": "3–6 LPA", "6to10": "6–10 LPA",
              "10to15": "10–15 LPA", "15to25": "15–25 LPA", "25to50": "25–50 LPA"}
EDGES = [0, 3, 6, 10, 15, 25, 50]          # band edges in LPA
SENIORITY = {
    "senior": r"\b(?:senior|sr)\b", "lead": r"\b(?:lead|principal|staff)\b",
    "manager": r"\b(?:manager|mgr)\b", "head": r"\b(?:head|director|vp|vice|chief|avp|gm)\b",
    "architect": r"\b(?:architect)\b", "junior": r"\b(?:junior|jr|associate|trainee|intern|fresher|graduate)\b",
}
STOP = set("and of the for in at to with a an or sr jr".split())

def norm_list(s):
    return re.sub(r"\s*,\s*", ",", (s or "").lower()).strip()

def parse_experience(s):
    m = re.search(r"(\d+)\s*-\s*(\d+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)

def role_words(text):
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in STOP and len(w) > 1}

def build_vocab(df, n_skills=200, n_words=150, n_loc=20):
    sk = df.skills_text.str.split(",").explode().str.strip()
    sk = sk[sk != ""].value_counts()
    lo = df.loc_text.str.split(",").explode().str.strip().value_counts()
    rw = pd.Series([w for t in df.role_text for w in role_words(t)]).value_counts()
    return {"skills": sk[sk >= 25].head(n_skills).index.tolist(),
            "words": rw[rw >= 25].head(n_words).index.tolist(),
            "locs": lo.head(n_loc).index.tolist()}

def featurize(df, vocab):
    """df needs: role_text, skills_text, loc_text, exp_min, exp_max."""
    out = pd.DataFrame(index=df.index)
    out["exp_min"], out["exp_max"] = df.exp_min.values, df.exp_max.values
    out["exp_mid"] = (df.exp_min + df.exp_max) / 2
    out["exp_span"] = df.exp_max - df.exp_min
    for k, rx in SENIORITY.items():
        out[f"sen_{k}"] = df.role_text.str.contains(rx, regex=True).astype(int)
    out["n_skills"] = df.skills_text.str.count(",") + (df.skills_text != "")
    out["n_locs"] = df.loc_text.str.count(",") + (df.loc_text != "")
    rs = [set(s.split(",")) for s in df.skills_text]
    rl = [set(s.split(",")) for s in df.loc_text]
    rw = [role_words(t) for t in df.role_text]
    cols = {f"skill_{s}": [int(s in x) for x in rs] for s in vocab["skills"]}
    cols.update({f"loc_{s}": [int(s in x) for x in rl] for s in vocab["locs"]})
    cols.update({f"word_{s}": [int(s in x) for x in rw] for s in vocab["words"]})
    return pd.concat([out, pd.DataFrame(cols, index=df.index)], axis=1)

def mono_vector(columns):
    return [1 if c in ("exp_min", "exp_max", "exp_mid") else 0 for c in columns]

def cum_to_band_probs(cum):
    """cum: (n,5) array of P(salary > edge_k), k=1..5. Enforce monotone, return (n,6)."""
    cum = np.minimum.accumulate(np.clip(cum, 0, 1), axis=1)
    p = np.hstack([1 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]])
    return np.clip(p, 0, 1)

def novelty(row_text_role, row_skills, vocab):
    """Return what share of the user's input the model actually knows."""
    w = role_words(row_text_role)
    s = {x for x in row_skills.split(",") if x}
    return {"role_known": len(w & set(vocab["words"])), "role_total": len(w),
            "skills_known": len(s & set(vocab["skills"])), "skills_total": len(s)}
