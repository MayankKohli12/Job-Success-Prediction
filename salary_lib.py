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

# ---- city handling: every city maps to a canonical name, a tier, and a data-driven pay index
CITY_ALIAS = {"bangalore": "bengaluru", "bangalore rural": "bengaluru", "bengaluru": "bengaluru", "gurugram": "gurgaon", "new delhi": "delhi ncr", "delhi": "delhi ncr",
              "ncr": "delhi ncr", "greater noida": "noida", "navi mumbai": "mumbai", "mumbai suburbs": "mumbai", "thane": "mumbai", "bombay": "mumbai",
              "ernakulam": "kochi", "cochin": "kochi", "calcutta": "kolkata", "madras": "chennai", "bhubaneshwar": "bhubaneswar", "thiruvananthapuram": "trivandrum",
              "vizag": "visakhapatnam", "panchkula": "chandigarh", "mohali": "chandigarh", "gandhinagar": "ahmedabad"}
TIER1 = {"bengaluru", "hyderabad", "pune", "chennai", "mumbai", "gurgaon", "noida", "delhi ncr", "kolkata", "ahmedabad", "faridabad", "ghaziabad"}
TIER2 = {"chandigarh", "jaipur", "indore", "kochi", "coimbatore", "nagpur", "lucknow", "bhopal", "trivandrum", "vadodara", "surat", "visakhapatnam", "mysore",
         "mangalore", "bhubaneswar", "patna", "ranchi", "nashik", "kanpur", "dehradun", "guwahati", "vijayawada", "madurai", "raipur", "jodhpur", "udaipur",
         "ludhiana", "agra", "varanasi", "amritsar", "tiruchirappalli", "hubli", "belgaum", "goa", "panaji", "rajkot", "meerut", "jamshedpur", "kozhikode", "thrissur"}
TIER_NAME = {2: "metro", 1: "tier-2 city", 0: "smaller city"}

def canon_city(c):
    c = re.sub(r"\(.*?\)", "", (c or "").lower()).strip()
    c = re.sub(r"\s+", " ", c)
    return CITY_ALIAS.get(c, c)

def city_tier(c):
    return 2 if c in TIER1 else 1 if c in TIER2 else 0

def city_list(loc_text):
    return [x for x in dict.fromkeys(canon_city(c) for c in (loc_text or "").split(",")) if x]

def city_table(df, y, k=20):
    """Per-city pay index = shrunken mean of (log salary - overall mean). Cities with few postings are pulled towards their tier, tiers towards 0."""
    mu = float(np.mean(y)); sums, cnt = {}, {}
    for t, v in zip(df.loc_text, y):
        for c in city_list(t): sums[c] = sums.get(c, 0.0) + (v - mu); cnt[c] = cnt.get(c, 0) + 1
    tier_s, tier_n = {}, {}
    for c, n in cnt.items():
        t = city_tier(c); tier_s[t] = tier_s.get(t, 0.0) + sums[c]; tier_n[t] = tier_n.get(t, 0) + n
    tier_idx = {t: tier_s[t] / (tier_n[t] + k) for t in tier_s}
    return {"mu": mu, "k": k, "sums": sums, "cnt": cnt, "tier_idx": tier_idx}

def city_feats(cities, tab, own=None):
    """(tier, pay index, postings behind it) for a list of canonical cities. `own` = the row's own residual, removed when scoring training rows."""
    if not cities: return 0.0, 0.0, 0
    ti, ix, ns = [], [], []
    for c in cities:
        t = city_tier(c); base = tab["tier_idx"].get(t, 0.0)
        s, n = tab["sums"].get(c, 0.0), tab["cnt"].get(c, 0)
        if own is not None and n: s, n = s - own, n - 1
        ti.append(t); ix.append((s + tab["k"] * base) / (n + tab["k"])); ns.append(n)
    return float(max(ti)), float(np.mean(ix)), int(max(ns))

def city_columns(df, y, folds=5, seed=42):
    """Out-of-fold city features for training rows: each posting is scored from a table built on the OTHER folds, so it never sees its own salary."""
    rng = np.random.RandomState(seed); fid = rng.randint(0, folds, len(df)); out = np.zeros((len(df), 3))
    for f in range(folds):
        tab = city_table(df[fid != f], y[fid != f]); m = np.where(fid == f)[0]
        out[m] = [city_feats(city_list(t), tab) for t in df.loc_text.iloc[m]]
    return out

def parse_experience(s):
    m = re.search(r"(\d+)\s*-\s*(\d+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)

def role_words(text):
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in STOP and len(w) > 1}

def build_vocab(df, n_skills=200, n_words=150, n_loc=20, min_skill=25, min_word=25):
    sk = df.skills_text.str.split(",").explode().str.strip()
    sk = sk[sk != ""].value_counts()
    lo = df.loc_text.str.split(",").explode().str.strip().value_counts()
    rw = pd.Series([w for t in df.role_text for w in role_words(t)]).value_counts()
    return {"known_words": rw[rw >= 5].index.tolist(),      # for the in-domain check (broader than model features)
            "known_skills": sk[sk >= 5].index.tolist(),
            "skills": sk[sk >= min_skill].head(n_skills).index.tolist(),
            "words": rw[rw >= min_word].head(n_words).index.tolist(),
            "locs": lo.head(n_loc).index.tolist()}

def featurize(df, vocab):
    """df needs: role_text, skills_text, loc_text, exp_min, exp_max."""
    out = pd.DataFrame(index=df.index)
    out["exp_min"], out["exp_max"] = df.exp_min.values, df.exp_max.values
    out["exp_mid"] = (df.exp_min + df.exp_max) / 2
    out["exp_span"] = df.exp_max - df.exp_min
    for k, rx in SENIORITY.items():
        out[f"sen_{k}"] = df.role_text.str.contains(rx, regex=True).astype(int)
    out["city_tier"] = df["city_tier"].values if "city_tier" in df else 0.0
    out["city_idx"] = df["city_idx"].values if "city_idx" in df else 0.0
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

def novelty(role_text, skills_text, vocab):
    """How much of the user's input the training data has seen."""
    w = role_words(role_text)
    kw, ks = set(vocab["known_words"]), set(vocab["known_skills"])
    s = {x for x in skills_text.split(",") if x}
    return {"role_known": len(w & kw), "role_total": len(w),
            "role_ratio": (len(w & kw) / len(w)) if w else 0.0,
            "skills_known": len(s & ks), "skills_total": len(s)}

def in_domain(role_text, vocab, min_ratio=0.6):
    """True if the job title looks like something from the training postings."""
    n = novelty(role_text, "", vocab)
    return n["role_total"] > 0 and n["role_ratio"] >= min_ratio
