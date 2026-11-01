import re, numpy as np, pandas as pd

def _lpa(tok):
    """'3 Lacs' / '50,000' / '1.5 Cr' / '5 L' -> LPA (lakhs per annum)."""
    t = tok.strip().lower().replace(",", "")
    m = re.match(r"^([\d.]+)\s*(lacs?|l|cr|crores?)?$", t)
    if not m: return None
    v, u = float(m.group(1)), (m.group(2) or "")
    if u.startswith("cr"): return v * 100
    if u.startswith("l"): return v
    return v / 1e5                      # plain rupees per year

def parse_salary(s):
    """Return (lo, hi) in LPA or None. Handles '12-15 Lacs PA', '50,000-3 Lacs PA', '80 Lacs-1.5 Cr PA', '25,000/month'."""
    if not isinstance(s, str): return None
    s = s.strip()
    if s in ("Not disclosed", "Unpaid", "") or s.startswith("0 PA"): return None
    monthly = "/month" in s.lower() or "monthly" in s.lower()
    core = re.sub(r"\s*(pa|p\.a\.|/month|\(monthly\))\s*", " ", s, flags=re.I).strip()
    parts = [p for p in re.split(r"\s*-\s*", core) if p]
    if monthly:
        vals = [(_lpa(p) or 0) * 12 * (1e5 if "lac" not in p.lower() and "cr" not in p.lower() else 1) / 1e5 * 1 for p in parts]
        vals = [v for v in vals if v]
        return (vals[0], vals[-1]) if vals else None
    if len(parts) == 2:
        # unit may be given only on the second token ("50,000-3 Lacs" means 0.5-3 Lacs; "12-15 Lacs")
        a, b = parts
        ub = re.search(r"(lacs?|cr)", b, re.I)
        if ub and not re.search(r"(lacs?|cr)", a, re.I):
            va = float(a.replace(",", "")) if a.replace(",", "").replace(".", "").isdigit() else None
            if va is None: return None
            lo = va / 1e5 if va >= 1000 else _lpa(a + " " + ub.group(1))
            hi = _lpa(b)
        else:
            lo, hi = _lpa(a), _lpa(b)
    elif len(parts) == 1:
        lo = hi = _lpa(parts[0])
    else:
        return None
    if lo is None or hi is None or hi < lo or hi <= 0: return None
    return lo, hi

def parse_exp(s):
    m = re.search(r"(\d+)\s*-\s*(\d+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)

def clean_loc(s):
    s = (s or "").lower()
    s = re.sub(r"\b(hybrid|remote|work from home|wfh)\b\s*-?\s*", "", s)
    return re.sub(r"\s*,\s*", ",", s).strip(" ,-")

def clean_skills(s):
    toks = [t.strip().lower() for t in (s or "").split(",") if t.strip()]
    return ",".join(dict.fromkeys(toks))

def read_raw(paths):
    cols = ["jobId", "title", "companyName", "tagsAndSkills", "experience", "salary", "location", "vacancy"]
    return pd.concat([pd.read_json(p, lines=True)[cols] for p in paths], ignore_index=True).drop_duplicates("jobId")

def salary_frame(raw):
    a = raw.copy()
    sal = a.salary.map(parse_salary)
    a = a[sal.notna()].copy()
    a["sal_lo"] = [x[0] for x in sal[sal.notna()]]; a["sal_hi"] = [x[1] for x in sal[sal.notna()]]
    a["sal_mid"] = (a.sal_lo + a.sal_hi) / 2
    a[["exp_min", "exp_max"]] = [parse_exp(x) for x in a.experience]
    a = a.dropna(subset=["exp_min", "exp_max"])
    a = a[(a.sal_mid >= 1) & (a.sal_mid <= 150) & (a.sal_hi / a.sal_lo <= 4)]   # drop junk / absurdly wide ranges
    a["role_text"] = a.title.fillna("").str.lower().str.strip()
    a["skills_text"] = a.tagsAndSkills.fillna("").map(clean_skills)
    a["loc_text"] = a.location.fillna("").map(clean_loc)
    return a.reset_index(drop=True)

CATS = [("Data Scientist", r"data scien"), ("Machine Learning / AI Engineer", r"machine learning|\bml\b|\bai\b|gen ?ai|deep learning|\bnlp\b"),
        ("Data Engineer", r"data engineer"), ("Data Analyst", r"data analy|analytics|\bbi\b"),
        ("Business Analyst", r"business analyst"), ("Software Engineer / Developer", r"software|developer|java|python|\.net|full.?stack|backend|frontend|sde")]

def categorize(title):
    t = (title or "").lower()
    for name, rx in CATS:
        if re.search(rx, t): return name
    return None
