"""Job-market data layer: loads Naukri postings once, normalises them, and answers filter/aggregate queries."""
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from salary_lib import SENIORITY

D = Path("data")

# --------------------------------------------------------------------------- vocabulary
FAMILIES = [
    ("Data Scientist", r"data scien"),
    ("Machine Learning / AI Engineer", r"machine learning|\bml\b|\bai\b|gen ?ai|deep learning|\bnlp\b|computer vision"),
    ("Data Engineer", r"data engineer|\betl\b|big data|databricks|snowflake|hadoop|spark"),
    ("Data Analyst", r"data analy|analytics|\bbi\b|power ?bi|tableau"),
    ("Business Analyst", r"business analyst"),
    ("DevOps / Cloud", r"devops|cloud|\bsre\b|site reliability|kubernetes|infrastructure|platform engineer|\baws\b|azure"),
    ("QA / Testing", r"\bqa\b|test|sdet|quality"),
    ("Software Engineer / Developer", r"software|developer|java|python|\.net|full.?stack|backend|front.?end|sde|engineer|programmer|react|angular|node"),
]
OTHER = "Other tech"
LEVELS = ["Junior / Trainee", "Standard", "Senior", "Lead / Architect", "Manager & above"]
GENERIC = set("""development data software science languages tools processing tool technologies technology programming
 engineering developer application applications web design analysis analytical analytics management skills""".split()) - {"analytics"}
GENERIC |= {"development", "data", "software", "science", "languages", "tools", "processing", "technology", "programming", "design", "management"}
AGENCY_RX = r"consult|staffing|recruit|\bhr\b|human resource|manpower|placement|talent|headhunt|resourc|search"

ACRONYMS = {"sql", "aws", "gcp", "etl", "api", "apis", "ui", "ux", "qa", "ml", "ai", "nlp", "html", "css", "php", "bi", "sap", "rest",
            "sdlc", "oop", "oops", "nosql", "jvm", "ios", "ci", "cd", "ssis", "ssrs", "erp", "crm", "jdbc", "xml", "json", "sdet",
            "sre", "it", "hr", "ocr", "llm", "llms", "rpa", "vba", "dsa", "mvc", "orm", "tdd", "bdd", "jms", "soap", "ssas", "dax"}
SPECIAL = {"javascript": "JavaScript", "typescript": "TypeScript", "devops": "DevOps", "github": "GitHub", "mysql": "MySQL",
           "postgresql": "PostgreSQL", "mongodb": "MongoDB", "pyspark": "PySpark", "tensorflow": "TensorFlow", "pytorch": "PyTorch",
           "scikit-learn": "scikit-learn", "power bi": "Power BI", "node.js": "Node.js", "react.js": "React.js", "angular js": "AngularJS",
           "vue.js": "Vue.js", "next.js": "Next.js", "c#": "C#", "c++": "C++", ".net": ".NET", "asp.net": "ASP.NET", "vb.net": "VB.NET",
           "dotnet": ".NET", "springboot": "Spring Boot", "numpy": "NumPy", "powerbi": "Power BI", "linux": "Linux", "kafka": "Kafka",
           "servicenow": "ServiceNow", "salesforce": "Salesforce", "aws cloud": "AWS Cloud", "ms excel": "MS Excel"}
CITY_FIX = {"bangalore": "bengaluru", "bangalore rural": "bengaluru", "navi mumbai": "mumbai", "gurugram": "gurgaon",
            "new delhi": "delhi ncr", "delhi": "delhi ncr", "noida": "noida", "greater noida": "noida", "hyderabad / secunderabad": "hyderabad",
            "secunderabad": "hyderabad", "thane": "mumbai", "mumbai suburbs": "mumbai"}


def slugify(s):
    s = s.lower().replace("c++", "c plus plus").replace("c#", "c sharp").replace(".", " dot ").replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "x"


def fmt_skill(s):
    s = s.strip().lower()
    if s in SPECIAL: return SPECIAL[s]
    return " ".join(w.upper() if w in ACRONYMS else w[:1].upper() + w[1:] for w in s.split(" "))


def fmt_title(s):
    def one(m):
        w = m.group(0); lw = w.lower()
        if lw in ACRONYMS: return w.upper()
        if lw in SPECIAL: return SPECIAL[lw]
        return w[:1].upper() + w[1:].lower()
    return re.sub(r"[A-Za-z][A-Za-z0-9+#.]*", one, s or "")


def fmt_city(c):
    return {"delhi ncr": "Delhi NCR"}.get(c, c.title())


def norm_city(c):
    c = re.sub(r"\s*\((all areas|[^)]*)\)", "", c.strip().lower()).strip(" -")
    c = CITY_FIX.get(c, c)
    return c or None


def level_of(t):
    for key, label in (("head", "Manager & above"), ("manager", "Manager & above"), ("lead", "Lead / Architect"),
                       ("architect", "Lead / Architect"), ("senior", "Senior"), ("junior", "Junior / Trainee")):
        if re.search(SENIORITY[key], t): return label
    return "Standard"


def family_of(t):
    for name, rx in FAMILIES:
        if re.search(rx, t): return name
    return OTHER


# --------------------------------------------------------------------------- load
posts = pd.read_csv(D / "naukri_salary.csv")
posts["jobId"] = posts.jobId.astype(str)
posts["companyName"] = posts.companyName.fillna("Unnamed company").str.strip()
posts["title"] = posts.role_text.map(fmt_title)
posts["family"] = posts.role_text.map(family_of)
posts["level"] = posts.role_text.map(level_of)
posts["agency"] = posts.companyName.str.lower().str.contains(AGENCY_RX, regex=True)
posts["skills"] = posts.skills_text.fillna("").map(lambda s: list(dict.fromkeys(x.strip() for x in s.split(",") if x.strip())))
posts["skillset"] = posts.skills.map(frozenset)
posts["cities"] = posts.loc_text.fillna("").map(lambda s: list(dict.fromkeys(c for c in (norm_city(x) for x in s.split(",")) if c)))
posts["cityset"] = posts.cities.map(frozenset)
FOREIGN = {"united states", "united kingdom", "canada", "germany", "australia", "singapore", "uae", "saudi arabia", "dallas", "florida", "texas",
           "new jersey", "new york", "california", "london", "toronto", "dubai", "abu dhabi", "riyadh", "sydney", "melbourne", "berlin", "ireland",
           "netherlands", "qatar", "kuwait", "oman", "malaysia", "japan", "france", "usa", "uk", "bahrain", "philippines", "new zealand"}
ABROAD = int(posts.cityset.map(lambda s: bool(s & FOREIGN)).sum())
posts = posts[~posts.cityset.map(lambda s: bool(s & FOREIGN))].copy()      # pay for overseas roles is not in rupee terms
posts["exp_mid"] = (posts.exp_min + posts.exp_max) / 2
posts = posts.set_index("jobId", drop=False); posts.index.name = None

TOTAL = len(posts)
MEDIAN = float(posts.sal_mid.median())
FAMILY_LIST = [n for n, _ in FAMILIES] + [OTHER]
_slugs, FAM_SLUG = set(), {}
for n in FAMILY_LIST: FAM_SLUG[slugify(n)] = n


def _index(names):
    out, seen = {}, set()
    for n in names:
        s = slugify(n)
        while s in seen: s += "-2"
        seen.add(s); out[s] = n
    return out


COMPANY_SLUG = _index(posts.companyName.value_counts().index)
COMPANY_SLUG_REV = {v: k for k, v in COMPANY_SLUG.items()}
_skill_counts = Counter(s for ss in posts.skills for s in ss)
SKILL_COUNT = {s: n for s, n in _skill_counts.items() if s not in GENERIC}
SKILL_SLUG = _index([s for s, _ in sorted(SKILL_COUNT.items(), key=lambda x: -x[1])])
SKILL_SLUG_REV = {v: k for k, v in SKILL_SLUG.items()}
_city_counts = Counter(c for cs in posts.cities for c in cs)
CITY_COUNT = dict(_city_counts)

hiring = pd.read_csv(D / "company_hiring.csv")
cat_stats = pd.read_csv(D / "category_stats.csv").set_index("category")


# --------------------------------------------------------------------------- filters
SORTS = {"relevance": "Best match", "pay_desc": "Highest pay", "pay_asc": "Lowest pay", "exp_asc": "Least experience", "skills": "Most skills listed"}
LIST_KEYS = ("family", "level", "company", "skill", "xskill", "city")


class Filters:
    def __init__(self, qp):
        g = lambda k: [v for v in qp.getlist(k) if v != ""]
        num = lambda k: (lambda v: float(v) if re.fullmatch(r"\d+(\.\d+)?", v or "") else None)(qp.get(k))
        self.q = (qp.get("q") or "").strip()
        self.lists = {k: g(k) for k in LIST_KEYS}
        self.mode = "any" if qp.get("mode") == "any" else "all"
        self.exp, self.smin, self.smax = num("exp"), num("smin"), num("smax")
        self.hide_agency = qp.get("agency") == "hide"
        self.sort = qp.get("sort") if qp.get("sort") in SORTS else "relevance"
        try: self.page = max(1, int(qp.get("page") or 1))
        except ValueError: self.page = 1

    def raw(self):
        r = {k: list(v) for k, v in self.lists.items() if v}
        if self.q: r["q"] = [self.q]
        if self.mode == "any": r["mode"] = ["any"]
        for k, v in (("exp", self.exp), ("smin", self.smin), ("smax", self.smax)):
            if v is not None: r[k] = [("%g" % v)]
        if self.hide_agency: r["agency"] = ["hide"]
        if self.sort != "relevance": r["sort"] = [self.sort]
        return r

    def active(self):
        """Human labels for each applied filter: (label, drop-params)."""
        a = []
        if self.q: a.append((f"“{self.q}”", {"q": None}))
        for k, pre in (("family", ""), ("level", ""), ("company", ""), ("city", "")):
            for v in self.lists[k]: a.append((fmt_city(v) if k == "city" else v, {k: v}))
        for v in self.lists["skill"]: a.append((fmt_skill(v), {"skill": v}))
        for v in self.lists["xskill"]: a.append(("Not " + fmt_skill(v), {"xskill": v}))
        if self.exp is not None: a.append((f"{self.exp:g} yrs experience", {"exp": None}))
        if self.smin is not None or self.smax is not None:
            lo, hi = self.smin, self.smax
            a.append((f"₹{lo:g}–{hi:g} LPA" if lo is not None and hi is not None else f"≥ ₹{lo:g} LPA" if lo is not None else f"≤ ₹{hi:g} LPA", {"smin": None, "smax": None}))
        if self.hide_agency: a.append(("No recruiting agencies", {"agency": None}))
        return a


def apply(f, df=None):
    d = posts if df is None else df
    m = pd.Series(True, index=d.index)
    L = f.lists
    if f.q:
        hay = (d.role_text + " " + d.companyName.str.lower() + " " + d.skills_text.fillna(""))
        for tok in f.q.lower().split(): m &= hay.str.contains(re.escape(tok), regex=True)
    if L["family"]: m &= d.family.isin(L["family"])
    if L["level"]: m &= d.level.isin(L["level"])
    if L["company"]: m &= d.companyName.isin(L["company"])
    if L["city"]: m &= d.cityset.map(lambda s: any(c in s for c in L["city"]))
    if L["skill"]:
        want = frozenset(L["skill"])
        m &= d.skillset.map(lambda s: want <= s) if f.mode == "all" else d.skillset.map(lambda s: bool(want & s))
    if L["xskill"]:
        bad = frozenset(L["xskill"]); m &= d.skillset.map(lambda s: not (bad & s))
    if f.exp is not None: m &= (d.exp_min <= f.exp) & (d.exp_max >= f.exp)
    if f.smin is not None: m &= d.sal_mid >= f.smin
    if f.smax is not None: m &= d.sal_mid <= f.smax
    if f.hide_agency: m &= ~d.agency
    out = d[m]
    s = f.sort
    if s == "pay_desc" or (s == "relevance" and not L["skill"]): out = out.sort_values(["sal_mid", "jobId"], ascending=[False, True])
    elif s == "pay_asc": out = out.sort_values("sal_mid")
    elif s == "exp_asc": out = out.sort_values(["exp_min", "sal_mid"], ascending=[True, False])
    elif s == "skills": out = out.assign(_n=out.skills.map(len)).sort_values("_n", ascending=False)
    else:
        want = frozenset(L["skill"])
        out = out.assign(_m=out.skillset.map(lambda x: len(want & x))).sort_values(["_m", "sal_mid"], ascending=[False, False])
    return out


# --------------------------------------------------------------------------- aggregates
def quart(s):
    if len(s) == 0: return (0, 0, 0)
    return tuple(float(x) for x in np.percentile(s, [25, 50, 75]))


def hist(vals, top=60, step=3):
    bins = np.arange(0, top + step, step)
    c, _ = np.histogram(np.clip(vals, 0, top), bins=bins)
    return c.tolist(), bins.tolist()


def top_counts(series_of_lists, n=8, skip=(), generic=False):
    c = Counter(x for xs in series_of_lists for x in xs if x not in skip and (generic or x not in GENERIC))
    return c.most_common(n)


def insights(df, f=None):
    sel_s = set(f.lists["skill"]) if f else set()
    sel_c = set(f.lists["city"]) if f else set()
    q1, med, q3 = quart(df.sal_mid)
    counts, bins = hist(df.sal_mid.values)
    return {"n": len(df), "q1": q1, "med": med, "q3": q3, "counts": counts, "bins": bins,
            "skills": top_counts(df.skills, 10, sel_s), "cities": top_counts(df.cities, 8, sel_c),
            "companies": list(df.companyName.value_counts().head(8).items()),
            "families": list(df.family.value_counts().items()), "companies_n": df.companyName.nunique(),
            "exp_med": float(df.exp_mid.median()) if len(df) else 0}


def facet_counts(df, col):
    return df[col].value_counts().to_dict()


def skill_table(sub, q="", min_n=5, sort="demand", limit=200):
    if sub.empty: return []
    x = sub[["skills", "sal_mid"]].explode("skills").dropna()
    x = x[~x.skills.isin(GENERIC)]
    g = x.groupby("skills").sal_mid.agg(n="size", med="median").reset_index()
    base = float(sub.sal_mid.median())
    g["premium"] = g.med - base
    g["share"] = g.n / len(sub)
    g = g[g.n >= min_n]
    if q: g = g[g.skills.str.contains(re.escape(q.lower()))]
    key = {"demand": ("n", False), "pay": ("med", False), "premium": ("premium", False), "az": ("skills", True)}[sort]
    g = g.sort_values([key[0], "n"], ascending=[key[1], False]).head(limit)
    return [dict(name=r.skills, label=fmt_skill(r.skills), slug=SKILL_SLUG_REV.get(r.skills), n=int(r.n), med=r.med, premium=r.premium, share=r.share) for r in g.itertuples()]


def company_table(sub, q="", min_n=1, sort="postings", limit=300):
    if sub.empty: return [], 0
    g = sub.groupby("companyName").agg(n=("jobId", "size"), med=("sal_mid", "median"), top=("sal_mid", "max"),
                                       exp=("exp_mid", "median"), agency=("agency", "first")).reset_index()
    if q:
        for tok in q.lower().split(): g = g[g.companyName.str.lower().str.contains(re.escape(tok))]
    g = g[g.n >= min_n]
    total = len(g)
    key = {"postings": ("n", False), "pay": ("med", False), "top": ("top", False), "exp": ("exp", True), "az": ("companyName", True)}[sort]
    g = g.sort_values([key[0], "n"], ascending=[key[1], False]).head(limit)
    rows = []
    for r in g.itertuples():
        s = sub[sub.companyName == r.companyName]
        rows.append(dict(name=r.companyName, slug=COMPANY_SLUG_REV.get(r.companyName), n=int(r.n), med=r.med, top=r.top, exp=r.exp, agency=bool(r.agency),
                         skills=[fmt_skill(k) for k, _ in top_counts(s.skills, 3)], cities=[fmt_city(k) for k, _ in top_counts(s.cities, 2, generic=True)]))
    return rows, total


def exp_bands(df):
    bands = [(0, 2, "0–2 yrs"), (3, 5, "3–5 yrs"), (6, 9, "6–9 yrs"), (10, 14, "10–14 yrs"), (15, 99, "15+ yrs")]
    out = []
    for lo, hi, label in bands:
        s = df[(df.exp_mid >= lo) & (df.exp_mid < hi + 1)].sal_mid
        if len(s) >= 5:
            q1, med, q3 = quart(s); out.append(dict(label=label, n=len(s), q1=q1, med=med, q3=q3))
    return out


def co_skills(df, skill, n=10):
    sub = df[df.skillset.map(lambda s: skill in s)]
    c = Counter(x for xs in sub.skills for x in xs if x != skill and x not in GENERIC)
    out = []
    for k, v in c.most_common(40):
        base = SKILL_COUNT.get(k, 1) / TOTAL
        out.append((k, v, (v / max(len(sub), 1)) / base))
    return sorted(out, key=lambda x: -x[1])[:n], len(sub)


def suggest(kind, q, limit=12):
    q = (q or "").lower().strip()
    if kind == "skill": pool = [(k, v) for k, v in SKILL_COUNT.items() if v >= 3]; fmt = fmt_skill
    elif kind == "city": pool = list(CITY_COUNT.items()); fmt = fmt_city
    else: pool = list(posts.companyName.value_counts().items()); fmt = lambda x: x
    hits = [(k, v) for k, v in pool if q in k.lower()]
    hits.sort(key=lambda kv: (not kv[0].lower().startswith(q), -kv[1]))
    return [{"value": k, "label": fmt(k), "n": int(v)} for k, v in hits[:limit]]


def url(path, params, add=None, drop=None, set=None):
    p = {k: list(v) for k, v in params.items()}
    for k, v in (set or {}).items():
        p[k] = v if isinstance(v, list) else [v]
    for k, v in (add or {}).items():
        p.setdefault(k, [])
        if v not in p[k]: p[k].append(v)
    for k, v in (drop or {}).items():
        if v is None: p.pop(k, None)
        else: p[k] = [x for x in p.get(k, []) if x != v]
    if not (set and "page" in set): p.pop("page", None)
    pairs = [(k, x) for k, vs in p.items() for x in vs if x not in ("", None)]
    return path + ("?" + urlencode(pairs) if pairs else "")
