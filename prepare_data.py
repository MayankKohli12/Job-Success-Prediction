"""One-off: turn the big raw Naukri .jsonl files into the small CSVs the app/training use.
Usage: python prepare_data.py raw_data/naukri_data_scientist.jsonl raw_data/naukri_software_engineer.jsonl
(raw files are too big for GitHub, so only the processed CSVs in data/ are committed)."""
import sys, re
from collections import Counter
import numpy as np
import pandas as pd
from naukri_lib import read_raw, salary_frame, categorize
from salary_lib import role_words

paths = sys.argv[1:] or ["raw_data/naukri_data_scientist.jsonl", "raw_data/naukri_software_engineer.jsonl"]
raw = read_raw(paths)
print("unique postings:", len(raw))
sal = salary_frame(raw)
print("with usable salary:", len(sal))
sal[["jobId", "companyName", "role_text", "skills_text", "loc_text", "exp_min", "exp_max", "sal_lo", "sal_hi", "sal_mid"]].to_csv("data/naukri_salary.csv", index=False)

# vocabulary of titles / skills over ALL postings (used to detect out-of-scope inputs)
wc = Counter(w for t in raw.title.fillna("").str.lower() for w in set(role_words(t)))
pd.Series({w: c for w, c in wc.items() if c >= 10}).rename("n").to_csv("data/known_words.csv", header=True)
sc = Counter(s for t in raw.tagsAndSkills.fillna("") for s in set(x.strip().lower() for x in t.split(",") if x.strip()))
pd.Series({w: c for w, c in sc.items() if c >= 10}).rename("n").to_csv("data/known_skills.csv", header=True)

# company table: who is hiring for each role family, what experience and skills they ask for (all fields exist for every row)
from naukri_lib import parse_exp, clean_loc
GENERIC = {"data", "science", "processing", "languages", "language", "engineering", "development", "software", "analysis", "analytics", "tools",
           "technology", "web", "services", "systems", "design", "testing", "management", "learning", "machine", "it", "skills"}
raw["category"] = raw.title.map(categorize)
raw = raw.dropna(subset=["category", "companyName"]).copy()
raw[["emin", "emax"]] = [parse_exp(x) for x in raw.experience]
raw["city"] = raw.location.fillna("").map(lambda x: (clean_loc(x).split(",")[0].strip().title() or "India"))
raw["skill_list"] = raw.tagsAndSkills.fillna("").map(lambda t: [x.strip() for x in t.split(",") if x.strip() and x.strip().lower() not in GENERIC and len(x.strip()) > 1])

ROLE_ECHO = re.compile(r"analyst|scientist|engineer|developer|professional")
def top(items, k):
    """k most common items, case-insensitive, shown in their most common spelling."""
    from collections import Counter, defaultdict
    cnt, forms = Counter(), defaultdict(Counter)
    for x in items:
        if ROLE_ECHO.search(x.lower()): continue
        cnt[x.lower()] += 1; forms[x.lower()][x] += 1
    return ", ".join(forms[w].most_common(1)[0][0] for w, _ in cnt.most_common(k))

def agg(g):
    emin, emax = g.emin.median(), g.emax.median()
    return pd.Series({"postings": len(g),
                      "experience": "Varies" if np.isnan(emin) else f"{emin:.0f}–{emax:.0f} yrs",
                      "city": top(g.city, 2),
                      "skills": top([x for l in g.skill_list for x in l], 4) or "Varies by role"})

g = raw.groupby(["category", "companyName"]).apply(agg, include_groups=False).reset_index()
g = g[g.postings >= 3].sort_values(["category", "postings"], ascending=[True, False]).groupby("category").head(25)
g.to_csv("data/company_hiring.csv", index=False)

# headline numbers for each role family
sal_cat = sal.assign(category=sal.title.map(categorize)).dropna(subset=["category"])
cs = raw.groupby("category").agg(postings=("jobId", "count"), companies=("companyName", "nunique"), exp_med=("emin", "median")).reset_index()
cs = cs.merge(sal_cat.groupby("category").sal_mid.agg(median_salary_lpa="median", disclosed="count").reset_index(), on="category", how="left")
cs.round(1).to_csv("data/category_stats.csv", index=False)
print(cs.round(1).to_string()); print(g.groupby("category").size().to_dict())
