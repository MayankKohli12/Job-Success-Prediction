"""One-off: turn the big raw Naukri .jsonl files into the small CSVs the app/training use.
Usage: python prepare_data.py raw_data/naukri_data_scientist.jsonl raw_data/naukri_software_engineer.jsonl
(raw files are too big for GitHub, so only the processed CSVs in data/ are committed)."""
import sys, re
from collections import Counter
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

# company table: who is hiring for each role family, and what disclosed salaries look like
raw["category"] = raw.title.map(categorize)
raw = raw.dropna(subset=["category", "companyName"])
s = sal.set_index("jobId").sal_mid
raw["sal_mid"] = raw.jobId.map(s)
g = raw.groupby(["category", "companyName"]).agg(postings=("jobId", "count"), openings=("vacancy", lambda v: v.fillna(1).sum()),
                                                 disclosed=("sal_mid", "count"), median_salary_lpa=("sal_mid", "median")).reset_index()
g.loc[g.disclosed < 2, "median_salary_lpa"] = None
g = g[g.postings >= 3].sort_values(["category", "postings"], ascending=[True, False]).groupby("category").head(40)
g.round(1).to_csv("data/company_hiring.csv", index=False)
print(g.category.value_counts().to_dict())
