"""Stack Overflow Developer Survey 2026 -> compact modelling table.
Usage: python prepare_so.py raw_data/so_results.csv    (raw file is 140 MB, so only the output CSV is committed)

Success definitions (both are *relative*, so no currency conversion is needed):
  high_pay  = pay (USD) above the median of people in the same country with similar work experience
  satisfied = job satisfaction >= 8 on the survey's 0-10 scale
"""
import sys, re
import numpy as np, pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "raw_data/so_results.csv"
df = pd.read_csv(src, low_memory=False, encoding="utf-8-sig")
emp = df.Employment.fillna("")
df = df[emp.str.startswith("Employed") | emp.str.startswith("Independent")].copy()

df["exp"] = pd.to_numeric(df.WorkExp, errors="coerce").fillna(pd.to_numeric(df.YearsCode, errors="coerce"))
df = df[df.exp.between(0, 50)]
df["comp_usd"] = pd.to_numeric(df.ConvertedCompYearly, errors="coerce")
df.loc[~df.comp_usd.between(1500, 1_000_000), "comp_usd"] = np.nan        # drop typos / junk
df["jobsat"] = pd.to_numeric(df.JobSat, errors="coerce")

top = df.Country.value_counts()
countries = top[top >= 300].index                                          # enough respondents to compare against peers
df = df[df.Country.isin(countries)]
df["country"] = (df.Country.str.replace(" of America", "").str.replace(" of Great Britain and Northern Ireland", "")
                 .str.replace("Iran, Islamic Republic of...", "Iran", regex=False))

def edu(s):
    s = str(s)
    if s.startswith("Master"): return "Master's degree"
    if s.startswith("Bachelor"): return "Bachelor's degree"
    if s.startswith("Professional"): return "Doctorate / professional degree"
    if s.startswith(("Some college", "Associate")): return "Some college / associate"
    return "School or other"
df["education"] = df.EdLevel.map(edu)

def role(s):
    s = str(s)
    for k, v in [("Data scientist", "Data scientist"), ("Data engineer", "Data engineer"), ("AI/ML", "AI / ML engineer"),
                 ("Data or business", "Data / business analyst"), ("full-stack", "Full-stack developer"), ("back-end", "Back-end developer"),
                 ("front-end", "Front-end developer"), ("mobile", "Mobile developer"), ("DevOps", "DevOps / cloud / sysadmin"),
                 ("Cloud", "DevOps / cloud / sysadmin"), ("System admin", "DevOps / cloud / sysadmin"), ("Architect", "Architect"),
                 ("manager", "Manager / executive / founder"), ("Senior executive", "Manager / executive / founder"), ("Founder", "Manager / executive / founder"),
                 ("QA", "QA / security"), ("Cybersecurity", "QA / security"), ("Academic", "Researcher / academic"), ("Research", "Researcher / academic")]:
        if k.lower() in s.lower(): return v
    return "Other developer / IT"
df["role"] = df.DevType.map(role)

org = {"Just me": "Solo / freelancer", "Less than 20": "Under 20", "20 to 99": "20 - 99", "100 to 499": "100 - 499", "500 to 999": "500 - 999",
       "1,000 to 4,999": "1,000 - 4,999", "5,000 to 9,999": "5,000 - 9,999", "10,000 or more": "10,000+"}
def orgsz(s):
    for k, v in org.items():
        if str(s).startswith(k): return v
    return "Unknown"
df["org_size"] = df.OrgSize.map(orgsz)
df["manager"] = df.ICorPM.map({"People manager": "People manager", "Individual contributor": "Individual contributor"}).fillna("Individual contributor")
def rem(s):
    s = str(s)
    if s.startswith("Remote"): return "Remote"
    if s.startswith("Hybrid"): return "Hybrid"
    if s.startswith("In-person"): return "In-person"
    if s.startswith("Your choice"): return "Flexible"
    return "Unknown"
df["work_mode"] = df.RemoteWork.map(rem)

TOOLS = {  # label -> (column, token)
    "Python": ("LanguageHaveWorkedWith", "Python"), "SQL": ("LanguageHaveWorkedWith", "SQL"), "R": ("LanguageHaveWorkedWith", "R"),
    "Java": ("LanguageHaveWorkedWith", "Java"), "JavaScript": ("LanguageHaveWorkedWith", "JavaScript"), "TypeScript": ("LanguageHaveWorkedWith", "TypeScript"),
    "C#": ("LanguageHaveWorkedWith", "C#"), "Go": ("LanguageHaveWorkedWith", "Go"), "Rust": ("LanguageHaveWorkedWith", "Rust"),
    "PostgreSQL": ("DatabaseHaveWorkedWith", "PostgreSQL"), "MySQL": ("DatabaseHaveWorkedWith", "MySQL"), "MongoDB": ("DatabaseHaveWorkedWith", "MongoDB"),
    "AWS": ("PlatformHaveWorkedWith", "Amazon Web Services (AWS)"), "Azure": ("PlatformHaveWorkedWith", "Microsoft Azure"),
    "Google Cloud": ("PlatformHaveWorkedWith", "Google Cloud"), "Docker": ("PlatformHaveWorkedWith", "Docker"), "Kubernetes": ("PlatformHaveWorkedWith", "Kubernetes"),
}
for lab, (col, tok) in TOOLS.items():
    df["tool_" + lab] = df[col].fillna("").map(lambda s, t=tok: int(t in [x.strip() for x in s.split(";")]))
    print(f"{lab:12}{df['tool_' + lab].mean():.1%}")
for c, n in [("LanguageHaveWorkedWith", "n_languages"), ("DatabaseHaveWorkedWith", "n_databases"), ("PlatformHaveWorkedWith", "n_platforms")]:
    df[n] = df[c].fillna("").map(lambda s: len([x for x in s.split(";") if x.strip()]))

# peer-relative pay label
df["exp_band"] = pd.cut(df.exp, [-1, 2, 5, 9, 14, 24, 60], labels=["0-2", "3-5", "6-9", "10-14", "15-24", "25+"])
g = df.groupby(["country", "exp_band"], observed=True).comp_usd
med_cell, n_cell = g.transform("median"), g.transform("count")
med_country = df.groupby("country").comp_usd.transform("median")
ref = np.where(n_cell >= 15, med_cell, np.where(df.groupby("country").comp_usd.transform("count") >= 30, med_country, np.nan))
df["high_pay"] = np.where(df.comp_usd.notna() & ~np.isnan(ref), (df.comp_usd > ref).astype(float), np.nan)
df["satisfied"] = np.where(df.jobsat.notna(), (df.jobsat >= 8).astype(float), np.nan)

cols = ["country", "exp", "education", "role", "org_size", "manager", "work_mode", "n_languages", "n_databases", "n_platforms"] + \
       ["tool_" + t for t in TOOLS] + ["comp_usd", "jobsat", "high_pay", "satisfied"]
out = df[cols][df.high_pay.notna() | df.satisfied.notna()]
out.to_csv("data/so_success.csv.gz", index=False)
print(len(out), "rows | pay label:", int(out.high_pay.notna().sum()), "| satisfaction label:", int(out.satisfied.notna().sum()),
      "| both:", int((out.high_pay.notna() & out.satisfied.notna()).sum()))
print("countries:", out.country.nunique(), "| high_pay rate", out.high_pay.mean().round(3), "| satisfied rate", out.satisfied.mean().round(3))
print(out.country.value_counts().head(8).to_dict(), "India:", int((out.country == "India").sum()))
