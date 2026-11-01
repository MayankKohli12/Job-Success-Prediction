"""O*NET 31.x tables -> small files for the Career-fit tab.
Usage: python prepare_onet.py   (reads raw_data/onet_*.csv)"""
import re
import numpy as np, pandas as pd

R = "raw_data/onet_"
sk = pd.concat([pd.read_csv(R + "essential_skills.csv"), pd.read_csv(R + "transferable_skills.csv")])
code = "O*NET-SOC Code"
keep = lambda c: c.startswith("15-") or c in ("11-3021.00", "13-1111.00", "13-1161.00", "13-2099.01")
sk = sk[sk[code].map(keep)]
piv = sk.pivot_table(index=[code, "Title", "Element Name"], columns="Scale ID", values="Data Value").reset_index()
piv.columns = ["code", "occupation", "skill", "importance", "level"]
occs = piv.occupation.nunique(); print("occupations:", occs)

# the 14 skills that best separate these occupations (relevant = important on average, informative = varied)
st = piv.groupby("skill").agg(imp=("importance", "mean"), sd=("level", "std"))
chosen = st[st.imp >= 2.6].sort_values("sd", ascending=False).head(14).index.tolist()
print("skills:", chosen)
piv = piv[piv.skill.isin(chosen)]
piv.round(2).to_csv("data/onet_skills.csv", index=False)

# extra context per occupation: strongest knowledge areas, work styles, hot technologies
kn = pd.read_csv(R + "knowledge.csv"); kn = kn[(kn["Scale ID"] == "LV") & kn[code].isin(piv.code)]
top_kn = kn.sort_values("Data Value", ascending=False).groupby(code).head(3).groupby(code)["Element Name"].apply(", ".join)
ws = pd.read_csv(R + "work_styles.csv"); ws = ws[(ws["Scale ID"] == "WI") & ws[code].isin(piv.code)]
top_ws = ws.sort_values("Data Value", ascending=False).groupby(code).head(3).groupby(code)["Element Name"].apply(", ".join)
sw = pd.read_csv(R + "software_skills.csv"); sw = sw[(sw["In Demand"] == "Y") & sw[code].isin(piv.code)]
def _short(x):
    m = re.match(r"^(?:[A-Za-z]+ ){1,4}([A-Z][A-Z0-9#+]{1,6})$", x.strip())       # "Hypertext markup language HTML" -> "HTML"
    return m.group(1) if m else x

def _tools(s):
    names = [("AWS" if "Amazon Web Services" in x else re.sub(r"\s+(software|Software)$", "", re.sub(r"^(Microsoft|Amazon|Apache|Google|IBM|Oracle|Adobe)\s+", "", x))) for x in pd.unique(s)]
    return ", ".join(list(dict.fromkeys(_short(n) for n in names))[:6])
top_sw = sw.groupby(code)["Workplace Example"].apply(_tools)
info = piv[["code", "occupation"]].drop_duplicates().set_index("code")
info["knowledge"], info["work_styles"], info["tools"] = top_kn, top_ws, top_sw
info.fillna("").reset_index().to_csv("data/onet_info.csv", index=False)
print(info.head(4).to_string())
