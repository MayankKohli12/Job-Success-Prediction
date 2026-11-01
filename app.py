import copy, os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import QueryParams
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import charts, market as mk, tools as tl

BASE = Path(__file__).parent
app = FastAPI(title="SkillPulse", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
T = Jinja2Templates(directory=BASE / "templates")
PAGE = 20

env = T.env
env.globals.update(u=mk.url, fmt_skill=mk.fmt_skill, fmt_city=mk.fmt_city, charts=charts, SORTS=mk.SORTS, LEVELS=mk.LEVELS,
                   SKILL_SLUG_REV=mk.SKILL_SLUG_REV, COMPANY_SLUG_REV=mk.COMPANY_SLUG_REV, FAM_SLUG_REV={v: k for k, v in mk.FAM_SLUG.items()},
                   TOTAL=mk.TOTAL, MEDIAN=mk.MEDIAN)
env.filters["lpa"] = lambda v: f"₹{v:,.1f}".replace(".0", "") + " LPA" if v is not None else "n/a"
env.filters["fmt_skill"] = mk.fmt_skill
env.filters["fmt_city"] = mk.fmt_city
env.filters["skill_tokens"] = lambda xs: [(x, mk.fmt_skill(x)) for x in xs]
env.filters["city_tokens"] = lambda xs: [(x, mk.fmt_city(x)) for x in xs]
env.filters["plain_tokens"] = lambda xs: [(x, x) for x in xs]
env.filters["num"] = lambda v: f"{int(v):,}"
env.filters["signed"] = lambda v: f"{'+' if v >= 0 else '−'}{abs(v):.1f}"
env.filters["pct"] = lambda v: f"{v:.0%}"
env.filters["yrs"] = lambda r: f"{int(r.exp_min)}–{int(r.exp_max)} yrs"


def rk(items, href=None, label=None, sub=None):
    """(name, link, bar %, value, note) rows for the rank() macro."""
    items = [(k, v) for k, v in items]
    if not items: return []
    top = max(v for _, v in items) or 1
    return [(label(k) if label else k, href(k) if href else None, v / top * 100, f"{v:,}", sub(k, v) if sub else None) for k, v in items]


def page(request, tpl, /, status=200, **ctx):
    ctx.setdefault("active", "")
    return T.TemplateResponse(request, tpl, ctx, status_code=status)


def not_found(request, what="page"):
    return page(request, "404.html", 404, what=what, active="")


# ---------------------------------------------------------------------------------------- overview
HERO_CITIES = ["bengaluru", "hyderabad", "pune", "chennai", "mumbai", "gurgaon", "noida", "delhi ncr", "ahmedabad", "kolkata", "kochi"]
CITY_OPTS = [mk.fmt_city(c) for c, n in sorted(mk.CITY_COUNT.items(), key=lambda kv: -kv[1])[:60]] + ["Bangalore"]
ROLE_OPTS = ["Data Scientist", "Data Analyst", "Data Engineer", "Machine Learning Engineer", "AI Engineer", "Business Analyst", "Software Engineer", "DevOps Engineer", "QA Engineer", "Python Developer", "Full Stack Developer", "Analytics Consultant"]
HERO_DEFAULT = "role=Data+Scientist&exp=3"


CITY_ALIAS = {"bangalore": "bengaluru", "gurugram": "gurgaon", "delhi": "delhi ncr", "new delhi": "delhi ncr", "ncr": "delhi ncr", "bombay": "mumbai",
              "cochin": "kochi", "calcutta": "kolkata", "madras": "chennai", "trivandrum": "trivandrum", "thiruvananthapuram": "trivandrum"}
_FAM = {f.lower(): f for f in mk.FAMILY_LIST}


def hero_qp(qp):
    """Free-text 'role' and 'city' from the home sentence -> the filters the data layer understands."""
    if "role" not in qp and "city" not in qp:
        return qp
    items = [(k, v) for k, v in qp.multi_items() if k not in ("role", "city", "q", "family")]
    role = (qp.get("role") or "").strip()
    if role:
        fam = _FAM.get(role.lower())
        items.append(("family", fam) if fam else ("q", role))
    else:
        items += [("family", v) for v in qp.getlist("family") if v]
        if qp.get("q"): items.append(("q", qp.get("q")))
    for c in qp.getlist("city"):
        c = c.strip().lower()
        if c: items.append(("city", CITY_ALIAS.get(c, c)))
    return QueryParams(items)


def pulse_data(qp):
    qp = hero_qp(qp)
    f = mk.Filters(qp)
    df = mk.apply(f)
    href = "/jobs" + ("?" + str(qp) if str(qp) else "")
    if df.empty: return {"n": 0, "href": href}
    ins = mk.insights(df, f)
    return {"n": ins["n"], "med": ins["med"], "q1": ins["q1"], "q3": ins["q3"], "companies": ins["companies_n"], "href": href,
            "svg": str(charts.pulse(ins["counts"], ins["bins"], ins["med"], ins["q1"], ins["q3"]))}


@app.get("/api/pulse")
def api_pulse(request: Request):
    return JSONResponse(pulse_data(request.query_params))


@app.get("/")
def overview(request: Request):
    p = mk.posts
    raw = request.query_params if any(k in request.query_params for k in ("family", "role", "city", "exp", "skill")) else QueryParams(HERO_DEFAULT)
    qp = hero_qp(raw)
    hf = mk.Filters(qp)
    pulse = pulse_data(raw)
    hero_role = (raw.get("role") or raw.get("family") or "").strip()
    hero_city = (raw.get("city") or "").strip()
    miss = hero_city and CITY_ALIAS.get(hero_city.lower(), hero_city.lower()) not in mk.CITY_COUNT
    fam = p.groupby("family").agg(n=("jobId", "size"), med=("sal_mid", "median")).reset_index()
    fam = fam[fam.family != mk.OTHER].sort_values("med", ascending=False)
    fmax = float(fam.med.max())
    fam_rows = [(r.family, "/roles/" + mk.slugify(r.family), r.med / fmax * 100, f"{r.med:.1f} LPA", f"{r.n:,} postings") for r in fam.itertuples()]
    sk = mk.skill_table(p, min_n=40, sort="demand", limit=8)
    skill_rows = [(r["label"], "/skills/" + r["slug"], r["n"] / sk[0]["n"] * 100, f"{r['n']:,}", f"{r['share']:.0%} of postings") for r in sk]
    pay = mk.skill_table(p, min_n=40, sort="premium", limit=6)
    pmax = max(r["premium"] for r in pay)
    pay_rows = [(r["label"], "/skills/" + r["slug"], max(r["premium"], 0) / pmax * 100, f"+{r['premium']:.1f} LPA", f"{r['n']:,} postings") for r in pay]
    cities = mk.top_counts(p.cities, 7, generic=True)
    city_rows = [(mk.fmt_city(c), "/jobs?city=" + c.replace(" ", "+"), n / cities[0][1] * 100, f"{n:,}", f"{n / mk.TOTAL:.0%}") for c, n in cities]
    emp = p.companyName.value_counts().head(6)
    emp_rows = [(c, "/companies/" + mk.COMPANY_SLUG_REV[c], n / emp.iloc[0] * 100, f"{n:,}", "postings") for c, n in emp.items()]
    return page(request, "overview.html", active="overview", fam_rows=fam_rows, skill_rows=skill_rows, pay_rows=pay_rows, city_rows=city_rows,
                emp_rows=emp_rows, n_companies=p.companyName.nunique(), n_skills=len(mk.SKILL_COUNT), n_cities=len(mk.CITY_COUNT),
                pulse=pulse, hf=hf, hero_families=[x for x in mk.FAMILY_LIST if x != mk.OTHER], hero_cities=HERO_CITIES, role_opts=ROLE_OPTS, city_opts_all=CITY_OPTS, hero_role=hero_role, hero_city=hero_city, hero_city_miss=miss,
                hero_skills=[(s, mk.fmt_skill(s)) for s in hf.lists["skill"]], hero_exp=("%g" % hf.exp) if hf.exp is not None else "",
                roles_strip=[(r.family, r.med, r.n) for r in fam.itertuples()])


# ---------------------------------------------------------------------------------------- jobs
def _facet(f, key, col):
    f2 = copy.copy(f); f2.lists = {**f.lists, key: []}
    return mk.apply(f2)[col].value_counts()


@app.get("/jobs")
def jobs(request: Request):
    f = mk.Filters(hero_qp(request.query_params))
    df = mk.apply(f)
    ins = mk.insights(df, f)
    total_pages = max(1, -(-len(df) // PAGE)); pg = min(f.page, total_pages)
    rows = df.iloc[(pg - 1) * PAGE: pg * PAGE]
    fam_c, lvl_c = _facet(f, "family", "family"), _facet(f, "level", "level")
    f2 = copy.copy(f); f2.lists = {**f.lists, "city": []}
    city_c = [(c, n) for c, n in mk.top_counts(mk.apply(f2).cities, 8, generic=True)]
    for c in f.lists["city"]:
        if c not in dict(city_c): city_c.append((c, 0))
    ctx = dict(f=f, P=f.raw(), df_n=len(df), ins=ins, rows=rows.itertuples(), pg=pg, total_pages=total_pages, fam_c=fam_c, lvl_c=lvl_c, city_c=city_c,
               families=mk.FAMILY_LIST, active="jobs", start=(pg - 1) * PAGE + 1, end=min(pg * PAGE, len(df)))
    return page(request, "jobs.html", **ctx)


@app.get("/jobs/{job_id}")
def job_detail(request: Request, job_id: str):
    if job_id not in mk.posts.index: return not_found(request, "posting")
    j = mk.posts.loc[job_id]
    same = mk.posts[(mk.posts.family == j.family) & (mk.posts.exp_mid.sub(j.exp_mid).abs() <= 1.5)]
    q1, med, q3 = mk.quart(same.sal_mid)
    counts, bins = mk.hist(same.sal_mid.values)
    similar = mk.posts[(mk.posts.family == j.family) & (mk.posts.index != job_id) & mk.posts.skillset.map(lambda s: len(s & j.skillset) >= 2)].copy()
    similar["_o"] = similar.skillset.map(lambda s: len(s & j.skillset))
    similar = similar.sort_values(["_o", "sal_mid"], ascending=False).head(5)
    return page(request, "job_detail.html", active="jobs", j=j, med=med, q1=q1, q3=q3, same_n=len(same), counts=counts, bins=bins, similar=similar.itertuples(),
                co_slug=mk.COMPANY_SLUG_REV.get(j.companyName), fam_slug={v: k for k, v in mk.FAM_SLUG.items()}.get(j.family))


# ---------------------------------------------------------------------------------------- companies
@app.get("/companies")
def companies(request: Request):
    qp = request.query_params
    f = mk.Filters(qp)
    f.q = ""; name_q = (qp.get("name") or "").strip()
    sub = mk.apply(f)
    min_n = int(qp.get("min") or 1) if (qp.get("min") or "1").isdigit() else 1
    sort = qp.get("sort") if qp.get("sort") in ("postings", "pay", "top", "exp", "az") else "postings"
    rows, total = mk.company_table(sub, name_q, min_n, sort)
    P = f.raw()
    if name_q: P["name"] = [name_q]
    if min_n > 1: P["min"] = [str(min_n)]
    if sort != "postings": P["sort"] = [sort]
    P.pop("company", None)
    fam_c = _facet(f, "family", "family")
    return page(request, "companies.html", active="companies", rows=rows, total=total, f=f, P=P, name_q=name_q, min_n=min_n, sort=sort, fam_c=fam_c,
                families=mk.FAMILY_LIST, shown=len(rows), sub_n=len(sub))


@app.get("/companies/{slug}")
def company_detail(request: Request, slug: str):
    name = mk.COMPANY_SLUG.get(slug)
    if not name: return not_found(request, "company")
    d = mk.posts[mk.posts.companyName == name]
    q1, med, q3 = mk.quart(d.sal_mid)
    counts, bins = mk.hist(d.sal_mid.values)
    ins = mk.insights(d)
    return page(request, "company_detail.html", active="companies", name=name, n=len(d), q1=q1, med=med, q3=q3, counts=counts, bins=bins,
                rows=d.sort_values("sal_mid", ascending=False).head(8).itertuples(), agency=bool(d.agency.iloc[0]), exp_med=float(d.exp_mid.median()),
                skill_rows=rk(mk.top_counts(d.skills, 8), lambda k: "/skills/" + mk.SKILL_SLUG_REV[k] if k in mk.SKILL_SLUG_REV else None, mk.fmt_skill),
                fam_rows=rk(d.family.value_counts().items(), lambda k: "/roles/" + mk.slugify(k)),
                city_rows=rk(mk.top_counts(d.cities, 6, generic=True), None, mk.fmt_city), name_q=name.replace(" ", "+").replace("&", "%26"))


# ---------------------------------------------------------------------------------------- skills
@app.get("/skills")
def skills(request: Request):
    qp = request.query_params
    f = mk.Filters(qp); f.lists["skill"] = []; f.lists["xskill"] = []; f.lists["company"] = []; f.q = ""
    sub = mk.apply(f)
    name_q = (qp.get("name") or "").strip()
    min_n = int(qp.get("min")) if (qp.get("min") or "").isdigit() else 10
    sort = qp.get("sort") if qp.get("sort") in ("demand", "pay", "premium", "az") else "demand"
    rows = mk.skill_table(sub, name_q, min_n, sort)
    P = f.raw()
    if name_q: P["name"] = [name_q]
    if min_n != 10: P["min"] = [str(min_n)]
    if sort != "demand": P["sort"] = [sort]
    vmax = max([r["n"] for r in rows] + [1])
    pmax = max([abs(r["premium"]) for r in rows] + [1])
    return page(request, "skills.html", active="skills", rows=rows, f=f, P=P, name_q=name_q, min_n=min_n, sort=sort, families=mk.FAMILY_LIST,
                fam_c=_facet(f, "family", "family"), vmax=vmax, pmax=pmax, sub_n=len(sub), base=float(sub.sal_mid.median()) if len(sub) else 0)


@app.get("/skills/{slug}")
def skill_detail(request: Request, slug: str):
    sk = mk.SKILL_SLUG.get(slug)
    if not sk: return not_found(request, "skill")
    d = mk.posts[mk.posts.skillset.map(lambda s: sk in s)]
    q1, med, q3 = mk.quart(d.sal_mid)
    counts, bins = mk.hist(d.sal_mid.values)
    co, _ = mk.co_skills(mk.posts, sk, 12)
    no = mk.posts[~mk.posts.skillset.map(lambda s: sk in s)]
    return page(request, "skill_detail.html", active="skills", sk=sk, n=len(d), q1=q1, med=med, q3=q3, premium=med - float(no.sal_mid.median()),
                counts=counts, bins=bins, bands=mk.exp_bands(d), share=len(d) / mk.TOTAL,
                co_rows=rk([(k, v) for k, v, _ in co], lambda k: "/skills/" + mk.SKILL_SLUG_REV[k], mk.fmt_skill, lambda k, v: f"{v / max(len(d), 1):.0%} of its postings"),
                fam_rows=rk(d.family.value_counts().head(6).items(), lambda k: "/roles/" + mk.slugify(k)),
                co_rows2=rk(d.companyName.value_counts().head(8).items(), lambda k: "/companies/" + mk.COMPANY_SLUG_REV[k]),
                city_rows=rk(mk.top_counts(d.cities, 6, generic=True), lambda k: "/jobs?city=" + k.replace(" ", "+") + "&skill=" + sk.replace(" ", "+"), mk.fmt_city))


# ---------------------------------------------------------------------------------------- roles
@app.get("/roles")
def roles(request: Request):
    p = mk.posts
    rows = []
    for fam in mk.FAMILY_LIST:
        d = p[p.family == fam]
        if d.empty: continue
        q1, med, q3 = mk.quart(d.sal_mid)
        rows.append(dict(name=fam, slug=mk.slugify(fam), n=len(d), companies=d.companyName.nunique(), q1=q1, med=med, q3=q3, exp=float(d.exp_mid.median()),
                         skills=[mk.fmt_skill(s) for s, _ in mk.top_counts(d.skills, 4)]))
    rows.sort(key=lambda r: -r["med"])
    return page(request, "roles.html", active="roles", rows=rows, vmax=max(r["q3"] for r in rows))


@app.get("/roles/{slug}")
def role_detail(request: Request, slug: str):
    fam = mk.FAM_SLUG.get(slug)
    if not fam: return not_found(request, "role")
    d = mk.posts[mk.posts.family == fam]
    q1, med, q3 = mk.quart(d.sal_mid)
    counts, bins = mk.hist(d.sal_mid.values)
    sk = mk.skill_table(d, min_n=8, sort="demand", limit=10)
    hire = mk.hiring[mk.hiring.category == fam].sort_values("postings", ascending=False).head(10) if fam in mk.cat_stats.index else None
    cs = mk.cat_stats.loc[fam] if fam in mk.cat_stats.index else None
    lv = d.level.value_counts()
    return page(request, "role_detail.html", active="roles", fam=fam, n=len(d), q1=q1, med=med, q3=q3, counts=counts, bins=bins, bands=mk.exp_bands(d),
                skill_rows=[(r["label"], "/skills/" + r["slug"], r["n"] / sk[0]["n"] * 100, f"{r['n']:,}", f"{r['premium']:+.1f} LPA vs median") for r in sk],
                co_rows=rk(d.companyName.value_counts().head(8).items(), lambda k: "/companies/" + mk.COMPANY_SLUG_REV[k]),
                city_rows=rk(mk.top_counts(d.cities, 6, generic=True), lambda k: "/jobs?city=" + k.replace(" ", "+"), mk.fmt_city),
                lvl_rows=rk([(l, int(lv.get(l, 0))) for l in mk.LEVELS if lv.get(l, 0)]),
                hire=None if hire is None else list(hire.itertuples()), cs=cs, n_comp=d.companyName.nunique(), exp_med=float(d.exp_mid.median()), fam_q=fam.replace(" ", "+").replace("/", "%2F"))


# ---------------------------------------------------------------------------------------- tools
def _num(v, default, lo, hi):
    try: return min(max(float(v), lo), hi)
    except (TypeError, ValueError): return default


@app.get("/tools/salary")
def tool_salary(request: Request):
    qp = request.query_params
    has = any(k in qp for k in ("title", "skill", "city", "emin", "emax"))
    title = (qp.get("title") or "").strip() if has else "Data Scientist"
    emin = _num(qp.get("emin"), 3, 0, 30); emax = max(_num(qp.get("emax"), 6, 0, 35), emin)
    skills = list(dict.fromkeys(s for s in qp.getlist("skill") if s)) if has else ["python", "machine learning", "sql"]
    city = (qp.get("city") or "").strip() if has else "Bengaluru"
    res = tl.salary_estimate(title, skills, [city], emin, emax)
    popular = {k for k, _ in tl.SALARY_POPULAR}
    extra = [(k, mk.fmt_skill(k)) for k in skills if k not in popular]
    return page(request, "tool_salary.html", active="salary", v=dict(title=title, city=city, emin=emin, emax=emax, skills=skills),
                res=res, roles=tl.SALARY_ROLES, popular=tl.SALARY_POPULAR, extra=extra, city_opts=tl.SALARY_CITIES,
                title_opts=ROLE_OPTS, city_labels=CITY_OPTS)


TOOL_GROUPS = [("Languages", ["Python", "SQL", "R", "Java", "JavaScript", "TypeScript", "C#", "Go", "Rust"]),
               ("Databases", ["PostgreSQL", "MySQL", "MongoDB"]),
               ("Cloud and DevOps", ["AWS", "Azure", "Google Cloud", "Docker", "Kubernetes"])]
TOOL_PRESETS = [("Data science", ["Python", "SQL", "R"]), ("Web full-stack", ["JavaScript", "TypeScript", "PostgreSQL", "Docker"]),
                ("Cloud native", ["AWS", "Docker", "Kubernetes", "Go"]), ("Enterprise", ["Java", "C#", "SQL", "Azure"])]


@app.get("/tools/success")
def tool_success(request: Request):
    qp, C = request.query_params, tl.SO_CATS
    pick = lambda k, cat, d: qp.get(k) if qp.get(k) in C[cat] else (d if d in C[cat] else C[cat][0])
    has = "country" in qp
    v = dict(country=pick("country", "country", "India"), exp=_num(qp.get("exp"), 3, 0, 40), role=pick("role", "role", "Data scientist"),
             edu=pick("edu", "education", "Bachelor's degree"), org=pick("org", "org_size", "1,000 - 4,999"), mgr=pick("mgr", "manager", "Individual contributor"),
             mode=pick("mode", "work_mode", "Hybrid"), tools=[t for t in dict.fromkeys(qp.getlist("tool")) if t in tl.SO_TOOLS] if has else ["Python", "SQL"],
             extra=[s for s in dict.fromkeys(qp.getlist("xt")) if s in mk.SKILL_COUNT][:12])
    res = tl.success_odds(v["country"], v["exp"], v["role"], v["edu"], v["org"], v["mgr"], v["mode"], v["tools"])
    P = {"country": [v["country"]], "exp": ["%g" % v["exp"]], "role": [v["role"]], "edu": [v["edu"]], "org": [v["org"]], "mgr": [v["mgr"]], "mode": [v["mode"]],
         "tool": v["tools"], "xt": v["extra"]}
    tips = [(t, d, mk.url("/tools/success", P, add={"tool": t})) for t, d in res["tips"]]
    ex = []
    for s in v["extra"]:
        has_s = mk.posts.skillset.map(lambda x: s in x)
        d = mk.posts[has_s]
        if len(d): ex.append(dict(label=mk.fmt_skill(s), n=len(d), med=float(d.sal_mid.median()), prem=float(d.sal_mid.median() - mk.posts[~has_s].sal_mid.median())))
    return page(request, "tool_success.html", active="success", v=v, res=res, C=C, groups=[(g, [t for t in ts if t in tl.SO_TOOLS]) for g, ts in TOOL_GROUPS],
                presets=[(n, ",".join(ts)) for n, ts in TOOL_PRESETS], tips=tips, ex=ex, extra_tokens=[(s, mk.fmt_skill(s)) for s in v["extra"]])


@app.get("/tools/career")
def tool_career(request: Request):
    import json as _json
    qp = request.query_params
    idx = {s: i for i, s in enumerate(tl.SKILLS)}
    ratings = [int(_num(qp.get(f"r{i}"), tl.DEFAULT_RATINGS.get(s, 2), 0, 5)) for i, s in enumerate(tl.SKILLS)]
    res = tl.career_fit(ratings) if sum(ratings) > 0 else None
    groups = [(g, [(idx[s], s, tl.SK[s], ratings[idx[s]]) for s in ss if s in idx]) for g, ss in tl.CAREER_GROUPS]
    presets = [(n, _json.dumps([p.get(s, 1) for s in tl.SKILLS])) for n, p in tl.CAREER_PRESETS.items()]
    rank_rows = [(n, None, f * 100, f"{f:.0%}", None) for n, f in res["ranked"]] if res else []
    return page(request, "tool_career.html", active="career", groups=groups, res=res, rank_rows=rank_rows, presets=presets, empty=res is None)


# ---------------------------------------------------------------------------------------- misc
@app.get("/about")
def about(request: Request):
    return page(request, "about.html", active="about", S=tl.S, R=tl.R, SU=tl.SU, so_n=tl.SU["results"]["n"], onet_n=len(tl.occ_names))


@app.get("/api/suggest/{kind}")
def api_suggest(kind: str, q: str = ""):
    if kind == "mskill":
        ql = q.lower().strip()
        hits = [(k, l) for k, l in tl.SALARY_SKILLS if ql in k]
        hits.sort(key=lambda kv: (not kv[0].startswith(ql), -mk.SKILL_COUNT.get(kv[0], 0)))
        return JSONResponse([{"value": k, "label": l, "n": mk.SKILL_COUNT.get(k, 0)} for k, l in hits[:12]])
    if kind not in ("skill", "city", "company"): return JSONResponse([], status_code=404)
    return JSONResponse(mk.suggest(kind, q))


@app.exception_handler(404)
async def _404(request: Request, exc):
    return not_found(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
