"""
Extension: linear probability model (LPM) on the item-level data, panel structure respondent x statement.

    sel_ij = sum_k b_k (true_j x X_ik) + a * cong_ij + mu_i + lambda_j + e_ij

  sel_ij   = 1 if respondent i selects statement j as true (3 of 6 per quiz)
  true_j   = 1 if statement j is a real news story
  X_ik     = respondent characteristics (as in Table 9)
  cong_ij  = partisan congruence of statement j for i (authors' cong_opinion_rep_joined; 0 for independents)
  mu_i     = respondent fixed effect (absorbs every respondent-level variable, e.g. general propensity to tick)
  lambda_j = statement fixed effect (absorbs difficulty/appeal of each statement, incl. true_j itself)

b_k = how much higher (in pp) the gap P(select true) - P(select fake) is for group k:
the linear analogue of the authors' discernment shifters eta_k in Table 9.
a   = effect of partisan congruence, identified only from within-person and within-statement variation:
      Democrats and Republicans judging the SAME statement differently.

Sample: same as the authors' main sample (YouGov, incumbent quizzes, synthetic + Snopes fake news,
exactly 3 selections per quiz, no missing covariates). SEs clustered by respondent; two-way clustered
(respondent and statement) SEs as a robustness check.

Run from the package root:  python replication_final/code/<this file>
Outputs: replication_final/output/extension_lpm.tex, extension_ovb.csv, extension_robustness.tex, extension_selection.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = os.path.join("replication_final", "output")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ data
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=os.path.join("data", "cleaned", "analysis_dataset.csv"))
args = parser.parse_args()
raw = pd.read_csv(args.data, low_memory=False)
base = raw[(raw.provider == "yougov") & (raw.quiz_type == "incumbent") & (raw.snopes.isin([0, 1]))].copy()
base = base.rename(columns={"cong_opinion_rep_joined": "cong"})
base["totans"] = base.groupby(["respondent_id", "quiz_id", "survey_id"])["sel"].transform("sum")
base["duration_min"] = (base["endtime"] - base["starttime"]) / 60
base["totaltime_h"] = base["totaltime"] / 10   # totaltime is in hours; unit here = 10 hours (Table 9 reports per 1 hour)

DEMO = ["age_geq52", "income_geq60k", "college+", "female", "white"]
PART = ["democrat", "republican", "strong_partisan"]
MEDIA = ["news_interest", "sources_3+", "totaltime_h"]
LABEL = {"age_geq52": "Age $\\geq$ 52", "income_geq60k": "Inc. $\\geq$ 60k", "college+": "College+",
         "female": "Female", "white": "White", "democrat": "Democrat", "republican": "Republican",
         "strong_partisan": "Strong Partisan", "news_interest": "News Interest", "sources_3+": "Sources 3+",
         "totaltime_h": "Total Time (10 hrs)", "cong": "Congruence", "true": "True story"}


def sample(cols, exactly3=True):
    d = base[base.totans == 3] if exactly3 else base.copy()
    need = ["sel", "true", "cong", "statement_id"] + cols
    bad = d[need].isnull().any(axis=1).groupby([d.respondent_id, d.quiz_id]).transform("any")
    return d[~bad].copy()


def demean(df, cols, by):
    return df[cols] - df.groupby(by)[cols].transform("mean")


def absorb(df, cols, fes=("respondent_id", "statement_id"), tol=1e-10, maxit=1000):
    """Two-way within transformation by alternating projections (method of alternating projections,
    as in reghdfe/fixest). Removes respondent and statement means until convergence."""
    z = df[cols].astype(float).copy()
    for _ in range(maxit):
        old = z.copy()
        for fe in fes:
            z = z - z.groupby(df[fe]).transform("mean")
        if (z - old).abs().to_numpy().max() < tol:
            break
    return z


def fit(d, xs, fe=True, twoway=False):
    """OLS of sel on true x X (+cong). fe=True: respondent FE + statement FE absorbed (FWL theorem)."""
    d = d.copy()
    names = []
    for x in xs:
        d[f"T_{x}"] = d["true"] * d[x]
        names.append(f"T_{x}")
    names.append("cong")
    if fe:
        z = absorb(d, names + ["sel"])
        X, y = z[names], z["sel"]
    else:  # pooled OLS with main effects
        X = sm.add_constant(d[names + ["true"] + xs].astype(float))
        y = d["sel"]
    groups = np.column_stack([d.respondent_id.astype("category").cat.codes,
                              d.statement_id.astype("category").cat.codes]) if twoway \
        else d.respondent_id.astype("category").cat.codes
    res = sm.OLS(y.values, X.values).fit(cov_type="cluster", cov_kwds={"groups": groups})
    out = pd.DataFrame({"b": res.params, "se": res.bse}, index=X.columns)
    return out, res, d


def stars(b, se):
    z = abs(b / se)
    return "^{***}" if z > 2.576 else "^{**}" if z > 1.96 else "^{*}" if z > 1.645 else ""


def to_tex(cols, rows, extra):
    """cols: list of (header, coef DataFrame); rows: variable keys in display order."""
    lines = []
    for v in rows:
        key = "cong" if v == "cong" else ("true" if v == "true" else f"T_{v}")
        lab = LABEL[v] if v in ("cong", "true") else f"True $\\times$ {LABEL[v]}"
        e, s = [lab], [""]
        for _, c in cols:
            if key in c.index:
                b, se = c.loc[key, "b"], c.loc[key, "se"]
                e.append(f"${b:.3f}{stars(b, se)}$"); s.append(f"({se:.3f})")
            else:
                e.append(""); s.append("")
        lines += [" & ".join(e) + r" \\", " & ".join(s) + r" \\"]
    head = " & " + " & ".join(h for h, _ in cols) + r" \\"
    body = "\n".join(lines)
    ext = "\n".join(" & ".join(r) + r" \\" for r in extra)
    return ("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n" + head + "\n\\midrule\n" + body +
            "\n\\midrule\n" + ext + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------ main table
d1 = sample(DEMO)
d4 = sample(DEMO + PART + MEDIA)
specs = [
    ("(1)", d1, DEMO, False),
    ("(2)", d1, DEMO, True),
    ("(3)", sample(DEMO + PART), DEMO + PART, True),
    ("(4)", d4, DEMO + PART + MEDIA, True),
]
cols, extra_n, extra_r, extra_fe = [], ["Observations (resp. $\\times$ statement)"], ["Respondents"], ["Respondent \\& statement FE"]
fits = {}
for name, d, xs, fe in specs:
    c, res, dd = fit(d, xs, fe=fe)
    cols.append((name, c)); fits[name] = (c, dd, xs)
    extra_n.append(f"{len(dd):,}"); extra_r.append(f"{dd.respondent_id.nunique():,}")
    extra_fe.append("X" if fe else "")
    print(name, "N =", len(dd), "respondents =", dd.respondent_id.nunique())
mean_gap = d1.groupby("true").sel.mean()
extra_mean = ["Mean P(select) true / fake"] + [f"{mean_gap[1]:.2f} / {mean_gap[0]:.2f}"] * len(cols)
tex = to_tex(cols, ["true"] + DEMO + PART + MEDIA + ["cong"], [extra_n, extra_r, extra_fe, extra_mean])
open(os.path.join(OUT, "extension_lpm.tex"), "w").write(tex)
print(tex)

# ------------------------------------------------------------------ omitted variable bias, exact OLS decomposition
# Same sample for short and long regression (d4). Short: DEMO only. Long: DEMO + PART + MEDIA.
# For each included k:  b_short,k - b_long,k = sum_m  g_m * delta_{m,k}
#   g_m       = coefficient on the omitted regressor m in the long regression
#   delta_m,k = coefficient on included regressor k in an auxiliary regression of omitted regressor m on all included regressors
short, _, _ = fit(d4, DEMO)
long_, _, dl = fit(d4, DEMO + PART + MEDIA)
incd = absorb(dl, [f"T_{x}" for x in DEMO] + ["cong"])
ovb = []
for m in PART + MEDIA:
    om = absorb(dl.assign(om=dl["true"] * dl[m]), ["om"])["om"]
    delta = pd.Series(sm.OLS(om.values, incd.values).fit().params, index=incd.columns)
    for k in DEMO:
        ovb.append({"included": k, "omitted": m, "gamma_long": long_.loc[f"T_{m}", "b"],
                    "delta_aux": delta[f"T_{k}"], "contribution": long_.loc[f"T_{m}", "b"] * delta[f"T_{k}"]})
ovb = pd.DataFrame(ovb)
summ = ovb.groupby("included").contribution.sum().rename("sum_gamma_delta").to_frame()
summ["b_short"] = [short.loc[f"T_{k}", "b"] for k in summ.index]
summ["b_long"] = [long_.loc[f"T_{k}", "b"] for k in summ.index]
summ["check_diff"] = summ.b_short - summ.b_long
ovb.to_csv(os.path.join(OUT, "extension_ovb.csv"), index=False)
summ.to_csv(os.path.join(OUT, "extension_ovb_summary.csv"))
print("\nOVB decomposition (short - long = sum gamma*delta):\n", summ.round(4))
print(ovb.pivot(index="omitted", columns="included", values="contribution").round(4))

# ------------------------------------------------------------------ robustness (spec (2))
rob = []
c_tw, _, _ = fit(d1, DEMO, twoway=True)
rob.append(("Two-way cl.", c_tw, d1))
d_all = sample(DEMO, exactly3=False)            # keep respondents who did not tick exactly 3 (no imputation needed in an LPM)
c_all, _, dd = fit(d_all, DEMO); rob.append(("All resp.", c_all, dd))
dur = d1.groupby("respondent_id").duration_min.min()  # min, not first: independent of row order
fast = dur[dur < dur.quantile(0.10)].index
d_ns = d1[~d1.respondent_id.isin(fast)]
c_ns, _, dd = fit(d_ns, DEMO); rob.append(("No speeders", c_ns, dd))
d_syn = d1[d1.snopes == 0]; c_syn, _, dd = fit(d_syn, DEMO); rob.append(("Synthetic fake", c_syn, dd))
d_sno = d1[d1.snopes == 1]; c_sno, _, dd = fit(d_sno, DEMO); rob.append(("Snopes fake", c_sno, dd))
ext = [["Observations"] + [f"{len(x[2]):,}" for x in rob], ["Respondents"] + [f"{x[2].respondent_id.nunique():,}" for x in rob]]
tex_r = to_tex([(h, c) for h, c, _ in rob], DEMO + ["cong"], ext)
open(os.path.join(OUT, "extension_robustness.tex"), "w").write(tex_r)
print(tex_r)

# ------------------------------------------------------------------ robustness: keep respondents with missing income
NOINC = [x for x in DEMO if x != "income_geq60k"]

# ------------------------------------------------------------------ sample flow and who is dropped
K = ["respondent_id", "quiz_id", "survey_id"]
quiz = base.groupby(K).agg(totans=("totans", "first"), **{x: (x, "first") for x in DEMO + PART + ["news_interest"]}).reset_index()
quiz["kept"] = quiz.totans == 3
resp = quiz.groupby("respondent_id").agg(any_kept=("kept", "max"), income=("income_geq60k", "first"))
flow = pd.Series({"respondents": len(resp), "quizzes": len(quiz), "quizzes with exactly 3": int(quiz.kept.sum()),
                  "quizzes dropped": int((~quiz.kept).sum()), "respondents with >=1 valid quiz": int(resp.any_kept.sum()),
                  "of these: income missing": int((resp.any_kept & resp.income.isna()).sum()),
                  "final respondents": int((resp.any_kept & resp.income.notna()).sum()),
                  "final quizzes": int((quiz.kept & quiz.income_geq60k.notna()).sum())}, name="n")
flow.to_csv(os.path.join(OUT, "sample_flow.csv")); print(flow)
sel_tab = quiz.groupby("kept")[DEMO + PART + ["news_interest"]].mean().T
sel_tab["diff_kept_minus_dropped"] = sel_tab[True] - sel_tab[False]
sel_tab.to_csv(os.path.join(OUT, "extension_selection.csv"))
print("\nSelection by the exactly-3 rule (quiz level):\n", sel_tab.round(3))
c_ni, _, dd = fit(sample(NOINC), NOINC)
c_nis, _, _ = fit(d1, NOINC)
pd.concat({"all incl. missing income": c_ni, "col. (2) sample": c_nis}, axis=1).to_csv(os.path.join(OUT, "extension_noincome.csv"))

# ------------------------------------------------------------------ magnitudes in a common unit
# With exactly 3 of 6 selected, P(select|true) + P(select|fake) = 1 on average, so a gap b_k corresponds to
# b_k/2 in P(select true). Congruence: a x (Q75 - Q25 of b_j), the authors' Figure 1A contrast (from +1 to -1 party).
c2, _, _ = fit(d1, DEMO)
c3, _, d3 = fit(sample(DEMO + PART), DEMO + PART)
bt = d3[d3["true"] == 1].drop_duplicates(["quiz_id", "statement_id"]).opinion_rep_joined  # as in _2_baselineModel.py
iqr = bt.quantile(.75) - bt.quantile(.25)
mag = [{"contrast": k, "d_P_select_true": abs(c2.loc[f"T_{k}", "b"]) / 2, "se": c2.loc[f"T_{k}", "se"] / 2} for k in DEMO]
mag.append({"contrast": "all five", "d_P_select_true": sum(m["d_P_select_true"] for m in mag), "se": np.nan})
mag.append({"contrast": f"congruence col. (3), IQR={iqr:.3f}", "d_P_select_true": c3.loc["cong", "b"] * iqr, "se": c3.loc["cong", "se"] * iqr})
mag = pd.DataFrame(mag); mag.to_csv(os.path.join(OUT, "extension_magnitudes.csv"), index=False)
print(mag.round(4))
d3 = d3.copy()
d3["congT"], d3["congF"] = d3.cong * d3["true"], d3.cong * (1 - d3["true"])
ns = [f"T_{x}" for x in DEMO + PART] + ["congT", "congF"]
for x in DEMO + PART:
    d3[f"T_{x}"] = d3["true"] * d3[x]
z = absorb(d3, ns + ["sel"])
rs = sm.OLS(z["sel"].values, z[ns].values).fit(cov_type="cluster", cov_kwds={"groups": d3.respondent_id.astype("category").cat.codes})
pd.DataFrame({"b": rs.params, "se": rs.bse}, index=ns).loc[["congT", "congF"]].to_csv(os.path.join(OUT, "extension_cong_split.csv"))

# ------------------------------------------------------------------ ratio: all five gaps / congruence (P(select true) units),
# numerator and denominator from the SAME column; 95% Fieller intervals, clustered by respondent or two-way
sg = np.array([1, 1, 1, -1, 1.0])
rows, pair = [], []
for name, xs in (("(2)", DEMO), ("(3)", DEMO + PART)):
    d = sample(xs)
    bt = d[d["true"] == 1].drop_duplicates(["quiz_id", "statement_id"]).opinion_rep_joined
    iqr = bt.quantile(.75) - bt.quantile(.25)
    for tw in (False, True):
        c, res, _ = fit(d, xs, twoway=tw)
        b, V, k = c.b.values, np.asarray(res.cov_params()), len(c) - 1
        g = np.zeros(k + 1); g[:5] = sg / 2
        num, den = g @ b, b[k] * iqr
        vn, vd, cv, z2 = g @ V @ g, V[k, k] * iqr ** 2, (g @ V)[k] * iqr, 1.96 ** 2
        A, B, C = den ** 2 - z2 * vd, -2 * (num * den - z2 * cv), num ** 2 - z2 * vn
        D = (B * B - 4 * A * C) ** .5
        rows.append({"column": name, "cluster": "two-way" if tw else "respondent", "all_five": num, "congruence_IQR": den,
                     "ratio": num / den, "ci_lo": (-B - D) / (2 * A), "ci_hi": (-B + D) / (2 * A)})
        for j, n in enumerate(DEMO):
            h = np.zeros(k + 1); h[j] = sg[j] / 2; h[k] = -iqr
            pair.append({"column": name, "cluster": rows[-1]["cluster"], "characteristic": n, "minus_congruence": h @ b, "se": (h @ V @ h) ** .5})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extension_ratio.csv"), index=False)
pd.DataFrame(pair).to_csv(os.path.join(OUT, "extension_single_vs_congruence.csv"), index=False)
print(pd.DataFrame(rows).round(3)); print(pd.DataFrame(pair).round(4))

# congruence: true vs fake statements, test of equality
d3["congD"] = d3.cong * d3["true"]
ns2 = [f"T_{x}" for x in DEMO + PART] + ["cong", "congD"]
z = absorb(d3, ns2 + ["sel"])
gr = np.column_stack([d3.respondent_id.astype("category").cat.codes, d3.statement_id.astype("category").cat.codes])
eq = []
for lab, g in (("respondent", gr[:, 0]), ("two-way", gr)):
    r = sm.OLS(z["sel"].values, z[ns2].values).fit(cov_type="cluster", cov_kwds={"groups": g})
    eq.append({"cluster": lab, "true_minus_fake": r.params[-1], "se": r.bse[-1]})
pd.DataFrame(eq).to_csv(os.path.join(OUT, "extension_cong_equality.csv"), index=False)
