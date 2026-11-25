"""Figure 1A: authors' stored simulation draws vs our re-estimation. Same formulas as _2_baselineModel.py (part 5).
Run from the package root: python replication_final/code/<this file> --ours replication_final/build/figroot --data <dataset>
Outputs: replication_final/output/figure1A_compare.csv / .png"""
import argparse, os, gc, pickle, sys
sys.path.insert(0, "code/analysis")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser()
ap.add_argument("--ours", default="replication_final/build/figroot"); ap.add_argument("--data", default="data/cleaned/analysis_dataset.csv")
a = ap.parse_args()
COV = ["white", "income_geq60k", "female", "college+", "age_geq52"]
def run(root, data):
    d = pickle.load(open(os.path.join(root, "bmodel", "results_main_sample_z_samples.pkl"), "rb"))
    d = d.loc[d.target_vars.isna()]
    k = ["survey_id", "quiz_id", "respondent_id", "draw"]
    alpha = float(d["alpha"].iloc[0])
    t = d.loc[d.true == 1, k + ["sample_weight", "z"]].rename(columns={"z": "zt"})
    f = d.loc[d.true == 0, k + ["z"]].rename(columns={"z": "zf"})
    del d; gc.collect()
    raw = pd.read_csv(data, low_memory=False)
    raw = raw[(raw.provider == "yougov") & (raw.quiz_type == "incumbent") & raw.snopes.isin([0, 1])]
    # pi averaged over all true/fake pairs; aggregate to respondent x draw level
    m = t.merge(f, on=k); del t, f; gc.collect()
    r = m[["respondent_id", "sample_weight"]].copy()
    for n, sh in (("pi", 0.0),):
        r[n] = m.zt / (m.zt + m.zf)
    q = raw[raw.true == 1].drop_duplicates(["quiz_id", "question_id"]).opinion_rep_joined.quantile([.25, .75]).to_numpy()
    for n, b in (("lo", q[0]), ("hi", q[1])):
        for p in (-1, 1):
            zt = m.zt * np.exp(alpha * b * p); r[f"{n}{p}"] = zt / (zt + m.zf)
    del m; gc.collect()
    r["n"] = 1
    r = r.groupby("respondent_id").agg({"sample_weight": "mean", "n": "sum", **{c: "mean" for c in r.columns if c not in ("respondent_id", "sample_weight", "n")}})
    r = r.join(raw.groupby("respondent_id")[COV + ["democrat", "republican"]].first(), how="left")
    r["w"] = r.sample_weight * r.n
    avg = lambda s, c="pi": float(np.average(s[c], weights=s.w))
    out = {"All": avg(r)}
    L = {"white": ("White", "Nonwhite"), "income_geq60k": ("Inc. ≥ 60k", "Inc. < 60k"), "female": ("Women", "Men"),
         "college+": ("College+", "No college"), "age_geq52": ("Age ≥ 52", "Age < 52")}
    for v, (x, y) in L.items():
        out[x] = avg(r[r[v] == 1]); out[y] = avg(r[r[v] == 0])
    hi = (r.white == 1) & (r.income_geq60k == 1) & (r.female == 0) & (r["college+"] == 1) & (r.age_geq52 == 1)
    lo = (r.white == 0) & (r.income_geq60k == 0) & (r.female == 1) & (r["college+"] == 0) & (r.age_geq52 == 0)
    out["All five: most discerning"] = avg(r[hi]); out["All five: least discerning"] = avg(r[lo])
    D, R = r[r.democrat == 1], r[r.republican == 1]; tw = D.w.sum() + R.w.sum()
    out["Congruent story"] = (avg(D, "lo-1") * D.w.sum() + avg(R, "hi1") * R.w.sum()) / tw
    out["Noncongruent story"] = (avg(D, "hi-1") * D.w.sum() + avg(R, "lo1") * R.w.sum()) / tw
    return pd.Series(out)
paper = {"All": .82, "White": .83, "Nonwhite": .78, "Inc. ≥ 60k": .84, "Inc. < 60k": .80, "Men": .84, "Women": .80, "College+": .85,
         "No college": .80, "Age ≥ 52": .85, "Age < 52": .79, "All five: most discerning": .89, "All five: least discerning": .71,
         "Congruent story": .83, "Noncongruent story": .81}
res = pd.DataFrame({"paper (printed)": pd.Series(paper), "authors' draws": run(".", "data/cleaned/analysis_dataset.csv")}); gc.collect()
res["ours"] = run(a.ours, a.data); res = res.loc[list(paper)]
res.to_csv("replication_final/output/figure1A_compare.csv"); print(res.round(4))
lab = [x for x in paper if x != "All"][::-1]
fig, ax = plt.subplots(figsize=(6.3, 3.0)); y = np.arange(len(lab))
ax.axvline(res.loc["All", "ours"], color="0.5", ls="--", lw=.8)
ax.scatter(res.loc[lab, "paper (printed)"], y, s=42, facecolors="none", edgecolors="black", label="Paper (printed, two decimals)", zorder=3)
ax.scatter(res.loc[lab, "ours"], y, s=14, color="#1f5fa8", label="Our replication", zorder=4)
for i, l in enumerate(lab): ax.text(.915, i, f"{paper[l]:.2f} / {res.loc[l,'ours']:.3f}", fontsize=7, va="center")
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7.5); ax.tick_params(axis="x", labelsize=7.5); ax.set_xlim(.69, .955)
ax.set_xlabel("Predicted probability of selecting the true story", fontsize=8); ax.grid(axis="y", color="0.92"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(.45, 1.0), ncol=2)
fig.tight_layout(); fig.savefig("replication_final/output/figure1A_compare.png", dpi=250)
