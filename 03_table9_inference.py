"""
Table 9 (all 8 columns) of Angelucci & Prat (2024) with correctly computed bootstrap inference.

The bootstrap draws are the authors' own (hmodel/<sample>/bootstrap_results.csv, 1,000 respondent-level
resamples per column). Only the step from draws to intervals changes:
  authors : estimate +/- SEM(draws) = estimate +/- SD/sqrt(1000)   -> not a 95% confidence interval
  here    : bootstrap SE = SD(draws); 95% CI = estimate +/- 1.96 SE (percentiles also in the csv; the draws are not centered on the estimate);
            stars from |estimate / SE| (normal approximation).

Run from the package root:  python replication_final/code/<this file> [--sample main_sample]
Outputs: replication_final/output/table9_corrected.tex  (estimates, SE in parentheses, corrected 95% CI in brackets)
         replication_final/output/table9_inference.csv  (how much wider the correct CIs are)
"""
import argparse, json, os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--sample", default="main_sample")
args = parser.parse_args()

boot = pd.read_csv(os.path.join("hmodel", args.sample, "bootstrap_results.csv"))
boot["spec"] = boot["spec"].astype(str)
with open(os.path.join("hmodel", args.sample, "main_estimates.json")) as f:
    main = json.load(f)

order = ["democrat", "republican", "strong_partisan", "news_interest", "age_geq52", "income_geq60k",
         "college+", "female", "white", "sources_3+", "totaltime"]
labels = {"democrat": "Democrat", "republican": "Republican", "strong_partisan": "Strong Partisan",
          "news_interest": "News Interest", "age_geq52": "Age $\\geq$ 52", "income_geq60k": "Inc. $\\geq$ 60k",
          "college+": "College+", "female": "Female", "white": "White", "sources_3+": "Sources 3+",
          "totaltime": "Total Time (hrs)"}
specs = sorted(main.keys(), key=int)

def stars(z):
    return "^{***}" if abs(z) > 2.576 else "^{**}" if abs(z) > 1.96 else "^{*}" if abs(z) > 1.645 else ""

lines, ratios = [], []
for var in order:
    r_est, r_se, r_ci = [labels[var]], [""], [""]
    for s in specs:
        eta = main[s]["est"]["eta"]
        if var not in eta:
            r_est.append(""); r_se.append(""); r_ci.append(""); continue
        d = boot.loc[(boot.brep != "main") & (boot.spec == s), f"eta_{var}"].dropna()
        est, se = eta[var], d.std()
        lo, hi = d.quantile(0.025), d.quantile(0.975)
        r_est.append(f"${est:.3f}{stars(est / se)}$")
        r_se.append(f"({se:.3f})")
        r_ci.append(f"[{est - 1.96 * se:.2f}, {est + 1.96 * se:.2f}]")
        ratios.append({"spec": s, "variable": var, "estimate": est, "boot_se": se, "sem": d.sem(),
                       "paper_lo": est - d.sem(), "paper_hi": est + d.sem(), "ci_lo": est - 1.96 * se, "ci_hi": est + 1.96 * se,
                       "pct_lo": lo, "pct_hi": hi, "boot_mean": d.mean(),
                       "width_ratio_pct_vs_paper": (hi - lo) / (2 * d.sem()),
                       "width_ratio_normal_vs_paper": 1.96 * se / d.sem()})
    lines += [" & ".join(r_est) + r" \\", " & ".join(r_se) + r" \\", r"{\footnotesize " + " & ".join(r_ci) + r"} \\"]

n_row = "Observations & " + " & ".join(f"{int(main[s]['est']['n_respondents']):,}" for s in specs) + r" \\"
emc = "Extra media controls & " + " & ".join("X" if main[s]["extra_media_controls"] else "" for s in specs) + r" \\"
tex = ("\\begin{tabular}{l" + "c" * len(specs) + "}\n\\toprule\n & " +
       " & ".join(f"({s})" for s in specs) + " \\\\\n\\midrule\n" + "\n".join(lines) +
       "\n\\midrule\n" + n_row + "\n" + emc + "\n\\bottomrule\n\\end{tabular}\n")

os.makedirs(os.path.join("replication_final", "output"), exist_ok=True)
with open(os.path.join("replication_final", "output", "table9_corrected.tex"), "w") as f:
    f.write(tex)
rat = pd.DataFrame(ratios)
rat.to_csv(os.path.join("replication_final", "output", "table9_inference.csv"), index=False)
print(tex)
print("Corrected interval / printed interval width:", round(rat.width_ratio_normal_vs_paper.mean(), 2))
print("Not significant at 5 percent:\n", rat[rat.estimate.abs() < 1.96 * rat.boot_se][["spec", "variable", "estimate", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
# Democrat - Republican contrast and how the paper's printed SEs of alpha, delta compare with the bootstrap SD
d5 = boot[(boot.brep != "main") & (boot.spec == "5")]
e5 = main["5"]["est"]["eta"]
d1 = boot[(boot.brep != "main") & (boot.spec == "1")]
chk = pd.Series({"dem_minus_rep_col5": e5["democrat"] - e5["republican"], "se": (d5.eta_democrat - d5.eta_republican).std(),
                 "sd_alpha (paper SE 0.015)": d1.alpha.std(), "sd_delta (paper SE 0.018)": d1.delta.std(),
                 "sd_sigma (paper SE 0.034)": np.exp(main["1"]["est"]["log_sigma_theta"]) * d1.log_sigma_theta.std(),
                 "sigma": np.exp(main["1"]["est"]["log_sigma_theta"]), "boot_mean_age_col1": d1.eta_age_geq52.mean()}, name="value")
chk.to_csv(os.path.join("replication_final", "output", "table9_checks.csv")); print(chk.round(4))
