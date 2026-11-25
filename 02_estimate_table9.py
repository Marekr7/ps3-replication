"""
Re-estimate Table 9, column 1 (specification 1) of Angelucci & Prat (2024) from the
cleaned data, using the authors' own model code (code/analysis/_1_1_functions_targeting.py).

This mirrors code/analysis/_3_1_hierarchicalModel.py for the main sample
(YouGov, incumbent quizzes, synthetic + Snopes fake news) but estimates only
specification 1 (theta_i depends on age, income, gender, race, education)
and does NOT overwrite hmodel/main_sample/main_estimates.json.

Run from the package root:
    python replication_final/code/<this file>
Output: replication_final/output/table9_col1_reestimated.csv
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join("code", "analysis"))
import jax
jax.config.update("jax_enable_x64", True)
import pandas as pd
import _1_1_functions_targeting as fct

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=os.path.join("data", "cleaned", "analysis_dataset.csv"),
                    help="analysis dataset to use (default: the authors' cleaned file)")
parser.add_argument("--tag", default="", help="suffix for the output file")
args = parser.parse_args()
data = pd.read_csv(args.data, low_memory=False)

# Main sample, same filters as _3_1_hierarchicalModel.py with default arguments
df = data[(data.quiz_type == "incumbent") & (data.snopes.isin([0, 1])) & (data.provider == "yougov")].copy()
# Both fake-news types are pooled -> the authors use the "joined" partisan-slant variables
df = df.rename(columns={"opinion_rep_joined": "opinion_rep", "cong_opinion_rep_joined": "cong_opinion_rep"})
df["totaltime"] = df["totaltime"] / 25

with open(os.path.join("hmodel", "specifications.json")) as f:
    covariates = json.load(f)["specs"]["1"]["covariates"]

t0 = time.time()
est = fct.run_model(df=df, covariates=covariates, impute=False, target_vars=[], rng_key=707)
print(f"Estimation time: {(time.time() - t0) / 60:.1f} min")

with open(os.path.join("hmodel", "main_sample", "main_estimates.json")) as f:
    paper = json.load(f)["1"]["est"]

rows = [{"parameter": f"eta_{v}", "ours": float(est["params"]["eta"][i]), "authors": paper["eta"][v]}
        for i, v in enumerate(covariates)]
for p in ["delta", "alpha", "log_sigma_theta"]:
    rows.append({"parameter": p, "ours": float(est["params"][p]), "authors": paper[p]})
rows.append({"parameter": "N respondents", "ours": float(est["model"].i), "authors": paper["n_respondents"]})
out = pd.DataFrame(rows)
out["difference"] = out["ours"] - out["authors"]

os.makedirs(os.path.join("replication_final", "output"), exist_ok=True)
out.to_csv(os.path.join("replication_final", "output", f"table9_col1_reestimated{args.tag}.csv"), index=False)
print(out.round(4).to_string(index=False))
