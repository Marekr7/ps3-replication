"""Writes the LaTeX bodies of Tables 1 and 2 of the report from the script outputs."""
import pandas as pd, json
O = "replication_final/output/"
t = pd.read_csv(O + "table9_inference.csv"); t = t[t.spec == 1].set_index("variable")
e = pd.read_csv(O + "table9_col1_reestimated_rebuilt.csv").set_index("parameter")
import numpy as np
L = [("age_geq52", r"Age $\geq$ 52"), ("income_geq60k", r"Income $\geq$ 60k"), ("college+", "College+"), ("female", "Female"), ("white", "White")]
f = lambda x: f"{x:.3f}" if x >= 0 else f"$-{abs(x):.3f}$"
body = [" & Paper & Ours & Printed ``95\\% CI'' & Corrected 95\\% CI \\\\", "\\midrule"]
for v, l in L:
    r = t.loc[v]
    body.append(f"{l} & {f(r.estimate)} & {f(e.ours['eta_' + v])} & ({f(r.paper_lo)}, {f(r.paper_hi)}) & ({f(r.ci_lo)}, {f(r.ci_hi)}) \\\\")
sg = [f"{np.exp(e[c]['log_sigma_theta']):.2f} / {e[c]['alpha']:.2f} / {e[c]['delta']:.2f}" for c in ("authors", "ours")]
body += ["\\midrule", f"$\\sigma$ / $\\alpha$ / $\\delta$ & {sg[0]} & {sg[1]} & & \\\\",
         f"Respondents & {int(e.authors['N respondents']):,} & {int(e.ours['N respondents']):,} & & \\\\"]
open(O + "report_table1.tex", "w").write("\\begin{tabular}{lcccc}\n\\toprule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")
# Table 2: two-line script output -> coefficient (SE) in one cell, without stars
import re
rows = [x.strip().rstrip("\\").split("&") for x in open(O + "extension_lpm.tex") if "&" in x]
rows = [[c.strip() for c in r] for r in rows if "\\" not in r[0] or "times" in r[0] or "geq" in r[0]]
out, i = [], 1
clean = lambda s: re.sub(r"\^\{\*+\}", "", s).strip("$")
while not rows[i][0].startswith("Observations"):
    b, s = rows[i], rows[i + 1]; i += 2
    lab = b[0].replace("Inc.", "Income").replace("Total Time (10 hrs)", "News time (10 hrs)").replace("Congruence", "Congruence $b_jp_i$").replace("True story", "True story")
    cs = [(f"${clean(x)}$" if clean(x).startswith("-") else clean(x)) + f" {y}" if x else "" for x, y in zip(b[1:], s[1:])]
    out.append(lab + " & " + " & ".join(cs) + " \\\\")
obs = rows[i]
txt = "\\begin{tabular}{lcccc}\n\\toprule\n & (1) Pooled & (2) FE & (3) + Party & (4) + Media \\\\\n\\midrule\n" + "\n".join(out)
txt += "\n\\midrule\nPerson and statement FE & no & yes & yes & yes \\\\\nObservations & " + " & ".join(obs[1:]) + " \\\\\n\\bottomrule\n\\end{tabular}\n"
open(O + "report_table2.tex", "w").write(txt)
print(open(O + "report_table1.tex").read(), txt)

import shutil
for n in ("report_table1.tex", "report_table2.tex", "figure1A_compare.png"):
    shutil.copy(O + n, "replication_final/report/")
