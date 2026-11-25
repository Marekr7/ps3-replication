"""Copies the authors' code to replication_final/build/ and makes two changes there (the package itself is never modified):
 - _0_roots.py: root points to replication_final/build/figroot
 - cleaning: os.listdir() is sorted. Its order depends on the file system and determines the respondent ids, hence which
   simulated theta draws a respondent gets in Figure 1A. Sorting makes every run on every machine identical.
Run from the package root: python replication_final/code/01_prepare.py"""
import os, re, shutil
B = "replication_final/build"
for d in ("precleaned/respondent", "precleaned/resp_quiz_question", "bridge_ids", "figroot/data/cleaned", "figroot/bmodel", "figroot/results/main_sample"):
    os.makedirs(f"{B}/{d}", exist_ok=True)
for src in ("analysis", "cleaning"):
    shutil.rmtree(f"{B}/code_{src}", ignore_errors=True); shutil.copytree(f"code/{src}", f"{B}/code_{src}")
p = f"{B}/code_analysis/_0_roots.py"; s = open(p).read()
s = re.sub(r"^root = .*$", lambda m: "root = " + repr(os.path.abspath(f"{B}/figroot")), s, flags=re.M); open(p, "w").write(s)
p = f"{B}/code_cleaning/gen_analysis_data.py"; s = open(p).read()
assert s.count("os.listdir(") == 3
s = re.sub(r"os\.listdir\(([^)]*)\)", r"sorted(os.listdir(\1))", s); open(p, "w").write(s)
# see README: only bridge_ids3.sav reproduces the published numbers
shutil.copy("data/raw/yougov/bridge_ids/bridge_ids3.sav", f"{B}/bridge_ids/")
