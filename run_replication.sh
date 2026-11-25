#!/bin/bash
# Master script, PS3 replication of Angelucci & Prat (2024). Run from the package root (folder with code/, data/, hmodel/, bmodel/):
#   bash replication_final/run_replication.sh
# Starts from the raw survey data. Writes only to replication_final/build/ and replication_final/output/.
set -euo pipefail
R=replication_final; B=$R/build; D=$B/analysis_dataset_rebuilt.csv
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
mkdir -p $R/output
python $R/code/01_prepare.py
for s in code/cleaning/survey_cleaning_scripts/*.py; do
    python "$s" --resp_dir $B/precleaned/respondent/ --resp_quiz_question_dir $B/precleaned/resp_quiz_question/ > /dev/null
done
python $B/code_cleaning/gen_analysis_data.py --resp $B/precleaned/respondent/ --resp_quiz_quest $B/precleaned/resp_quiz_question/ \
    --yougov_bridgeid $B/bridge_ids/ --output $D
python $R/code/04_check_data.py
python $R/code/02_estimate_table9.py --data $D --tag _rebuilt
python $R/code/03_table9_inference.py
# Sensitivity: bridge_ids1.sav instead of bridge_ids3.sav
mkdir -p $B/bridge_ids_alt && cp data/raw/yougov/bridge_ids/bridge_ids1.sav $B/bridge_ids_alt/
python $B/code_cleaning/gen_analysis_data.py --resp $B/precleaned/respondent/ --resp_quiz_quest $B/precleaned/resp_quiz_question/ \
    --yougov_bridgeid $B/bridge_ids_alt/ --output $B/analysis_dataset_bridge1.csv
python $R/code/02_estimate_table9.py --data $B/analysis_dataset_bridge1.csv --tag _bridge1
cp $D $B/figroot/data/cleaned/analysis_dataset.csv
python $B/code_analysis/_2_baselineModel.py -sn main_sample -pfb "0.7 0.95" -rfb "0.5 0.85" -pl cpu
cp $B/figroot/results/main_sample/group_inequalities_pitrue.jpg $R/output/figure1A_full.jpg
for t in avg_prob_selection odds_table partisanship tier_pitrue_heterogeneity_table; do
    cmp -s $B/figroot/results/main_sample/$t.tex results/main_sample/$t.tex && echo "$t identical" || echo "$t DIFFERENT"
done | tee $R/output/other_tables_check.txt
python $R/code/05_figure1A.py --ours $B/figroot --data $D
python $R/code/06_extension_lpm.py --data $D
python $R/code/07_report_tables.py
echo "Done: $(date)"
