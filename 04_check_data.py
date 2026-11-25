"""Rebuilt data vs the authors' cleaned file: same content? (respondent/question labels depend on file order and are ignored)
Output: replication_final/output/data_check.csv"""
import pandas as pd
IDS = ["respondent_id", "question_id", "statement_id", "quiz_group_id"]
def h(p):
    d = pd.read_csv(p, low_memory=False); c = sorted(x for x in d if x not in IDS)
    return d, pd.util.hash_pandas_object(d[c].round(8).astype(str), index=False).sort_values().to_numpy()
a, ha = h("data/cleaned/analysis_dataset.csv"); b, hb = h("replication_final/build/analysis_dataset_rebuilt.csv")
pat = lambda d: sorted(d[["respondent_id", "provider", "survey_id"]].drop_duplicates().groupby("respondent_id").size())
r = pd.Series({"rows authors": len(a), "rows ours": len(b), "share of identical rows": float((ha == hb).mean()) if len(ha) == len(hb) else 0,
               "same respondent-wave pattern": pat(a) == pat(b)}, name="value")
r.to_csv("replication_final/output/data_check.csv"); print(r)
