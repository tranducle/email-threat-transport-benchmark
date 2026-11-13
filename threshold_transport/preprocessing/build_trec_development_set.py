"""Build the TREC5+TREC6 development set (61,325 messages) from the raw MeAJOR table.

Calls ``preprocess_corpora.normalize_dataset("meajor")``, which also writes
processed/meajor_preprocessed.parquet (the table read by the anchor-training scripts), keeps
the trec5 and trec6 sources, checks the expected source counts, and writes
processed/meajor_trec56_reconstructed.parquet. When the cross-fitted anchor scores exist
(models/crossfit/crossfit/<source>_a2_crossfit.parquet), the script also requires that the row
order, row set and labels of each source match them exactly; run it again after
crossfit_anchor_scores.py to perform that check.
"""
from __future__ import annotations
import importlib.util, json, hashlib, os, sys
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve()
REPO_ROOT=HERE.parents[2]
DATA_ROOT=Path(os.environ.get("THRESHOLD_TRANSPORT_DATA", REPO_ROOT/"data"))
PREPROCESS=HERE.parent/"preprocess_corpora.py"
CROSSFIT_RUN=DATA_ROOT/"models"/"crossfit"
OUT=DATA_ROOT/"processed"/"meajor_trec56_reconstructed.parquet"
REC=DATA_ROOT/"results"/"receipts"/"trec_development_set.json"

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

spec=importlib.util.spec_from_file_location("preprocess_corpora",PREPROCESS)
m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m
assert spec.loader is not None; spec.loader.exec_module(m)
df,res=m.normalize_dataset("meajor")
dev=df[df["source"].isin(["trec5","trec6"])].copy().reset_index(drop=True)
counts={str(k):int(v) for k,v in dev["source"].value_counts().sort_index().items()}
labels={s:{str(k):int(v) for k,v in g["label"].value_counts().sort_index().items()} for s,g in dev.groupby("source")}
expected_counts={"trec5":46640,"trec6":14685}
if counts!=expected_counts:
 raise RuntimeError(f"Reconstructed source counts mismatch: {counts} != {expected_counts}")
if len(dev)!=61325:
 raise RuntimeError(f"Expected 61325 dev rows, got {len(dev)}")

parity={}
crossfit_paths={s:CROSSFIT_RUN/"crossfit"/f"{s}_a2_crossfit.parquet" for s in ["trec5","trec6"]}
for source in (["trec5","trec6"] if all(p.exists() for p in crossfit_paths.values()) else []):
 cross=pd.read_parquet(crossfit_paths[source])
 cur=dev[dev["source"].eq(source)][["row_id","label"]].reset_index(drop=True)
 parity[source]={
  "reconstructed_rows":len(cur),
  "crossfit_rows":len(cross),
  "row_order_match":cur["row_id"].astype(str).tolist()==cross["row_id"].astype(str).tolist(),
  "row_set_match":set(cur["row_id"].astype(str))==set(cross["row_id"].astype(str)),
  "label_match_after_row_merge":bool(cur.merge(cross[["row_id","label"]],on="row_id",suffixes=("_new","_old"),validate="one_to_one").eval("label_new == label_old").all())
 }
 if not all([parity[source]["row_order_match"],parity[source]["row_set_match"],parity[source]["label_match_after_row_merge"]]):
  raise RuntimeError(f"Crossfit parity failed for {source}: {parity[source]}")

if not parity:
 parity="not checked: cross-fitted anchor scores not found; rerun after crossfit_anchor_scores.py"
 print(parity)

OUT.parent.mkdir(parents=True,exist_ok=True)
REC.parent.mkdir(parents=True,exist_ok=True)
dev.to_parquet(OUT,index=False)
receipt={
 "schema":"trec_development_set",
 "status":"PASS",
 "preprocess_script":PREPROCESS.relative_to(REPO_ROOT).as_posix(),
 "preprocess_script_sha256":sha(PREPROCESS),
 "raw_meajor_path":"raw/meajor_cleaned_preprocessed.parquet.gzip",
 "raw_meajor_sha256":sha(m.RAW_DIR/"meajor_cleaned_preprocessed.parquet.gzip"),
 "full_output_rows":int(len(df)),
 "preprocess_result":{
  "input_rows":res.input_rows,"output_rows":res.output_rows,
  "dropped_missing_label":res.dropped_missing_label,
  "dropped_empty_text":res.dropped_empty_text,
  "dropped_conflict_rows":res.dropped_conflict_rows,
  "removed_duplicate_rows":res.removed_duplicate_rows
 },
 "dev_rows":int(len(dev)),
 "source_counts":counts,
 "source_label_counts":labels,
 "unique_row_ids":int(dev["row_id"].nunique()),
 "unique_canonical_hashes":int(dev["canonical_hash"].nunique()),
 "crossfit_parity":parity,
 "output_path":OUT.relative_to(DATA_ROOT).as_posix(),
 "output_sha256":sha(OUT)
}
REC.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(receipt,indent=2))
