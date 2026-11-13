"""Rebuild the SpaPhish URL features with the corrected parser and rerun the URL residual test.

Order, each step stops the run on failure:
 1. rebuild the seven URL features from urls_raw with url_parser.url_features, require every
    other column to stay unchanged, write processed/spaphish_confirmation_urlfixed.parquet and
    report parity with url_count_source;
 2. apply the ridge URL probe (scaler and ridge fitted on the cross-fitted development scores of
    crossfit_anchor_scores.py, with its coefficient selection) to the original adapter table,
    whose URL features are all zero;
 3. rerun it on the rebuilt table with the same settings and the same frozen A2 scores, with
    class-stratified bootstrap intervals (seed 20261105, 1,000 replicates);
 4. control: offset-only probe (every URL feature at its development mean), i.e. a pure logit
    shift. The URL block is credited with information only if the full probe passes the residual
    rule and beats the offset-only probe on AUPRC in both routes.
Writes results/spaphish_url_residual_urlfixed.json. Input files are only read.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from url_parser import url_features  # noqa: E402

REPO_ROOT = HERE.parents[1]
DATA_ROOT = Path(os.environ.get("THRESHOLD_TRANSPORT_DATA", REPO_ROOT / "data"))
PROBE_SCRIPT = REPO_ROOT / "threshold_transport" / "training" / "crossfit_anchor_scores.py"
CROSSFIT_RUN = DATA_ROOT / "models" / "crossfit"
CONFIG = REPO_ROOT / "threshold_transport" / "configs" / "residual_probe.json"
DEV = DATA_ROOT / "processed" / "meajor_trec56_reconstructed.parquet"
AD_OLD = DATA_ROOT / "processed" / "spaphish_confirmation.parquet"
PRED = DATA_ROOT / "scores" / "spaphish_frozen_scores.parquet"
AD_FIX = DATA_ROOT / "processed" / "spaphish_confirmation_urlfixed.parquet"
OUT = DATA_ROOT / "results" / "spaphish_url_residual_urlfixed.json"
BOOT_SEED, BOOT_REPS = 20261105, 1000
THRESHOLD = 0.9996199607849122
URL_COLS = ["url_indicator", "url_count_raw", "url_count_parsed", "url_length_max",
            "url_length_avg", "url_subdom_max", "url_subdom_avg"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_probe_module():
    spec = importlib.util.spec_from_file_location("crossfit_anchor_scores", PROBE_SCRIPT)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def ext_frame(ad: pd.DataFrame) -> pd.DataFrame:
    """URL-block feature frame; url_count comes from the source url_count field."""
    return pd.DataFrame({
        "row_id": ad.row_id.astype(str), "label": ad.label.astype(int),
        "url_indicator": ad.url_indicator.astype(float), "url_count": ad.url_count_source.astype(float),
        "url_count_parsed": ad.url_count_parsed.astype(float), "url_length_max": ad.url_length_max.astype(float),
        "url_length_avg": ad.url_length_avg.astype(float), "url_subdom_max": ad.url_subdom_max.astype(float),
        "url_subdom_avg": ad.url_subdom_avg.astype(float)})


def fit_external(probe, train_df, train_scores, ext_df, ext_scores, config):
    """Fit the ridge URL probe on development scores, apply it to the external table, and
    also return the offset-only control."""
    tr = train_df.merge(train_scores[["row_id", "score"]], on="row_id", validate="one_to_one")
    ex = ext_df.merge(ext_scores[["row_id", "score"]], on="row_id", validate="one_to_one")
    if len(tr) != len(train_df) or len(ex) != len(ext_df):
        raise RuntimeError("score join mismatch")
    x, names = probe.feature_matrix(tr, "U_URL_STRUCTURE")
    xv, _ = probe.feature_matrix(ex, "U_URL_STRUCTURE")
    y = tr.label.to_numpy(int); p0 = tr.score.to_numpy(float)
    idx_fit, idx_sel = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                                        random_state=int(config["probe"]["selection_seed"]))
    scaler = StandardScaler().fit(x[idx_fit])
    ridge = Ridge(alpha=float(config["probe"]["ridge_alpha"])).fit(scaler.transform(x[idx_fit]), y[idx_fit] - p0[idx_fit])
    rsel = ridge.predict(scaler.transform(x[idx_sel]))
    cands = []
    for a in config["probe"]["coefficient_grid"]:
        pp = probe.sigmoid(probe.logit(p0[idx_sel]) + float(a) * rsel)
        cands.append((float(log_loss(y[idx_sel], np.clip(pp, 1e-7, 1 - 1e-7))), float(a)))
    cands.sort(key=lambda z: (z[0], z[1]))
    chosen = cands[0][1]
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=float(config["probe"]["ridge_alpha"])).fit(scaler.transform(x), y - p0)
    rext = ridge.predict(scaler.transform(xv))
    base = ex.score.to_numpy(float)
    fused = probe.sigmoid(probe.logit(base) + chosen * rext)
    # offset-only control: every feature at its development mean -> constant r_hat (pure logit shift)
    r_const = float(ridge.predict(scaler.transform(scaler.mean_.reshape(1, -1)))[0])
    offset = probe.sigmoid(probe.logit(base) + chosen * r_const)
    yext = ex.label.to_numpy(int)
    return {"features": names, "coefficient": chosen,
            "coefficient_selection_log_loss": {str(a): l for l, a in cands},
            "baseline_prob": base, "fused_prob": fused, "offset_prob": offset, "label": yext,
            "r_ext_std": float(np.std(rext)), "r_const": r_const,
            "baseline": probe.metric_pack(yext, base, THRESHOLD),
            "fused": probe.metric_pack(yext, fused, THRESHOLD),
            "offset_only": probe.metric_pack(yext, offset, THRESHOLD)}


def bootstrap(probe, y, b, f):
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0); rng = np.random.default_rng(BOOT_SEED)
    keys = ["auprc", "log_loss", "brier", "ece15", "fpr", "recall"]; vals = {k: [] for k in keys}
    for _ in range(BOOT_REPS):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        mb = probe.metric_pack(y[idx], b[idx], THRESHOLD); mf = probe.metric_pack(y[idx], f[idx], THRESHOLD)
        for k in keys: vals[k].append(mf[k] - mb[k])
    return {k: {"mean": float(np.mean(v)), "ci95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))]} for k, v in vals.items()}


def run(probe, ad, config, dev, pred, with_boot):
    out = {}
    for fs in ["trec5", "trec6"]:
        train = dev[dev.source.eq(fs)].copy().reset_index(drop=True)
        ts = pd.read_parquet(CROSSFIT_RUN / "crossfit" / f"{fs}_a2_crossfit.parquet")
        es = pd.DataFrame({"row_id": ad.row_id.astype(str), "score": pred[f"fit_{fs}_a2"].to_numpy(float)})
        g = fit_external(probe, train, ts, ext_frame(ad), es, config)
        d = lambda key: {k: g[key][k] - g["baseline"][k] for k in ["auprc", "log_loss", "brier", "ece15", "fpr", "recall"]}
        r = {"coefficient": g["coefficient"], "r_ext_std": g["r_ext_std"], "r_const": g["r_const"],
             "baseline": g["baseline"], "fused": g["fused"], "offset_only": g["offset_only"],
             "delta_full": d("fused"), "delta_offset_only": d("offset_only")}
        r["full_minus_offset_auprc"] = g["fused"]["auprc"] - g["offset_only"]["auprc"]
        r["passes_gate_full"] = bool(r["delta_full"]["auprc"] > 0 and r["delta_full"]["log_loss"] < 0 and r["delta_full"]["fpr"] <= 0.002)
        if with_boot:
            r["bootstrap_full"] = bootstrap(probe, g["label"], g["baseline_prob"], g["fused_prob"])
            r["bootstrap_offset_only"] = bootstrap(probe, g["label"], g["baseline_prob"], g["offset_prob"])
            r["bootstrap_full_minus_offset"] = bootstrap(probe, g["label"], g["offset_prob"], g["fused_prob"])
        out[fs] = r
    return out


def main():
    probe = load_probe_module()
    config = json.loads(CONFIG.read_text())
    dev = pd.read_parquet(DEV)
    pred = pd.read_parquet(PRED)
    ad_old = pd.read_parquet(AD_OLD)
    if not pred.row_id.equals(ad_old.row_id):
        raise RuntimeError("external prediction alignment mismatch")

    # 1. rebuild URL features
    ad_fix = ad_old.copy()
    u = pd.DataFrame([url_features(x) for x in ad_fix["urls_raw"]])
    for c in URL_COLS:
        ad_fix[c] = u[c].to_numpy()
    for c in ad_old.columns:  # everything except the URL features must be byte-for-byte unchanged
        if c not in URL_COLS and not ad_old[c].equals(ad_fix[c]):
            raise RuntimeError(f"non-URL column changed: {c}")
    diff = (ad_fix.url_count_parsed - ad_fix.url_count_source)
    parity = {"rows": int(len(ad_fix)), "parsed_equals_source_count": int((diff == 0).sum()),
              "parsed_lt_source": int((diff < 0).sum()), "parsed_gt_source": int((diff > 0).sum()),
              "url_indicator_rate_by_label": ad_fix.groupby("label").url_indicator.mean().round(4).to_dict(),
              "nonzero_share_old": {c: float((ad_old[c] != 0).mean()) for c in URL_COLS},
              "nonzero_share_fixed": {c: float((ad_fix[c] != 0).mean()) for c in URL_COLS}}
    AD_FIX.parent.mkdir(exist_ok=True); ad_fix.to_parquet(AD_FIX, index=False)

    # 2. the same probe on the original table (URL features all zero)
    old = run(probe, ad_old, config, dev, pred, with_boot=False)

    # 3 + 4. re-run on the fixed adapter, with the offset-only control
    new = run(probe, ad_fix, config, dev, pred, with_boot=True)
    survives = all(new[fs]["passes_gate_full"] and new[fs]["full_minus_offset_auprc"] > 0 for fs in ["trec5", "trec6"])
    res = {"schema": "spaphish_url_residual_urlfixed",
           "inputs": {p.name: sha(p) for p in [DEV, AD_OLD, PRED, CONFIG, PROBE_SCRIPT]},
           "fixed_adapter_sha256": sha(AD_FIX), "threshold": THRESHOLD,
           "parser_parity": parity,
           "original_adapter_offset_only": {fs: {"delta_offset_only": old[fs]["delta_offset_only"],
                                                 "full_minus_offset_auprc": old[fs]["full_minus_offset_auprc"]} for fs in old},
           "fixed_adapter": new,
           "decision_rule": "URL information credited only if the full probe passes the gate AND beats the offset-only probe on AUPRC in both routes",
           "url_information_survives": bool(survives)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"parity": parity, "url_information_survives": survives,
                      "original_offset_only": res["original_adapter_offset_only"],
                      "fixed": {fs: {k: new[fs][k] for k in ["coefficient", "delta_full", "delta_offset_only",
                                                              "full_minus_offset_auprc", "passes_gate_full"]} for fs in new}}, indent=1))


if __name__ == "__main__":
    main()
