#!/usr/bin/env python3
"""Loss-Weighting Parity (A1u), Multi-Family Nonlinear Benchmark (A3: PhishKAN, MLP, HistGBDT),
Blend Sensitivity, Latency Profiling, and Nonlinear Structural Residual Probes.

Writes the exp2_*.csv tables and the per-family score parquets scores/exp2_all_families_<corpus>.parquet
that chronological_and_cluster_certification.py reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("THRESHOLD_TRANSPORT_DATA", REPO_ROOT / "data"))

ANCHOR_RUN = DATA_ROOT / "models" / "anchors"
CROSSFIT_RUN = DATA_ROOT / "models" / "crossfit"
DEV_DATA = DATA_ROOT / "processed" / "meajor_trec56_reconstructed.parquet"
SPA_DATA = DATA_ROOT / "processed" / "spaphish_confirmation_urlfixed.parquet"
SPA_PRED = DATA_ROOT / "scores" / "spaphish_frozen_scores.parquet"
CEAS_DATA = DATA_ROOT / "processed" / "ceas08_confirmation.parquet"
CEAS_PRED = DATA_ROOT / "scores" / "ceas08_frozen_scores.parquet"

PRED_DIR = DATA_ROOT / "scores"
RESULTS_DIR = DATA_ROOT / "results"
RUNS_DIR = DATA_ROOT / "results" / "receipts"

SEEDS = (20260503, 20260504, 20260505)
ALPHAS = (0.005, 0.01, 0.02)
BLEND_WEIGHTS = (0.0, 0.15, 0.50, 0.85, 1.0)
COEF_GRID = (0.0, 0.25, 0.5, 1.0)
PROB_EPS = 1e-12
LOGIT_EPS = 1e-6


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_c = np.clip(np.asarray(x, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_c))


def logit(p: np.ndarray) -> np.ndarray:
    p_c = np.clip(np.asarray(p, dtype=np.float64), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p_c / (1.0 - p_c))


def ece15(y: np.ndarray, p: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 16)
    val = 0.0
    for i in range(15):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < 14 else (p <= edges[i + 1]))
        if m.any():
            val += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return val


def threshold_at_max_fpr(y: np.ndarray, scores: np.ndarray, alpha: float = 0.01) -> float:
    neg = np.sort(np.asarray(scores, dtype=np.float64)[np.asarray(y, dtype=int) == 0])[::-1]
    allowed = int(math.floor(alpha * len(neg)))
    if allowed >= len(neg):
        return float(np.min(neg))
    return float(np.nextafter(neg[allowed], np.inf))


def metric_pack(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=np.float64)
    flag = p_arr >= threshold
    tp = int(np.sum(flag & (y_arr == 1)))
    fn = int(np.sum((~flag) & (y_arr == 1)))
    fp = int(np.sum(flag & (y_arr == 0)))
    tn = int(np.sum((~flag) & (y_arr == 0)))
    return {
        "auroc": float(roc_auc_score(y_arr, p_arr)),
        "auprc": float(average_precision_score(y_arr, p_arr)),
        "log_loss": float(log_loss(y_arr, np.clip(p_arr, 1e-7, 1.0 - 1e-7))),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "ece15": ece15(y_arr, p_arr),
        "fpr": fp / max(1, fp + tn),
        "recall": tp / max(1, tp + fn),
        "fp": fp,
        "tp": tp,
        "flagged": int(flag.sum()),
        "threshold": float(threshold),
    }


class CubicBSplineKANLayer(nn.Module):
    """Cubic B-spline Kolmogorov-Arnold layer (Liu et al., 2024) via Cox-de Boor recursion."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3, grid_range: float = 1.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (2.0 * grid_range) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h - grid_range
        self.register_buffer("grid", grid.unsqueeze(0))  # (1, G + 2k + 1)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.bias)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features), mapped cleanly into bounded domain via tanh(x / 2) * 1.5
        x_b = torch.tanh(x * 0.5) * 1.45
        x_u = x_b.unsqueeze(-1)  # (B, in_features, 1)
        grid = self.grid.unsqueeze(0)  # (1, 1, G + 2k + 1)
        bases = ((x_u >= grid[:, :, :-1]) & (x_u < grid[:, :, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            denom1 = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
            denom2 = grid[:, :, k + 1 :] - grid[:, :, 1:-k]
            term1 = ((x_u - grid[:, :, :-(k + 1)]) / denom1) * bases[:, :, :-1]
            term2 = ((grid[:, :, k + 1 :] - x_u) / denom2) * bases[:, :, 1:]
            bases = term1 + term2
        return bases  # (B, in_features, G + k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = torch.nn.functional.silu(x) @ self.base_weight.t()
        bs = self.b_splines(x)  # (B, I, C)
        spline_out = torch.einsum("bic,oic->bo", bs, self.spline_weight)
        return base_out + spline_out + self.bias


class PhishKANClassifier(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 64, grid_size: int = 5) -> None:
        super().__init__()
        self.kan1 = CubicBSplineKANLayer(in_dim, hidden_dim, grid_size=grid_size, spline_order=3)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(0.10)
        self.kan2 = CubicBSplineKANLayer(hidden_dim, 1, grid_size=grid_size, spline_order=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.norm1(self.kan1(x)))
        return self.kan2(h).squeeze(-1)


class MatchedMLPClassifier(nn.Module):
    """Parameter-matched MLP (~148k params matching PhishKAN [256, 64, 1])."""

    def __init__(self, in_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(384, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ProbeKANRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.kan1 = CubicBSplineKANLayer(in_dim, hidden_dim, grid_size=5, spline_order=3)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.kan2 = CubicBSplineKANLayer(hidden_dim, 1, grid_size=5, spline_order=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan2(self.norm1(self.kan1(x))).squeeze(-1)


class ProbeMLPRegressor(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=20000,
        min_df=2,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
    )


def platt_calibrate_logits(val_logits: np.ndarray, val_y: np.ndarray) -> tuple[float, float]:
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    lr.fit(val_logits.reshape(-1, 1), val_y.astype(int))
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    # Guard monotonicity: a must remain positive for strictly increasing probability calibration
    if a <= 0.0:
        return 1.0, 0.0
    return a, b


def train_torch_classifier(
    model: nn.Module,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    seed: int,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> tuple[nn.Module, tuple[float, float], float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ds = TensorDataset(torch.from_numpy(x_fit.astype(np.float32)), torch.from_numpy(y_fit.astype(np.float32)))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.BCEWithLogitsLoss()

    x_cal_t = torch.from_numpy(x_cal.astype(np.float32)).to(device)
    best_loss = float("inf")
    best_state = None
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            cal_logits = model(x_cal_t).detach().cpu().numpy().astype(np.float64)
            cal_p = sigmoid(cal_logits)
            vl = float(log_loss(y_cal, np.clip(cal_p, 1e-7, 1.0 - 1e-7)))
            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        cal_logits = model(x_cal_t).detach().cpu().numpy().astype(np.float64)
    platt_ab = platt_calibrate_logits(cal_logits, y_cal)
    train_sec = time.perf_counter() - t0
    return model, platt_ab, train_sec


def predict_torch_classifier(model: nn.Module, x: np.ndarray, platt_ab: tuple[float, float], batch_size: int = 2048) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size].astype(np.float32)).to(device)
            lg = model(xb).detach().cpu().numpy().astype(np.float64)
            out.append(lg)
    logits = np.concatenate(out, axis=0)
    a, b = platt_ab
    return sigmoid(a * logits + b)


def count_listish(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "[]", "()"}:
        return 0.0
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return float(len(obj))
    except Exception:
        pass
    return float(len([x for x in s.replace(";", ",").split(",") if x.strip()]))


def sender_url_match(sender: object, url_domains: object) -> float:
    sd = "" if sender is None else str(sender).casefold().strip().rstrip(".")
    if not sd:
        return 0.0
    try:
        values = json.loads(str(url_domains))
        if isinstance(values, list):
            domains = {str(x).casefold().strip().rstrip(".") for x in values}
            return float(sd in domains)
    except Exception:
        return 0.0
    return 0.0


def static_feature_blocks(df: pd.DataFrame, block: str) -> tuple[np.ndarray, list[str]]:
    """Base residual-probe blocks (U_7, D_4, M_6, ALL_17) plus the expanded 25-feature set (D_9, M_9, ALL_25)."""
    u_names = [
        "url_indicator",
        "url_count",
        "url_count_parsed",
        "url_length_max",
        "url_length_avg",
        "url_subdom_max",
        "url_subdom_avg",
    ]
    if block == "U_URL_7":
        x = np.column_stack([pd.to_numeric(df[n], errors="coerce").fillna(0.0).to_numpy(np.float64) for n in u_names])
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), u_names

    s = df["sender_domain_norm"].fillna("").astype(str)
    r = df["receiver_domain_norm"].fillna("").astype(str)
    d4_cols = [
        s.str.len().to_numpy(np.float64),
        r.str.len().to_numpy(np.float64),
        ((s != "") & (s == r)).astype(np.float64).to_numpy(),
        np.array([sender_url_match(a, b) for a, b in zip(s, df["url_domains_json"])], dtype=np.float64),
    ]
    d4_names = ["sender_domain_length", "receiver_domain_length", "sender_receiver_domain_equal", "sender_url_domain_match"]
    if block == "D_DOMAIN_4":
        return np.nan_to_num(np.column_stack(d4_cols), nan=0.0, posinf=0.0, neginf=0.0), d4_names

    d9_extra = [
        (s.str.count(r"\.") >= 2).astype(np.float64).to_numpy(),
        (r.str.count(r"\.") >= 2).astype(np.float64).to_numpy(),
        s.str.count("-").to_numpy(np.float64),
        r.str.count("-").to_numpy(np.float64),
        s.str.count(r"\d").to_numpy(np.float64),
    ]
    d9_names = d4_names + [
        "sender_has_subdomain",
        "receiver_has_subdomain",
        "sender_hyphen_count",
        "receiver_hyphen_count",
        "sender_digit_count",
    ]
    if block == "D_DOMAIN_9":
        return np.nan_to_num(np.column_stack(d4_cols + d9_extra), nan=0.0, posinf=0.0, neginf=0.0), d9_names

    tlen = pd.to_numeric(df["text_length_chars"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    toks = pd.to_numeric(df["token_count"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    attc = pd.to_numeric(df["attachment_count"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    hasa = pd.to_numeric(df["has_attachments"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    ctyp = np.array([count_listish(v) for v in df["content_types"]], dtype=np.float64)
    atyp = np.array([count_listish(v) for v in df["attachment_types"]], dtype=np.float64)
    m6_cols = [tlen, toks, attc, hasa, ctyp, atyp]
    m6_names = ["text_length_chars", "token_count", "attachment_count", "has_attachments", "content_type_count", "attachment_type_count"]
    if block == "M_MESSAGE_6":
        return np.nan_to_num(np.column_stack(m6_cols), nan=0.0, posinf=0.0, neginf=0.0), m6_names

    ucnt = pd.to_numeric(df["url_count"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    uparsed = pd.to_numeric(df["url_count_parsed"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    m9_extra = [
        tlen / np.maximum(1.0, toks),
        (1000.0 * ucnt) / np.maximum(1.0, tlen),
        uparsed / np.maximum(1.0, ucnt),
    ]
    m9_names = m6_names + ["chars_per_token", "urls_per_1k_chars", "url_parse_ratio"]
    if block == "M_MESSAGE_9":
        return np.nan_to_num(np.column_stack(m6_cols + m9_extra), nan=0.0, posinf=0.0, neginf=0.0), m9_names

    if block == "ALL_17":
        xu, nu = static_feature_blocks(df, "U_URL_7")
        xd, nd = static_feature_blocks(df, "D_DOMAIN_4")
        xm, nm = static_feature_blocks(df, "M_MESSAGE_6")
        return np.column_stack([xu, xd, xm]), nu + nd + nm

    if block == "ALL_25":
        xu, nu = static_feature_blocks(df, "U_URL_7")
        xd, nd = static_feature_blocks(df, "D_DOMAIN_9")
        xm, nm = static_feature_blocks(df, "M_MESSAGE_9")
        return np.column_stack([xu, xd, xm]), nu + nd + nm

    raise KeyError(f"Unknown block: {block}")


def fit_residual_regressor(probe_family: str, x_tr: np.ndarray, r_tr: np.ndarray, seed: int = 20260503):
    if probe_family == "Ridge":
        reg = Ridge(alpha=1.0)
        reg.fit(x_tr, r_tr)
        return lambda x_eval: reg.predict(x_eval).astype(np.float64)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ProbeKANRegressor(x_tr.shape[1]).to(device) if probe_family == "PhishKAN" else ProbeMLPRegressor(x_tr.shape[1]).to(device)
    ds = TensorDataset(torch.from_numpy(x_tr.astype(np.float32)), torch.from_numpy(r_tr.astype(np.float32)))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=256, shuffle=True, generator=gen)
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(10):
        net.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()

    def _pred(x_eval: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(x_eval), 2048):
                xb = torch.from_numpy(x_eval[i : i + 2048].astype(np.float32)).to(device)
                out.append(net(xb).detach().cpu().numpy().astype(np.float64))
        return np.concatenate(out, axis=0)

    return _pred


def eval_residual_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    p0_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    p0_eval: np.ndarray,
    probe_family: str,
    threshold: float,
    with_offset_control: bool = False,
) -> dict:
    idx_fit, idx_sel = train_test_split(np.arange(len(y_train)), test_size=0.20, stratify=y_train, random_state=20261025)
    sc_inner = StandardScaler().fit(x_train[idx_fit])
    pred_fn_inner = fit_residual_regressor(
        probe_family,
        sc_inner.transform(x_train[idx_fit]),
        y_train[idx_fit] - p0_train[idx_fit],
        seed=20260503,
    )
    r_sel = pred_fn_inner(sc_inner.transform(x_train[idx_sel]))
    cands = []
    for lam in COEF_GRID:
        p_cand = sigmoid(logit(p0_train[idx_sel]) + float(lam) * r_sel)
        cands.append((float(log_loss(y_train[idx_sel], np.clip(p_cand, 1e-7, 1.0 - 1e-7))), float(lam)))
    cands.sort(key=lambda z: (z[0], z[1]))
    chosen_lam = cands[0][1]

    sc_full = StandardScaler().fit(x_train)
    pred_fn_full = fit_residual_regressor(probe_family, sc_full.transform(x_train), y_train - p0_train, seed=20260503)
    r_eval = pred_fn_full(sc_full.transform(x_eval))
    p_fused = sigmoid(logit(p0_eval) + chosen_lam * r_eval)

    base_m = metric_pack(y_eval, p0_eval, threshold)
    fused_m = metric_pack(y_eval, p_fused, threshold)
    delta = {k: float(fused_m[k] - base_m[k]) for k in ("auprc", "log_loss", "brier", "ece15", "fpr", "recall")}
    passes = bool(delta["auprc"] > 0.0 and delta["log_loss"] < 0.0 and delta["fpr"] <= 0.002)

    out = {
        "chosen_lambda": chosen_lam,
        "n_features": int(x_train.shape[1]),
        "base_auprc": float(base_m["auprc"]),
        "fused_auprc": float(fused_m["auprc"]),
        "delta_auprc": delta["auprc"],
        "base_log_loss": float(base_m["log_loss"]),
        "fused_log_loss": float(fused_m["log_loss"]),
        "delta_log_loss": delta["log_loss"],
        "delta_brier": delta["brier"],
        "delta_ece15": delta["ece15"],
        "delta_fpr": delta["fpr"],
        "delta_recall": delta["recall"],
        "passes_single_fold_rule": passes,
    }
    if with_offset_control:
        r_const = float(pred_fn_full(np.zeros((1, x_train.shape[1]), dtype=np.float64))[0])
        p_off = sigmoid(logit(p0_eval) + chosen_lam * r_const)
        off_m = metric_pack(y_eval, p_off, threshold)
        out["offset_only_auprc"] = float(off_m["auprc"])
        out["full_minus_offset_auprc"] = float(fused_m["auprc"] - off_m["auprc"])
        out["beats_offset_control"] = bool(out["full_minus_offset_auprc"] > 0.0)
    ci_pack = paired_stratified_bootstrap_diff(y_eval, p0_eval, p_fused, reps=2000, seed=20261112)
    out["delta_auprc_ci95_lo"] = ci_pack["auprc_ci95"][0]
    out["delta_auprc_ci95_hi"] = ci_pack["auprc_ci95"][1]
    out["delta_log_loss_ci95_lo"] = ci_pack["log_loss_ci95"][0]
    out["delta_log_loss_ci95_hi"] = ci_pack["log_loss_ci95"][1]
    return out


def paired_stratified_bootstrap_diff(
    y: np.ndarray, p_ref: np.ndarray, p_cand: np.ndarray, reps: int = 2000, seed: int = 20261112
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    d_ap, d_ll, d_br = [], [], []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        yi = y[idx]
        pr = p_ref[idx]
        pc = p_cand[idx]
        d_ap.append(float(average_precision_score(yi, pc) - average_precision_score(yi, pr)))
        d_ll.append(float(log_loss(yi, np.clip(pc, 1e-7, 1.0 - 1e-7)) - log_loss(yi, np.clip(pr, 1e-7, 1.0 - 1e-7))))
        d_br.append(float(brier_score_loss(yi, pc) - brier_score_loss(yi, pr)))
    return {
        "auprc_ci95": (float(np.quantile(d_ap, 0.025)), float(np.quantile(d_ap, 0.975))),
        "log_loss_ci95": (float(np.quantile(d_ll, 0.025)), float(np.quantile(d_ll, 0.975))),
        "brier_ci95": (float(np.quantile(d_br, 0.025)), float(np.quantile(d_br, 0.975))),
    }


def holm_bonferroni(p_vals: list[float]) -> list[float]:
    m = len(p_vals)
    order = np.argsort(p_vals)
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p_vals[idx] * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Quick 512-row sanity check of A1u, PhishKAN, MLP, HistGBDT, and 25-feature blocks")
    args = ap.parse_args()

    t0 = time.time()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    dev_full = pd.read_parquet(DEV_DATA)
    v6_ens = pd.read_parquet(ANCHOR_RUN / "fit_trec5__val_trec6" / "A2_MDISTILBERT" / "ensemble_validation.parquet")
    v5_ens = pd.read_parquet(ANCHOR_RUN / "fit_trec6__val_trec5" / "A2_MDISTILBERT" / "ensemble_validation.parquet")
    v6_a1 = pd.read_parquet(ANCHOR_RUN / "fit_trec5__val_trec6" / "A1_TFIDF_LR" / "tfidf_lr_validation.parquet")
    v5_a1 = pd.read_parquet(ANCHOR_RUN / "fit_trec6__val_trec5" / "A1_TFIDF_LR" / "tfidf_lr_validation.parquet")

    trec5_df = v5_ens[["row_id"]].merge(dev_full[dev_full["source"] == "trec5"], on="row_id", validate="one_to_one")
    trec6_df = v6_ens[["row_id"]].merge(dev_full[dev_full["source"] == "trec6"], on="row_id", validate="one_to_one")

    spa_df = pd.read_parquet(SPA_DATA)
    spa_frozen = pd.read_parquet(SPA_PRED)
    ceas_raw = pd.read_parquet(CEAS_DATA)
    ceas_df = ceas_raw.loc[~ceas_raw["contamination_removed"].astype(bool)].reset_index(drop=True)
    ceas_frozen = pd.read_parquet(CEAS_PRED)

    if args.smoke:
        sub5 = trec5_df.iloc[:512].copy()
        sub6 = trec6_df.iloc[:256].copy()
        x25, n25 = static_feature_blocks(sub5, "ALL_25")
        assert x25.shape == (512, 25) and len(n25) == 25 and np.isfinite(x25).all()
        vec = make_tfidf()
        xt = vec.fit_transform(sub5["text_clean"].tolist())
        xv = vec.transform(sub6["text_clean"].tolist())
        svd = TruncatedSVD(n_components=64, random_state=20260503)
        zt = StandardScaler().fit_transform(svd.fit_transform(xt))
        zv = StandardScaler().fit(svd.transform(xt)).transform(svd.transform(xv))
        kan = PhishKANClassifier(in_dim=64, hidden_dim=32)
        mlp = MatchedMLPClassifier(in_dim=64)
        kan_params = sum(p.numel() for p in PhishKANClassifier(256, 64).parameters())
        mlp_params = sum(p.numel() for p in MatchedMLPClassifier(256).parameters())
        m_kan, ab_k, _ = train_torch_classifier(kan, zt[:400], sub5["label"].to_numpy(int)[:400], zt[400:], sub5["label"].to_numpy(int)[400:], 20260503, epochs=2)
        pk = predict_torch_classifier(m_kan, zv, ab_k)
        assert len(pk) == 256 and np.isfinite(pk).all() and (pk >= 0).all() and (pk <= 1).all()
        print(
            json.dumps(
                {
                    "smoke_status": "PASS",
                    "all_25_shape": list(x25.shape),
                    "phishkan_256_params": kan_params,
                    "mlp_256_params": mlp_params,
                    "param_ratio": round(mlp_params / kan_params, 4),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    # Full run
    y_t5 = trec5_df["label"].to_numpy(int)
    y_t6 = trec6_df["label"].to_numpy(int)
    y_spa = spa_df["label"].to_numpy(int)
    y_ceas = ceas_df["label"].to_numpy(int)

    routes = {
        "fit_trec5": (trec5_df, {"trec6": trec6_df, "spaphish": spa_df, "ceas08": ceas_df}),
        "fit_trec6": (trec6_df, {"trec5": trec5_df, "spaphish": spa_df, "ceas08": ceas_df}),
        "final": (pd.concat([trec5_df, trec6_df], ignore_index=True), {"spaphish": spa_df, "ceas08": ceas_df}),
    }

    preds: dict[tuple[str, str, str], np.ndarray] = {}
    # Seed existing A1 and A2 predictions
    preds[("A1", "fit_trec5", "trec6")] = v6_a1["score"].to_numpy(np.float64)
    preds[("A1", "fit_trec6", "trec5")] = v5_a1["score"].to_numpy(np.float64)
    preds[("A2", "fit_trec5", "trec6")] = v6_ens["score"].to_numpy(np.float64)
    preds[("A2", "fit_trec6", "trec5")] = v5_ens["score"].to_numpy(np.float64)
    for rname in ("fit_trec5", "fit_trec6", "final"):
        preds[("A1", rname, "spaphish")] = spa_frozen[f"{rname}_a1"].to_numpy(np.float64)
        preds[("A1", rname, "ceas08")] = ceas_frozen[f"{rname}_a1"].to_numpy(np.float64)
        preds[("A2", rname, "spaphish")] = spa_frozen[f"{rname}_a2"].to_numpy(np.float64)
        preds[("A2", rname, "ceas08")] = ceas_frozen[f"{rname}_a2"].to_numpy(np.float64)

    latency_rows: list[dict] = []
    seed_pred_store: dict[str, dict[str, np.ndarray]] = {c: {} for c in ("trec6", "trec5", "spaphish", "ceas08")}

    for rname, (train_df, eval_map) in routes.items():
        print(f"[multifamily] Training route={rname} (n_train={len(train_df)})...", flush=True)
        y_tr = train_df["label"].to_numpy(int)
        texts_tr = train_df["text_clean"].tolist()

        # 1. TF-IDF + Unweighted Logistic Regression (A1u)
        t_vec0 = time.perf_counter()
        vec = make_tfidf()
        x_tr_tfidf = vec.fit_transform(texts_tr)
        tfidf_fit_sec = time.perf_counter() - t_vec0

        t_a1u0 = time.perf_counter()
        lr_u = LogisticRegression(C=1.0, class_weight=None, solver="liblinear", max_iter=1000)
        lr_u.fit(x_tr_tfidf, y_tr)
        a1u_fit_sec = tfidf_fit_sec + (time.perf_counter() - t_a1u0)

        x_eval_tfidf = {cname: vec.transform(edf["text_clean"].tolist()) for cname, edf in eval_map.items()}
        for cname, xm in x_eval_tfidf.items():
            t_inf = time.perf_counter()
            p_u = lr_u.predict_proba(xm)[:, 1].astype(np.float64)
            inf_sec = time.perf_counter() - t_inf
            preds[("A1u", rname, cname)] = p_u
            seed_pred_store[cname][f"{rname}_A1u"] = p_u
            if rname == "final" and cname == "ceas08":
                # Single-message latency check on 200 samples
                s_times = []
                sample_texts = eval_map[cname]["text_clean"].iloc[:200].tolist()
                for stxt in sample_texts:
                    ts0 = time.perf_counter()
                    _ = lr_u.predict_proba(vec.transform([stxt]))[0, 1]
                    s_times.append((time.perf_counter() - ts0) * 1000.0)
                latency_rows.append(
                    {
                        "model": "A1u_TFIDF_LR_unweighted",
                        "device": "CPU",
                        "route": rname,
                        "train_seconds": round(a1u_fit_sec, 3),
                        "batch_eval_rows": len(p_u),
                        "batch_ms_per_msg": round((inf_sec * 1000.0) / len(p_u), 4),
                        "single_msg_p50_ms": round(float(np.median(s_times)), 4),
                        "single_msg_p95_ms": round(float(np.quantile(s_times, 0.95)), 4),
                        "n_parameters": int(lr_u.coef_.size + lr_u.intercept_.size),
                    }
                )

        # 2. Nonlinear family A3 (TruncatedSVD-256 + PhishKAN, MLP, HistGBDT) across 3 seeds
        fam_seed_Accumulator: dict[tuple[str, str], list[np.ndarray]] = {}
        fam_train_times: dict[str, list[float]] = {"A3a_PhishKAN": [], "A3b_MLP": [], "A3c_HistGBDT": []}

        for seed in SEEDS:
            t_svd0 = time.perf_counter()
            svd = TruncatedSVD(n_components=256, random_state=seed)
            z_tr_raw = svd.fit_transform(x_tr_tfidf)
            scaler = StandardScaler().fit(z_tr_raw)
            z_tr = scaler.transform(z_tr_raw)
            svd_sec = tfidf_fit_sec + (time.perf_counter() - t_svd0)

            z_eval = {cname: scaler.transform(svd.transform(xm)) for cname, xm in x_eval_tfidf.items()}
            idx_fit, idx_cal = train_test_split(np.arange(len(y_tr)), test_size=0.20, stratify=y_tr, random_state=seed)
            zf, yf = z_tr[idx_fit], y_tr[idx_fit]
            zc, yc = z_tr[idx_cal], y_tr[idx_cal]

            # A3a: PhishKAN
            kan_model, kan_ab, kan_sec = train_torch_classifier(
                PhishKANClassifier(in_dim=256, hidden_dim=64, grid_size=5), zf, yf, zc, yc, seed=seed, epochs=12, batch_size=256, lr=1e-3
            )
            fam_train_times["A3a_PhishKAN"].append(svd_sec + kan_sec)

            # A3b: Matched MLP
            mlp_model, mlp_ab, mlp_sec = train_torch_classifier(
                MatchedMLPClassifier(in_dim=256), zf, yf, zc, yc, seed=seed, epochs=12, batch_size=256, lr=1e-3
            )
            fam_train_times["A3b_MLP"].append(svd_sec + mlp_sec)

            # A3c: HistGBDT + inner Platt calibration
            t_gb0 = time.perf_counter()
            gbdt = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            )
            gbdt.fit(zf, yf)
            gb_cal_raw = gbdt.decision_function(zc).astype(np.float64)
            gb_ab = platt_calibrate_logits(gb_cal_raw, yc)
            gb_sec = svd_sec + (time.perf_counter() - t_gb0)
            fam_train_times["A3c_HistGBDT"].append(gb_sec)

            for cname, ze in z_eval.items():
                p_kan = predict_torch_classifier(kan_model, ze, kan_ab)
                p_mlp = predict_torch_classifier(mlp_model, ze, mlp_ab)
                gb_dec = gbdt.decision_function(ze).astype(np.float64)
                p_gb = sigmoid(gb_ab[0] * gb_dec + gb_ab[1])

                seed_pred_store[cname][f"{rname}_A3a_PhishKAN_seed_{seed}"] = p_kan
                seed_pred_store[cname][f"{rname}_A3b_MLP_seed_{seed}"] = p_mlp
                seed_pred_store[cname][f"{rname}_A3c_HistGBDT_seed_{seed}"] = p_gb

                fam_seed_Accumulator.setdefault(("A3a_PhishKAN", cname), []).append(p_kan)
                fam_seed_Accumulator.setdefault(("A3b_MLP", cname), []).append(p_mlp)
                fam_seed_Accumulator.setdefault(("A3c_HistGBDT", cname), []).append(p_gb)

            # Latency profiling on final route, seed 20260503, CEAS-08
            if rname == "final" and seed == SEEDS[0]:
                ze_ceas = z_eval["ceas08"]
                sample_texts = eval_map["ceas08"]["text_clean"].iloc[:200].tolist()
                for m_label, torch_mod, ab_tup in (
                    ("A3a_PhishKAN", kan_model, kan_ab),
                    ("A3b_MLP", mlp_model, mlp_ab),
                ):
                    tb0 = time.perf_counter()
                    _ = predict_torch_classifier(torch_mod, ze_ceas, ab_tup)
                    b_sec = time.perf_counter() - tb0
                    s_times = []
                    for stxt in sample_texts:
                        ts0 = time.perf_counter()
                        zs = scaler.transform(svd.transform(vec.transform([stxt])))
                        _ = predict_torch_classifier(torch_mod, zs, ab_tup, batch_size=1)
                        s_times.append((time.perf_counter() - ts0) * 1000.0)
                    latency_rows.append(
                        {
                            "model": m_label,
                            "device": "CUDA" if torch.cuda.is_available() else "CPU",
                            "route": rname,
                            "train_seconds": round(float(fam_train_times[m_label][0]), 3),
                            "batch_eval_rows": len(ze_ceas),
                            "batch_ms_per_msg": round((b_sec * 1000.0) / len(ze_ceas), 4),
                            "single_msg_p50_ms": round(float(np.median(s_times)), 4),
                            "single_msg_p95_ms": round(float(np.quantile(s_times, 0.95)), 4),
                            "n_parameters": int(sum(p.numel() for p in torch_mod.parameters())),
                        }
                    )
                tb0 = time.perf_counter()
                _ = sigmoid(gb_ab[0] * gbdt.decision_function(ze_ceas) + gb_ab[1])
                b_sec = time.perf_counter() - tb0
                s_times = []
                for stxt in sample_texts:
                    ts0 = time.perf_counter()
                    zs = scaler.transform(svd.transform(vec.transform([stxt])))
                    _ = sigmoid(gb_ab[0] * gbdt.decision_function(zs) + gb_ab[1])
                    s_times.append((time.perf_counter() - ts0) * 1000.0)
                latency_rows.append(
                    {
                        "model": "A3c_HistGBDT",
                        "device": "CPU",
                        "route": rname,
                        "train_seconds": round(float(fam_train_times["A3c_HistGBDT"][0]), 3),
                        "batch_eval_rows": len(ze_ceas),
                        "batch_ms_per_msg": round((b_sec * 1000.0) / len(ze_ceas), 4),
                        "single_msg_p50_ms": round(float(np.median(s_times)), 4),
                        "single_msg_p95_ms": round(float(np.quantile(s_times, 0.95)), 4),
                        "n_parameters": int(300 * 31),
                    }
                )

        for (fam_name, cname), arrs in fam_seed_Accumulator.items():
            ens_p = np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float64)
            preds[(fam_name, rname, cname)] = ens_p
            seed_pred_store[cname][f"{rname}_{fam_name}"] = ens_p

    # Add A1 and A2 latency benchmarks on CEAS-08 final model for complete operational utility table
    import joblib

    # Training times of the A1 and A2 anchors are measured by training/time_anchor_training.py.
    timing_path = RESULTS_DIR / "anchor_training_timing.json"
    if timing_path.exists():
        anchor_timing = json.loads(timing_path.read_text(encoding="utf-8"))
        a1_train_s = float(anchor_timing["A1"]["train_seconds"])
        a2_train_s_seed = float(anchor_timing["A2"]["train_seconds_mean_per_seed"])
        a2_train_s_three = float(anchor_timing["A2"]["train_seconds_three_seeds"])
    else:
        print(f"{timing_path} not found: run training/time_anchor_training.py first; A1/A2 train_seconds are written as NaN", flush=True)
        a1_train_s = a2_train_s_seed = a2_train_s_three = float("nan")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    a1_bundle = joblib.load(ANCHOR_RUN / "final_models" / "A1_TFIDF_LR" / "tfidf_lr.joblib")
    tb0 = time.perf_counter()
    _ = a1_bundle["model"].predict_proba(a1_bundle["vectorizer"].transform(ceas_df["text_clean"].tolist()))[:, 1]
    a1_bsec = time.perf_counter() - tb0
    sample_200 = ceas_df["text_clean"].iloc[:200].tolist()
    a1_st = []
    for stxt in sample_200:
        ts0 = time.perf_counter()
        _ = a1_bundle["model"].predict_proba(a1_bundle["vectorizer"].transform([stxt]))[0, 1]
        a1_st.append((time.perf_counter() - ts0) * 1000.0)
    latency_rows.append(
        {
            "model": "A1_TFIDF_LR_balanced",
            "device": "CPU",
            "route": "final",
            "train_seconds": a1_train_s,
            "batch_eval_rows": len(ceas_df),
            "batch_ms_per_msg": round((a1_bsec * 1000.0) / len(ceas_df), 4),
            "single_msg_p50_ms": round(float(np.median(a1_st)), 4),
            "single_msg_p95_ms": round(float(np.quantile(a1_st, 0.95)), 4),
            "n_parameters": int(a1_bundle["model"].coef_.size + 1),
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a2_dir = ANCHOR_RUN / "final_models" / "A2_MDISTILBERT" / "seed_20260503"
    a2_tok = AutoTokenizer.from_pretrained(a2_dir)
    a2_mod = AutoModelForSequenceClassification.from_pretrained(a2_dir).to(device).eval()
    # Time batch scoring on 2,048 rows (1 seed vs 3-seed ensemble)
    sub_2048 = ceas_df["text_clean"].iloc[:2048].tolist()
    tb0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(sub_2048), 256):
            enc = a2_tok(sub_2048[i : i + 256], truncation=True, padding="max_length", max_length=192, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            _ = torch.softmax(a2_mod(**enc).logits, dim=-1)[:, 1]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    a2_1seed_bsec = time.perf_counter() - tb0
    a2_st = []
    with torch.no_grad():
        for stxt in sample_200[:60]:
            ts0 = time.perf_counter()
            enc = a2_tok([stxt], truncation=True, padding="max_length", max_length=192, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            _ = torch.softmax(a2_mod(**enc).logits, dim=-1)[:, 1]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            a2_st.append((time.perf_counter() - ts0) * 1000.0)
    a2_params = int(sum(p.numel() for p in a2_mod.parameters()))
    del a2_mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    latency_rows.append(
        {
            "model": "A2_mDistilBERT_1seed",
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "route": "final",
            "train_seconds": a2_train_s_seed,
            "batch_eval_rows": 2048,
            "batch_ms_per_msg": round((a2_1seed_bsec * 1000.0) / 2048, 4),
            "single_msg_p50_ms": round(float(np.median(a2_st)), 4),
            "single_msg_p95_ms": round(float(np.quantile(a2_st, 0.95)), 4),
            "n_parameters": a2_params,
        }
    )
    latency_rows.append(
        {
            "model": "A2_mDistilBERT_3seed_ensemble",
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "route": "final",
            "train_seconds": a2_train_s_three,
            "batch_eval_rows": 2048,
            "batch_ms_per_msg": round((3.0 * a2_1seed_bsec * 1000.0) / 2048, 4),
            "single_msg_p50_ms": round(float(3.0 * np.median(a2_st)), 4),
            "single_msg_p95_ms": round(float(3.0 * np.quantile(a2_st, 0.95)), 4),
            "n_parameters": 3 * a2_params,
        }
    )

    # Save frozen score parquets for all new families (A1u, A3a, A3b, A3c) + A1, A2
    corpus_frames = {"trec6": trec6_df, "trec5": trec5_df, "spaphish": spa_df, "ceas08": ceas_df}
    for cname, cdf in corpus_frames.items():
        out_df = pd.DataFrame({"row_id": cdf["row_id"].astype(str), "label": cdf["label"].astype(int)})
        for (m_fam, rname, cc), p_vec in sorted(preds.items()):
            if cc == cname:
                out_df[f"{rname}_{m_fam}"] = p_vec
        for col_k, col_v in sorted(seed_pred_store[cname].items()):
            if col_k not in out_df.columns:
                out_df[col_k] = col_v
        out_df.to_parquet(PRED_DIR / f"exp2_all_families_{cname}.parquet", index=False)

    # 3. Compute locked development cutoffs & full multi-family metric table + prior-shift table across all 6 families
    all_families = ("A1", "A1u", "A2", "A3a_PhishKAN", "A3b_MLP", "A3c_HistGBDT")
    fam_cutoffs: dict[tuple[str, float], dict[str, float]] = {}
    for fam in all_families:
        for alpha in ALPHAS:
            t_fit5 = threshold_at_max_fpr(y_t6, preds[(fam, "fit_trec5", "trec6")], alpha)
            t_fit6 = threshold_at_max_fpr(y_t5, preds[(fam, "fit_trec6", "trec5")], alpha)
            fam_cutoffs[(fam, alpha)] = {"fit_trec5": t_fit5, "fit_trec6": t_fit6, "common": max(t_fit5, t_fit6)}

    # Exhaustive reciprocal feasibility across all 6 families at alpha = 0.01
    from prior_shift_and_seed_ceilings import (
        bbse_priors,
        eval_at_cutoff,
        exhaustive_reciprocal_feasibility,
        p3_rank_map,
        p3b_prior_quantile_cutoff,
        saerens_adjust_scores,
        saerens_em_prior,
    )

    fam_reciprocal_rows = []
    for fam in all_families:
        for alpha in ALPHAS:
            rec = exhaustive_reciprocal_feasibility(
                y_t6, preds[(fam, "fit_trec5", "trec6")], y_t5, preds[(fam, "fit_trec6", "trec5")], alpha
            )
            fam_reciprocal_rows.append({"model": fam, "alpha": alpha, **rec})
    pd.DataFrame(fam_reciprocal_rows).to_csv(RESULTS_DIR / "exp2_multifamily_reciprocal_feasibility.csv", index=False)

    multifamily_rows = []
    for fam in all_families:
        tau_com = fam_cutoffs[(fam, 0.01)]["common"]
        eval_cells = [
            ("trec6", "fit_trec5", y_t6, fam_cutoffs[(fam, 0.01)]["fit_trec5"]),
            ("trec5", "fit_trec6", y_t5, fam_cutoffs[(fam, 0.01)]["fit_trec6"]),
        ]
        for rname in ("final", "fit_trec5", "fit_trec6"):
            own_t = fam_cutoffs[(fam, 0.01)][rname] if rname != "final" else tau_com
            eval_cells.append(("spaphish", rname, y_spa, own_t))
            eval_cells.append(("ceas08", rname, y_ceas, own_t))

        for cname, rname, y_vec, tau_own in eval_cells:
            p_vec = preds[(fam, rname, cname)]
            m_com = metric_pack(y_vec, p_vec, tau_com)
            m_own = metric_pack(y_vec, p_vec, tau_own)
            multifamily_rows.append(
                {
                    "model": fam,
                    "route": rname,
                    "corpus": cname,
                    "n": len(y_vec),
                    "auroc": m_com["auroc"],
                    "auprc": m_com["auprc"],
                    "log_loss": m_com["log_loss"],
                    "brier": m_com["brier"],
                    "ece15": m_com["ece15"],
                    "tau_common_0.01": tau_com,
                    "fpr_common_0.01": m_com["fpr"],
                    "recall_common_0.01": m_com["recall"],
                    "flagged_common_0.01": m_com["flagged"],
                    "tau_own_0.01": tau_own,
                    "fpr_own_0.01": m_own["fpr"],
                    "recall_own_0.01": m_own["recall"],
                }
            )
    df_mf = pd.DataFrame(multifamily_rows)
    df_mf.to_csv(RESULTS_DIR / "exp2_multifamily_metrics.csv", index=False)

    # Paired stratified bootstrap (2,000 reps) & Holm-Bonferroni across primary family contrasts vs A2
    contrast_rows = []
    raw_pvals = []
    n_boot_contrasts = 2000
    for cname, rname, y_vec in (
        ("trec6", "fit_trec5", y_t6),
        ("trec5", "fit_trec6", y_t5),
        ("spaphish", "final", y_spa),
        ("ceas08", "final", y_ceas),
    ):
        rng = np.random.default_rng(20261112)
        pos = np.flatnonzero(y_vec == 1)
        neg = np.flatnonzero(y_vec == 0)
        fam_ap_boot: dict[str, np.ndarray] = {f: np.empty(n_boot_contrasts, dtype=np.float64) for f in all_families}
        fam_ll_boot: dict[str, np.ndarray] = {f: np.empty(n_boot_contrasts, dtype=np.float64) for f in all_families}
        fam_clipped = {f: np.clip(preds[(f, rname, cname)], 1e-7, 1.0 - 1e-7) for f in all_families}
        for b_i in range(n_boot_contrasts):
            idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
            yi = y_vec[idx]
            for f in all_families:
                pf = preds[(f, rname, cname)][idx]
                fam_ap_boot[f][b_i] = float(average_precision_score(yi, pf))
                fam_ll_boot[f][b_i] = float(log_loss(yi, fam_clipped[f][idx]))

        p_a2 = preds[("A2", rname, cname)]
        for other in ("A1", "A1u", "A3a_PhishKAN", "A3b_MLP", "A3c_HistGBDT"):
            p_ot = preds[(other, rname, cname)]
            arr_ap = fam_ap_boot["A2"] - fam_ap_boot[other]
            arr_ll = fam_ll_boot["A2"] - fam_ll_boot[other]
            p_two = float(2.0 * min((arr_ap <= 0).mean(), (arr_ap >= 0).mean()))
            p_two = max(p_two, 1.0 / float(n_boot_contrasts))
            raw_pvals.append(p_two)
            contrast_rows.append(
                {
                    "corpus": cname,
                    "route": rname,
                    "n_boot_reps": n_boot_contrasts,
                    "contrast": f"A2_minus_{other}",
                    "delta_auprc": float(average_precision_score(y_vec, p_a2) - average_precision_score(y_vec, p_ot)),
                    "delta_auprc_ci95_lo": float(np.quantile(arr_ap, 0.025)),
                    "delta_auprc_ci95_hi": float(np.quantile(arr_ap, 0.975)),
                    "delta_log_loss": float(
                        log_loss(y_vec, np.clip(p_a2, 1e-7, 1.0 - 1e-7)) - log_loss(y_vec, np.clip(p_ot, 1e-7, 1.0 - 1e-7))
                    ),
                    "delta_log_loss_ci95_lo": float(np.quantile(arr_ll, 0.025)),
                    "delta_log_loss_ci95_hi": float(np.quantile(arr_ll, 0.975)),
                    "p_boot_raw": p_two,
                }
            )
    holm_p = holm_bonferroni(raw_pvals)
    for row, hp in zip(contrast_rows, holm_p):
        row["p_holm_fwer"] = hp
        row["sig_fwer_0.05"] = bool(hp <= 0.05)
    pd.DataFrame(contrast_rows).to_csv(RESULTS_DIR / "exp2_multifamily_contrasts_holm.csv", index=False)

    # 4. Uniform Prior-Shift evaluation across all 6 families (INT2)
    pi_train = {
        "fit_trec5": float(y_t5.mean()),
        "fit_trec6": float(y_t6.mean()),
        "final": float(np.concatenate([y_t5, y_t6]).mean()),
    }
    mf_prior_rows = []
    for fam in all_families:
        s6_dev = preds[(fam, "fit_trec5", "trec6")]
        s5_dev = preds[(fam, "fit_trec6", "trec5")]
        pooled_s = np.concatenate([s5_dev, s6_dev])
        pooled_y = np.concatenate([y_t5, y_t6])
        pooled_sorted = np.sort(pooled_s)
        tpr_01 = float((pooled_s[pooled_y == 1] >= fam_cutoffs[(fam, 0.01)]["common"]).mean())

        for target, y_tgt in (("spaphish", y_spa), ("ceas08", y_ceas)):
            s_tgt = preds[(fam, "final", target)]
            pi_src = pi_train["final"]
            pi_true = float(y_tgt.mean())
            pi_em, _ = saerens_em_prior(s_tgt, pi_src)
            bb = bbse_priors(pooled_y, pooled_s, s_tgt, cut=0.5)
            tau_com = fam_cutoffs[(fam, 0.01)]["common"]

            pol_map = {
                "P0_COMMON": (s_tgt, tau_com, pi_src),
                "P3_MARGINAL_QUANTILE": (p3_rank_map(s_tgt, pooled_sorted), tau_com, pi_src),
                "P3b_EM_PRIOR_QUANTILE": (s_tgt, p3b_prior_quantile_cutoff(s_tgt, 0.01, tpr_01, pi_em), pi_em),
                "P5_SAERENS_EM": (saerens_adjust_scores(s_tgt, pi_src, pi_em), tau_com, pi_em),
                "P6_BBSE_SOFT": (saerens_adjust_scores(s_tgt, pi_src, bb["pi_bbse_soft"]), tau_com, bb["pi_bbse_soft"]),
                "P6_BBSE_HARD": (saerens_adjust_scores(s_tgt, pi_src, bb["pi_bbse_hard"]), tau_com, bb["pi_bbse_hard"]),
                "P5_ORACLE_PRIOR": (saerens_adjust_scores(s_tgt, pi_src, pi_true), tau_com, pi_true),
            }
            for pol_name, (sc_v, cut_v, pi_est) in pol_map.items():
                ev = eval_at_cutoff(y_tgt, sc_v, cut_v)
                mf_prior_rows.append(
                    {
                        "target": target,
                        "model": fam,
                        "route": "final",
                        "alpha": 0.01,
                        "policy": pol_name,
                        "pi_estimated": pi_est,
                        "cutoff": cut_v,
                        "fpr": ev["fpr"],
                        "fpr_ci_lo": ev["fpr_ci_lo"],
                        "fpr_ci_hi": ev["fpr_ci_hi"],
                        "recall": ev["recall"],
                        "exceeds_0.01": bool(ev["fpr"] > 0.01),
                    }
                )
    pd.DataFrame(mf_prior_rows).to_csv(RESULTS_DIR / "exp2_multifamily_prior_shift.csv", index=False)

    # 5. Blend weight sensitivity w * p_lex + (1 - w) * p_A2 for w in {0.0, 0.15, 0.50, 0.85, 1.0}
    blend_rows = []
    for lex_fam in ("A1", "A1u", "A3a_PhishKAN"):
        for w in BLEND_WEIGHTS:
            b_t6 = w * preds[(lex_fam, "fit_trec5", "trec6")] + (1.0 - w) * preds[("A2", "fit_trec5", "trec6")]
            b_t5 = w * preds[(lex_fam, "fit_trec6", "trec5")] + (1.0 - w) * preds[("A2", "fit_trec6", "trec5")]
            tau_b = max(threshold_at_max_fpr(y_t6, b_t6, 0.01), threshold_at_max_fpr(y_t5, b_t5, 0.01))
            for cname, rname, y_vec, b_vec in (
                ("trec6", "fit_trec5", y_t6, b_t6),
                ("trec5", "fit_trec6", y_t5, b_t5),
                ("spaphish", "final", y_spa, w * preds[(lex_fam, "final", "spaphish")] + (1.0 - w) * preds[("A2", "final", "spaphish")]),
                ("ceas08", "final", y_ceas, w * preds[(lex_fam, "final", "ceas08")] + (1.0 - w) * preds[("A2", "final", "ceas08")]),
            ):
                mp = metric_pack(y_vec, b_vec, tau_b)
                blend_rows.append(
                    {
                        "pair": f"{lex_fam}_plus_A2",
                        "weight_lexical": w,
                        "corpus": cname,
                        "route": rname,
                        "auroc": mp["auroc"],
                        "auprc": mp["auprc"],
                        "log_loss": mp["log_loss"],
                        "brier": mp["brier"],
                        "ece15": mp["ece15"],
                        "tau_common_0.01": tau_b,
                        "fpr_common_0.01": mp["fpr"],
                        "recall_common_0.01": mp["recall"],
                    }
                )
    pd.DataFrame(blend_rows).to_csv(RESULTS_DIR / "exp2_blend_sensitivity.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(RESULTS_DIR / "exp2_latency_cost.csv", index=False)

    # 6. Cross-fitted Linear Ridge, PhishKAN, and MLP Structural Residual Probes
    print("[multifamily] Running cross-fitted Ridge, PhishKAN, and MLP structural residual probes...", flush=True)
    cf_t5 = pd.read_parquet(CROSSFIT_RUN / "crossfit" / "trec5_a2_crossfit.parquet")
    cf_t6 = pd.read_parquet(CROSSFIT_RUN / "crossfit" / "trec6_a2_crossfit.parquet")
    t5_cf_joined = trec5_df.merge(cf_t5[["row_id", "score"]], on="row_id", validate="one_to_one").rename(columns={"score": "p0_oof"})
    t6_cf_joined = trec6_df.merge(cf_t6[["row_id", "score"]], on="row_id", validate="one_to_one").rename(columns={"score": "p0_oof"})

    locked_tau_a2 = fam_cutoffs[("A2", 0.01)]["common"]
    probe_blocks = ("U_URL_7", "D_DOMAIN_4", "D_DOMAIN_9", "M_MESSAGE_6", "M_MESSAGE_9", "ALL_17", "ALL_25")
    probe_families = ("Ridge", "PhishKAN", "MLP")
    residual_rows = []

    for pfam in probe_families:
        for bname in probe_blocks:
            x5, _ = static_feature_blocks(t5_cf_joined, bname)
            x6, _ = static_feature_blocks(t6_cf_joined, bname)

            # Direction 1: Fit TREC5 OOF -> Eval TREC6 outer A2
            res_t6 = eval_residual_probe(
                x5,
                t5_cf_joined["label"].to_numpy(int),
                t5_cf_joined["p0_oof"].to_numpy(np.float64),
                x6,
                y_t6,
                preds[("A2", "fit_trec5", "trec6")],
                pfam,
                locked_tau_a2,
            )
            # Direction 2: Fit TREC6 OOF -> Eval TREC5 outer A2
            res_t5 = eval_residual_probe(
                x6,
                t6_cf_joined["label"].to_numpy(int),
                t6_cf_joined["p0_oof"].to_numpy(np.float64),
                x5,
                y_t5,
                preds[("A2", "fit_trec6", "trec5")],
                pfam,
                locked_tau_a2,
            )
            conj_pass = bool(res_t6["passes_single_fold_rule"] and res_t5["passes_single_fold_rule"])
            residual_rows.append({"scope": "RB_G2_DEV", "probe_family": pfam, "block": bname, "fit_source": "trec5", "eval_corpus": "trec6", "conjunctive_two_source_pass": conj_pass, **res_t6})
            residual_rows.append({"scope": "RB_G2_DEV", "probe_family": pfam, "block": bname, "fit_source": "trec6", "eval_corpus": "trec5", "conjunctive_two_source_pass": conj_pass, **res_t5})

        # External SpaPhish U_URL_7 transfer (corrected URL features from rebuild_spaphish_url_features.py + offset-only control)
        spa_ext_df = pd.DataFrame(
            {
                "url_indicator": spa_df["url_indicator"].astype(float),
                "url_count": spa_df["url_count_source"].astype(float),
                "url_count_parsed": spa_df["url_count_parsed"].astype(float),
                "url_length_max": spa_df["url_length_max"].astype(float),
                "url_length_avg": spa_df["url_length_avg"].astype(float),
                "url_subdom_max": spa_df["url_subdom_max"].astype(float),
                "url_subdom_avg": spa_df["url_subdom_avg"].astype(float),
            }
        )
        x_spa_u, _ = static_feature_blocks(spa_ext_df, "U_URL_7")
        x5_u, _ = static_feature_blocks(t5_cf_joined, "U_URL_7")
        x6_u, _ = static_feature_blocks(t6_cf_joined, "U_URL_7")

        ext_t5 = eval_residual_probe(
            x5_u,
            t5_cf_joined["label"].to_numpy(int),
            t5_cf_joined["p0_oof"].to_numpy(np.float64),
            x_spa_u,
            y_spa,
            preds[("A2", "fit_trec5", "spaphish")],
            pfam,
            locked_tau_a2,
            with_offset_control=True,
        )
        ext_t6 = eval_residual_probe(
            x6_u,
            t6_cf_joined["label"].to_numpy(int),
            t6_cf_joined["p0_oof"].to_numpy(np.float64),
            x_spa_u,
            y_spa,
            preds[("A2", "fit_trec6", "spaphish")],
            pfam,
            locked_tau_a2,
            with_offset_control=True,
        )
        ext_survives = bool(
            ext_t5["passes_single_fold_rule"]
            and ext_t5["beats_offset_control"]
            and ext_t6["passes_single_fold_rule"]
            and ext_t6["beats_offset_control"]
        )
        residual_rows.append({"scope": "EXTERNAL_SPAPHISH_URL", "probe_family": pfam, "block": "U_URL_7", "fit_source": "trec5", "eval_corpus": "spaphish", "conjunctive_two_source_pass": ext_survives, **ext_t5})
        residual_rows.append({"scope": "EXTERNAL_SPAPHISH_URL", "probe_family": pfam, "block": "U_URL_7", "fit_source": "trec6", "eval_corpus": "spaphish", "conjunctive_two_source_pass": ext_survives, **ext_t6})

    df_res = pd.DataFrame(residual_rows)
    df_res.to_csv(RESULTS_DIR / "exp2_nonlinear_residuals.csv", index=False)

    receipt = {
        "schema": "multifamily_training_and_probes_receipt",
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 2),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "summary": {
            "dev_and_final_metrics": df_mf[df_mf["route"].isin(["fit_trec5", "fit_trec6", "final"]) & (
                ((df_mf["corpus"] == "trec6") & (df_mf["route"] == "fit_trec5"))
                | ((df_mf["corpus"] == "trec5") & (df_mf["route"] == "fit_trec6"))
                | ((df_mf["corpus"].isin(["spaphish", "ceas08"])) & (df_mf["route"] == "final"))
            )][["model", "corpus", "route", "auroc", "auprc", "log_loss", "brier", "ece15", "fpr_common_0.01", "recall_common_0.01"]].to_dict("records"),
            "development_conjunctive_passes": df_res[df_res["scope"] == "RB_G2_DEV"][["probe_family", "block", "conjunctive_two_source_pass"]].drop_duplicates().to_dict("records"),
            "spaphish_url_passes": df_res[df_res["scope"] == "EXTERNAL_SPAPHISH_URL"][["probe_family", "fit_source", "delta_auprc", "delta_log_loss", "full_minus_offset_auprc", "conjunctive_two_source_pass"]].to_dict("records"),
        },
        "outputs": {
            "exp2_multifamily_metrics.csv": sha256_file(RESULTS_DIR / "exp2_multifamily_metrics.csv"),
            "exp2_multifamily_reciprocal_feasibility.csv": sha256_file(RESULTS_DIR / "exp2_multifamily_reciprocal_feasibility.csv"),
            "exp2_multifamily_contrasts_holm.csv": sha256_file(RESULTS_DIR / "exp2_multifamily_contrasts_holm.csv"),
            "exp2_multifamily_prior_shift.csv": sha256_file(RESULTS_DIR / "exp2_multifamily_prior_shift.csv"),
            "exp2_blend_sensitivity.csv": sha256_file(RESULTS_DIR / "exp2_blend_sensitivity.csv"),
            "exp2_latency_cost.csv": sha256_file(RESULTS_DIR / "exp2_latency_cost.csv"),
            "exp2_nonlinear_residuals.csv": sha256_file(RESULTS_DIR / "exp2_nonlinear_residuals.csv"),
        },
    }
    (RUNS_DIR / "multifamily_training_and_probes.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
