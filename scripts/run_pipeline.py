"""End-to-end pipeline: data -> features -> model selection -> regimes -> walk-forward -> backtest.

Examples
--------
    python scripts/run_pipeline.py --source sim
    python scripts/run_pipeline.py --source yahoo --ticker BTC-USD
    python scripts/run_pipeline.py --source csv --csv data/raw/ma_base.csv --name btc_base

Outputs go to results/<name>/ (figures PNG, CSV tables, summary.md).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hmm_crypto import plotting as P
from hmm_crypto.backtest import run_backtest, summary_table
from hmm_crypto.data import fetch_yahoo, load_csv
from hmm_crypto.features import build_features
from hmm_crypto.hmm import GaussianHMM
from hmm_crypto.selection import (Standardizer, choose_k, label_regimes, regime_statistics,
                                  select_n_states)
from hmm_crypto.simulate import simulate_market
from hmm_crypto.strategy import (STRATEGIES, WalkForwardConfig, signal_from_forecast,
                                 walk_forward_forecast)

log = logging.getLogger("pipeline")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["sim", "yahoo", "csv"], default="sim")
    p.add_argument("--ticker", default="BTC-USD")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--csv", type=Path)
    p.add_argument("--name", help="output folder name (default: derived from source)")
    p.add_argument("--features", default="log_ret,vol_20",
                   help="comma-separated HMM observation features")
    p.add_argument("--n-states", default="3", help="integer, or 'auto' to use --select (see README: criteria over-select K)")
    p.add_argument("--k-range", default="2,6", help="min,max K explored by BIC/AIC")
    p.add_argument("--select", choices=["bic", "aic", "cv"], default="cv",
                   help="criterion for K when --n-states auto (cv = hold-out log-likelihood)")
    p.add_argument("--train-window", type=int, default=730)
    p.add_argument("--refit-every", type=int, default=30)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load(args) -> tuple[pd.DataFrame, pd.Series | None, str]:
    if args.source == "sim":
        df, truth = simulate_market(2500, random_state=args.seed)
        return df, truth, args.name or "simulation"
    if args.source == "yahoo":
        return fetch_yahoo(args.ticker, start=args.start), None, args.name or args.ticker
    if not args.csv:
        raise SystemExit("--csv is required with --source csv")
    if not args.csv.exists():
        available = sorted(p.name for p in Path("data/raw").glob("*") if p.is_file())
        raise SystemExit(f"Fichier introuvable : {args.csv}\n"
                         f"Fichiers présents dans data/raw : {available or 'aucun'}")
    return load_csv(args.csv), None, args.name or args.csv.stem


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    df, truth, name = load(args)
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    log.info("Data: %d rows  %s -> %s", len(df), df.index[0].date(), df.index[-1].date())

    feats = build_features(df).dropna()
    cols = [c.strip() for c in args.features.split(",")]
    X_raw = feats[cols].to_numpy()

    # ---- 1. Model selection: on the FIRST training window only (no peeking at the test period)
    sel_slice = slice(0, args.train_window)
    X_sel = Standardizer().fit_transform(X_raw[sel_slice])
    kmin, kmax = map(int, args.k_range.split(","))
    sel = select_n_states(X_sel, range(kmin, kmax + 1), random_state=args.seed)
    sel.to_csv(out / "model_selection.csv")
    P.plot_model_selection(sel, out / "model_selection.png")
    if args.n_states == "auto":
        K, how = choose_k(sel, args.select), f"chosen by {args.select}"
    else:
        K = int(args.n_states)
        how = (f"fixed by --n-states; for reference bic->{choose_k(sel, 'bic')}, "
               f"cv->{choose_k(sel, 'cv')}")
    log.info("Model selection:\n%s\n-> K = %d (%s)", sel.round(3).to_string(), K, how)

    # ---- 2. Ex-post descriptive analysis on the full sample (interpretation only)
    scaler = Standardizer().fit(X_raw)
    X = scaler.transform(X_raw)
    model = GaussianHMM(K, n_init=5, random_state=args.seed).fit(X)
    stats = regime_statistics(model, X, feats["log_ret"].to_numpy())
    labels = label_regimes(stats)
    stats.index = labels
    stats.to_csv(out / "regime_statistics.csv")
    log.info("Regimes (full sample, ex-post):\n%s", stats.round(4).to_string())

    path = pd.Series([labels[s] for s in model.predict(X)], index=feats.index, name="regime")
    path.to_csv(out / "viterbi_regimes.csv")
    P.plot_regimes(df["close"], path, f"{name} — régimes décodés par Viterbi (ex-post)",
                   out / "regimes_viterbi.png")
    P.plot_transition_matrix(model.transmat_, labels, out / "transition_matrix.png")
    P.plot_convergence(model.history_.log_likelihoods, out / "em_convergence.png")

    extra = {}
    if truth is not None:
        # decoding accuracy vs ground truth (best label permutation)
        from scipy.optimize import linear_sum_assignment
        t = truth.loc[feats.index].to_numpy()
        z = model.predict(X)
        C = np.zeros((K, int(t.max()) + 1))
        for a, b in zip(z, t):
            C[a, b] += 1
        r, c = linear_sum_assignment(-C)
        extra["viterbi_accuracy_vs_truth"] = float(C[r, c].sum() / len(t))
        log.info("Viterbi accuracy vs ground truth: %.1f%%", 100 * extra["viterbi_accuracy_vs_truth"])

    # ---- 3. Out-of-sample walk-forward + backtest
    cfg = WalkForwardConfig(feature_cols=tuple(cols), n_states=K, train_window=args.train_window,
                            refit_every=args.refit_every, random_state=args.seed)
    log.info("Walk-forward (%d refits)...", int(np.ceil((len(feats) - cfg.train_window) / cfg.refit_every)))
    fc = walk_forward_forecast(feats, cfg)
    fc.to_csv(out / "walk_forward_forecast.csv")
    names = {"expected_return": "HMM directionnelle", "mean_variance": "HMM moyenne-variance"}
    bts = {}
    for method in STRATEGIES:
        pos = signal_from_forecast(fc, method=method, allow_short=args.allow_short)
        bts[names[method]] = run_backtest(df["close"], pos, cost_bps=args.cost_bps)
        bts[names[method]].to_csv(out / f"backtest_{method}.csv")
    perf = summary_table(bts)
    perf.to_csv(out / "performance.csv")
    equities = {k: v["strategy_equity"] for k, v in bts.items()}
    equities["Buy & Hold"] = next(iter(bts.values()))["buyhold_equity"]
    P.plot_equity(equities, f"{name} — hors-échantillon walk-forward, frais {args.cost_bps:g} bps",
                  out / "equity_curve.png")
    P.plot_regimes(df["close"], fc["regime_now"], f"{name} — régime filtré en temps réel (causal)",
                   out / "regimes_filtered_oos.png")
    log.info("Performance (out-of-sample):\n%s", perf.round(3).to_string())

    # ---- 4. Summary
    (out / "run_config.json").write_text(json.dumps({**vars(args), "csv": str(args.csv), "K": K, **extra},
                                                     indent=2, default=str))
    md = [f"# Résultats — {name}\n",
          f"- Période : {feats.index[0].date()} → {feats.index[-1].date()} ({len(feats)} jours)",
          f"- Features HMM : {', '.join(cols)}",
          f"- K choisi : **{K}** ({args.select if args.n_states == 'auto' else 'fixé'})\n",
          "## Sélection du modèle\n", sel.round(2).to_markdown(),
          "\n## Régimes (ex-post)\n", stats.round(4).to_markdown(),
          "\n## Performance hors-échantillon\n", perf.round(3).to_markdown()]
    if extra:
        md += ["\n## Validation sur simulation\n", json.dumps(extra, indent=2)]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Done -> %s", out.resolve())


if __name__ == "__main__":
    main()
