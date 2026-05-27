"""
Hyper-parameter tuning for LambdaMART via Optuna.

Optimises NDCG@10 on a validation fold.
Run after feature extraction is done (saves features to disk first).

Usage:
    python tune.py --smoke --n-trials 40
    python tune.py --features-dir features/ --n-trials 100 --n-jobs 4
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import optuna
import lightgbm as lgb

from lambdamart_ranker import (
    generate_synthetic_dataset,
    BM25,
    FeatureEngineer,
    RankingPipeline,
    evaluate_ranking,
    LambdaMARTRanker,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_dataset(n_queries: int = 1000, docs_per_query: int = 100):
    pairs, docs = generate_synthetic_dataset(n_queries, docs_per_query)
    pipeline = RankingPipeline()
    pipeline.bm25.index(docs)
    pipeline.feature_eng = FeatureEngineer(pipeline.bm25)
    X, y, qids = pipeline._extract_features(pairs)
    return X, y, qids


def make_objective(X_tr, y_tr, q_tr, X_vl, y_vl, q_vl):
    grp_tr = LambdaMARTRanker._query_group_sizes(q_tr)
    grp_vl = LambdaMARTRanker._query_group_sizes(q_vl)

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":        "rank:ndcg",
            "metric":           "ndcg",
            "ndcg_eval_at":     [10],
            "label_gain":       [0, 3, 7, 15, 31],
            "num_leaves":       trial.suggest_int("num_leaves", 63, 511),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 500, 3000, step=100),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
            "lambda_l2":        trial.suggest_float("lambda_l2", 0.0, 5.0),
            "lambda_l1":        trial.suggest_float("lambda_l1", 0.0, 1.0),
            "n_jobs":           -1,
            "verbose":          -1,
            "random_state":     42,
        }
        model = lgb.LGBMRanker(**params)
        model.fit(
            X_tr_s, y_tr, group=grp_tr,
            eval_set=[(X_vl_s, y_vl)],
            eval_group=[grp_vl],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(9999),
            ],
        )
        scores = model.predict(X_vl_s)
        res = evaluate_ranking(q_vl, y_vl, scores, k=10)
        return res["ndcg@10"]

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-queries",     type=int, default=1000)
    parser.add_argument("--docs-per-query",type=int, default=100)
    parser.add_argument("--n-trials",      type=int, default=50)
    parser.add_argument("--n-jobs",        type=int, default=1)
    parser.add_argument("--output",        default="best_params.json")
    args = parser.parse_args()

    log.info("Building dataset …")
    X, y, qids = build_dataset(args.n_queries, args.docs_per_query)

    # Simple 80/10/10 group split for tuning
    unique_q = np.unique(qids)
    np.random.default_rng(42).shuffle(unique_q)
    n = len(unique_q)
    tr_q = set(unique_q[:int(0.8*n)])
    vl_q = set(unique_q[int(0.8*n):int(0.9*n)])

    tr = np.array([q in tr_q for q in qids])
    vl = np.array([q in vl_q for q in qids])

    X_tr, y_tr, q_tr = X[tr], y[tr], qids[tr]
    X_vl, y_vl, q_vl = X[vl], y[vl], qids[vl]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(
        make_objective(X_tr, y_tr, q_tr, X_vl, y_vl, q_vl),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    best = study.best_params
    log.info("Best NDCG@10 = %.4f", study.best_value)
    log.info("Best params: %s", best)

    with open(args.output, "w") as f:
        json.dump({"best_ndcg10": study.best_value, "params": best}, f, indent=2)
    log.info("Saved best params to %s", args.output)


if __name__ == "__main__":
    main()
