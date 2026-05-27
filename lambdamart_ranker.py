"""
LambdaMART Learning-to-Rank System
Targets ≥18% NDCG improvement over BM25 baseline on 2M query-document pairs.

Architecture:
  - BM25 baseline scorer (Okapi BM25)
  - Feature engineering pipeline (36 features)
  - LambdaMART model (LightGBM backend)
  - Offline NDCG@10 evaluation harness
  - NDCG delta reporting vs BM25
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class QueryDocPair:
    query_id: str
    doc_id: str
    query_text: str
    doc_text: str
    title: str = ""
    relevance: int = 0          # 0-4 graded relevance label
    bm25_score: float = 0.0
    features: np.ndarray = field(default_factory=lambda: np.zeros(36))


@dataclass
class RankingResult:
    query_id: str
    ranked_doc_ids: list[str]
    scores: list[float]
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0


# ─────────────────────────────────────────────────────────────
# BM25 Baseline
# ─────────────────────────────────────────────────────────────

class BM25:
    """Okapi BM25 with Robertson IDF.

    Parameters follow the standard TREC configuration:
        k1=1.2, b=0.75, delta=0 (plain BM25).
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, delta: float = 0.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self._idf: dict[str, float] = {}
        self._doc_len: dict[str, int] = {}
        self._avg_dl: float = 0.0
        self._term_freq: dict[str, dict[str, int]] = defaultdict(dict)  # term → {doc_id → tf}
        self._doc_count: int = 0

    # ── index construction ──────────────────────────────────

    def index(self, docs: dict[str, str]) -> None:
        """Build index from {doc_id: text} mapping."""
        log.info("Building BM25 index over %d documents …", len(docs))
        t0 = time.perf_counter()

        df: dict[str, int] = defaultdict(int)
        total_len = 0

        for doc_id, text in docs.items():
            tokens = self._tokenize(text)
            self._doc_len[doc_id] = len(tokens)
            total_len += len(tokens)
            seen: set[str] = set()
            for tok in tokens:
                self._term_freq[tok][doc_id] = self._term_freq[tok].get(doc_id, 0) + 1
                if tok not in seen:
                    df[tok] += 1
                    seen.add(tok)

        self._doc_count = len(docs)
        self._avg_dl = total_len / max(1, self._doc_count)

        N = self._doc_count
        for term, freq in df.items():
            self._idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

        log.info("Index built in %.1fs", time.perf_counter() - t0)

    # ── scoring ─────────────────────────────────────────────

    def score(self, query: str, doc_id: str) -> float:
        tokens = self._tokenize(query)
        dl = self._doc_len.get(doc_id, 0)
        norm = 1 - self.b + self.b * dl / max(1, self._avg_dl)
        score = 0.0
        for tok in set(tokens):
            tf = self._term_freq.get(tok, {}).get(doc_id, 0)
            idf = self._idf.get(tok, 0.0)
            tf_norm = ((self.k1 + 1) * tf) / (self.k1 * norm + tf) + self.delta
            score += idf * tf_norm
        return score

    def score_batch(self, query: str, doc_ids: list[str]) -> np.ndarray:
        tokens = list(set(self._tokenize(query)))
        scores = np.zeros(len(doc_ids))
        for i, doc_id in enumerate(doc_ids):
            dl = self._doc_len.get(doc_id, 0)
            norm = 1 - self.b + self.b * dl / max(1, self._avg_dl)
            s = 0.0
            for tok in tokens:
                tf = self._term_freq.get(tok, {}).get(doc_id, 0)
                idf = self._idf.get(tok, 0.0)
                tf_norm = ((self.k1 + 1) * tf) / (self.k1 * norm + tf) + self.delta
                s += idf * tf_norm
            scores[i] = s
        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()


# ─────────────────────────────────────────────────────────────
# Feature Engineering — 36 features
# ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # ── BM25 family (6) ─────────────────────────────────────
    "bm25_body",          # BM25 on full body
    "bm25_title",         # BM25 on title field
    "bm25_url",           # BM25 on URL tokens
    "bm25_anchor",        # BM25 on anchor text (if available)
    "bm25_body_b0",       # BM25 body, b=0 (TF-IDF-like)
    "bm25_body_k2",       # BM25 body, k1=2.0

    # ── Query-document term overlap (6) ─────────────────────
    "idf_sum",            # sum of query term IDFs
    "qtf_coverage",       # fraction of query terms in doc
    "qtf_exact_phrase",   # phrase match indicator
    "query_len",          # number of query tokens
    "doc_len_log",        # log(doc length)
    "query_doc_ratio",    # query_len / doc_len

    # ── TF statistics (6) ──────────────────────────────────
    "tf_mean",            # mean TF of query terms in doc
    "tf_max",             # max TF
    "tf_min",             # min TF
    "tf_sum",             # sum TF
    "tf_var",             # variance of TF
    "tf_idf_sum",         # sum of TF-IDF scores

    # ── Document quality signals (6) ────────────────────────
    "doc_pagerank",       # PageRank-like static quality
    "doc_click_rate",     # historical CTR
    "doc_dwell_time",     # avg dwell time (seconds, log-scaled)
    "doc_freshness",      # recency score (decay of timestamp)
    "doc_spam_score",     # spam classifier score (inverted)
    "doc_authority",      # domain authority score

    # ── Query intent features (6) ───────────────────────────
    "query_clarity",      # KL divergence from collection LM
    "is_navigational",    # navigational query indicator
    "is_informational",   # informational query indicator
    "query_freq",         # log(query frequency in session logs)
    "avg_result_clicks",  # avg clicks on results for this query
    "reformulation_rate", # query reformulation indicator

    # ── Semantic similarity (6) ─────────────────────────────
    "cosine_tfidf",       # cosine similarity in TF-IDF space
    "bm25_rank_pct",      # percentile BM25 rank within query
    "lm_dir_score",       # language model score (Dirichlet)
    "axiomatic_f2exp",    # F2Exp axiomatic retrieval score
    "overlap_bigram",     # bigram overlap ratio
    "overlap_trigram",    # trigram overlap ratio
]

assert len(FEATURE_NAMES) == 36, f"Expected 36 features, got {len(FEATURE_NAMES)}"


class FeatureEngineer:
    """Extract 36 ranking features for each query-document pair."""

    def __init__(self, bm25_body: BM25, bm25_title: BM25 | None = None, mu: float = 2000.0):
        self.bm25_body = bm25_body
        self.bm25_title = bm25_title or bm25_body
        self.mu = mu  # Dirichlet smoothing parameter
        self._bm25_k2 = BM25(k1=2.0, b=0.75)
        self._bm25_b0 = BM25(k1=1.2, b=0.0)
        # Shallow copies — share underlying index
        self._bm25_k2._idf = bm25_body._idf
        self._bm25_k2._doc_len = bm25_body._doc_len
        self._bm25_k2._avg_dl = bm25_body._avg_dl
        self._bm25_k2._term_freq = bm25_body._term_freq
        self._bm25_k2._doc_count = bm25_body._doc_count
        self._bm25_b0._idf = bm25_body._idf
        self._bm25_b0._doc_len = bm25_body._doc_len
        self._bm25_b0._avg_dl = bm25_body._avg_dl
        self._bm25_b0._term_freq = bm25_body._term_freq
        self._bm25_b0._doc_count = bm25_body._doc_count

    def extract(self, pair: QueryDocPair) -> np.ndarray:
        q_tokens = BM25._tokenize(pair.query_text)
        d_tokens = BM25._tokenize(pair.doc_text)
        t_tokens = BM25._tokenize(pair.title)
        q_set = set(q_tokens)
        d_set = set(d_tokens)

        # ── BM25 family ─────────────────────────────────────
        bm25_body  = self.bm25_body.score(pair.query_text, pair.doc_id)
        bm25_title = self.bm25_title.score(pair.query_text, pair.doc_id)
        bm25_url   = 0.0                                          # placeholder
        bm25_anchor = 0.0                                         # placeholder
        bm25_b0    = self._bm25_b0.score(pair.query_text, pair.doc_id)
        bm25_k2    = self._bm25_k2.score(pair.query_text, pair.doc_id)

        # ── Term overlap ────────────────────────────────────
        idf_sum = sum(self.bm25_body._idf.get(t, 0.0) for t in q_tokens)
        covered = q_set & d_set
        qtf_coverage = len(covered) / max(1, len(q_set))
        qtf_exact_phrase = 1.0 if " ".join(q_tokens) in " ".join(d_tokens) else 0.0
        query_len = len(q_tokens)
        doc_len_log = math.log1p(len(d_tokens))
        query_doc_ratio = len(q_tokens) / max(1, len(d_tokens))

        # ── TF stats ────────────────────────────────────────
        tfs = [
            self.bm25_body._term_freq.get(tok, {}).get(pair.doc_id, 0)
            for tok in q_tokens
        ]
        if tfs:
            tf_mean = float(np.mean(tfs))
            tf_max  = float(np.max(tfs))
            tf_min  = float(np.min(tfs))
            tf_sum  = float(np.sum(tfs))
            tf_var  = float(np.var(tfs))
        else:
            tf_mean = tf_max = tf_min = tf_sum = tf_var = 0.0
        tf_idf_sum = sum(
            tfs[i] * self.bm25_body._idf.get(q_tokens[i], 0.0)
            for i in range(len(q_tokens))
        )

        # ── Document quality (use pair attributes if present) ─
        doc_pagerank   = getattr(pair, "pagerank",    0.5)
        doc_click_rate = getattr(pair, "click_rate",  0.5)
        doc_dwell_time = math.log1p(getattr(pair, "dwell_time", 60.0))
        doc_freshness  = getattr(pair, "freshness",   0.5)
        doc_spam_score = 1.0 - getattr(pair, "spam_score", 0.0)
        doc_authority  = getattr(pair, "authority",   0.5)

        # ── Query intent ────────────────────────────────────
        query_clarity       = self._query_clarity(q_tokens)
        is_navigational     = 1.0 if any(
            tok in d_set or tok in set(t_tokens) for tok in q_tokens
            if len(tok) > 8
        ) else 0.0
        is_informational    = 1.0 - is_navigational
        query_freq          = math.log1p(getattr(pair, "query_freq",   100.0))
        avg_result_clicks   = getattr(pair, "avg_clicks", 0.3)
        reformulation_rate  = getattr(pair, "reform_rate", 0.1)

        # ── Semantic similarity ──────────────────────────────
        cosine_tfidf  = self._cosine_tfidf(q_tokens, d_tokens)
        bm25_rank_pct = getattr(pair, "bm25_rank_pct", 0.5)
        lm_dir_score  = self._lm_dirichlet(q_tokens, pair.doc_id)
        axiomatic     = self._f2exp(q_tokens, pair.doc_id)
        overlap_bigram  = self._ngram_overlap(q_tokens, d_tokens, n=2)
        overlap_trigram = self._ngram_overlap(q_tokens, d_tokens, n=3)

        vec = np.array([
            bm25_body, bm25_title, bm25_url, bm25_anchor, bm25_b0, bm25_k2,
            idf_sum, qtf_coverage, qtf_exact_phrase, query_len, doc_len_log, query_doc_ratio,
            tf_mean, tf_max, tf_min, tf_sum, tf_var, tf_idf_sum,
            doc_pagerank, doc_click_rate, doc_dwell_time, doc_freshness, doc_spam_score, doc_authority,
            query_clarity, is_navigational, is_informational, query_freq, avg_result_clicks, reformulation_rate,
            cosine_tfidf, bm25_rank_pct, lm_dir_score, axiomatic, overlap_bigram, overlap_trigram,
        ], dtype=np.float32)
        return vec

    # ── helpers ─────────────────────────────────────────────

    def _query_clarity(self, q_tokens: list[str]) -> float:
        """Simplified query clarity: avg IDF of query terms."""
        if not q_tokens:
            return 0.0
        return np.mean([self.bm25_body._idf.get(t, 0.0) for t in q_tokens])

    def _cosine_tfidf(self, q_tokens: list[str], d_tokens: list[str]) -> float:
        if not q_tokens or not d_tokens:
            return 0.0
        vocab = set(q_tokens) | set(d_tokens)
        d_tf = defaultdict(int)
        for t in d_tokens:
            d_tf[t] += 1
        q_tf = defaultdict(int)
        for t in q_tokens:
            q_tf[t] += 1
        dot = 0.0
        q_norm = 0.0
        d_norm = 0.0
        for tok in vocab:
            idf = self.bm25_body._idf.get(tok, 0.0)
            qv = q_tf[tok] * idf
            dv = d_tf[tok] * idf
            dot += qv * dv
            q_norm += qv * qv
            d_norm += dv * dv
        if q_norm == 0 or d_norm == 0:
            return 0.0
        return dot / (math.sqrt(q_norm) * math.sqrt(d_norm))

    def _lm_dirichlet(self, q_tokens: list[str], doc_id: str) -> float:
        """LM with Dirichlet smoothing."""
        if not q_tokens:
            return 0.0
        dl = self.bm25_body._doc_len.get(doc_id, 0)
        total_terms = sum(self.bm25_body._doc_len.values()) or 1
        score = 0.0
        for tok in q_tokens:
            tf = self.bm25_body._term_freq.get(tok, {}).get(doc_id, 0)
            cf = sum(self.bm25_body._term_freq.get(tok, {}).values()) / total_terms
            p = (tf + self.mu * cf) / (dl + self.mu)
            score += math.log(max(p, 1e-12))
        return score

    def _f2exp(self, q_tokens: list[str], doc_id: str) -> float:
        """F2Exp axiomatic retrieval score (simplified)."""
        dl = self.bm25_body._doc_len.get(doc_id, 0)
        avg_dl = self.bm25_body._avg_dl or 1.0
        score = 0.0
        for tok in set(q_tokens):
            tf = self.bm25_body._term_freq.get(tok, {}).get(doc_id, 0)
            idf = self.bm25_body._idf.get(tok, 0.0)
            norm = tf / (tf + 1 + dl / avg_dl)
            score += idf * norm
        return score

    @staticmethod
    def _ngram_overlap(q_tokens: list[str], d_tokens: list[str], n: int) -> float:
        def ngrams(toks):
            return set(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))
        q_ng = ngrams(q_tokens)
        d_ng = ngrams(d_tokens)
        if not q_ng:
            return 0.0
        return len(q_ng & d_ng) / len(q_ng)


# ─────────────────────────────────────────────────────────────
# NDCG Metric
# ─────────────────────────────────────────────────────────────

def dcg_at_k(rels: list[int], k: int) -> float:
    rels = rels[:k]
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(rels: list[int], k: int) -> float:
    idcg = dcg_at_k(sorted(rels, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def evaluate_ranking(
    query_ids: np.ndarray,
    doc_rels: np.ndarray,
    scores: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate per-query NDCG@k and return mean ± std."""
    ndcg_scores = []
    for qid in np.unique(query_ids):
        mask = query_ids == qid
        q_rels  = doc_rels[mask]
        q_scores = scores[mask]
        order   = np.argsort(-q_scores)
        ranked_rels = q_rels[order].tolist()
        ndcg_scores.append(ndcg_at_k(ranked_rels, k))
    arr = np.array(ndcg_scores)
    return {
        f"ndcg@{k}": float(np.mean(arr)),
        f"ndcg@{k}_std": float(np.std(arr)),
        "n_queries": len(arr),
    }


# ─────────────────────────────────────────────────────────────
# LambdaMART Model
# ─────────────────────────────────────────────────────────────

class LambdaMARTRanker:
    """
    LambdaMART (LightGBM rank:ndcg objective).

    Hyper-parameters are tuned for the 2M-pair corpus.
    Key design choices:
      • rank:ndcg objective directly optimises NDCG.
      • label_gain maps 0-4 relevance to NDCG gains.
      • num_leaves=255 for high capacity; regularisation via
        min_child_samples and lambda_l2 prevents overfitting.
      • Early stopping on validation NDCG@10.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective":          "rank:ndcg",
        "metric":             "ndcg",
        "eval_at":            [5, 10, 20],
        "ndcg_eval_at":       [5, 10, 20],
        "label_gain":         [0, 3, 7, 15, 31],   # 2^grade - 1 for grades 0-4
        "num_leaves":         255,
        "max_depth":          -1,
        "learning_rate":      0.05,
        "n_estimators":       2000,
        "subsample":          0.8,
        "subsample_freq":     1,
        "colsample_bytree":   0.8,
        "min_child_samples":  50,
        "lambda_l1":          0.0,
        "lambda_l2":          1.0,
        "min_split_gain":     0.0,
        "n_jobs":             -1,
        "verbose":            -1,
        "random_state":       42,
    }

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: lgb.LGBMRanker | None = None
        self.scaler = RobustScaler()
        self._feature_names = FEATURE_NAMES

    # ── training ────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        groups_val: np.ndarray,
        early_stopping_rounds: int = 50,
    ) -> "LambdaMARTRanker":
        log.info(
            "Training LambdaMART | train=%d pairs | val=%d pairs | features=%d",
            len(X_train), len(X_val), X_train.shape[1],
        )
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)

        self.model = lgb.LGBMRanker(**self.params)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100),
        ]

        self.model.fit(
            X_train_s,
            y_train,
            group=self._query_group_sizes(groups_train),
            eval_set=[(X_val_s, y_val)],
            eval_group=[self._query_group_sizes(groups_val)],
            eval_names=["val"],
            callbacks=callbacks,
        )
        log.info("Best iteration: %d", self.model.best_iteration_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)

    def feature_importance(self) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self._feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path / "lgbm_ranker.txt"))
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        log.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LambdaMARTRanker":
        path = Path(path)
        obj = cls()
        obj.model = lgb.LGBMRanker()
        obj.model._Booster = lgb.Booster(model_file=str(path / "lgbm_ranker.txt"))
        with open(path / "scaler.pkl", "rb") as f:
            obj.scaler = pickle.load(f)
        return obj

    @staticmethod
    def _query_group_sizes(query_ids: np.ndarray) -> list[int]:
        """Convert sorted query ID array → list of group sizes for LightGBM."""
        _, counts = np.unique(query_ids, return_counts=True)
        return counts.tolist()


# ─────────────────────────────────────────────────────────────
# Data Loader  (plug in your own corpus reader here)
# ─────────────────────────────────────────────────────────────

def load_corpus_from_tsv(path: str | Path) -> dict[str, str]:
    """Load {doc_id: text} from a two-column TSV (doc_id \\t text)."""
    docs: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                docs[parts[0]] = parts[1]
    log.info("Loaded %d documents from %s", len(docs), path)
    return docs


def load_pairs_from_jsonl(path: str | Path) -> list[QueryDocPair]:
    """
    Load query-doc pairs from JSONL.
    Each line: {"qid": "...", "did": "...", "query": "...", "text": "...",
                "title": "...", "relevance": 3}
    """
    pairs: list[QueryDocPair] = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            pairs.append(QueryDocPair(
                query_id   = obj["qid"],
                doc_id     = obj["did"],
                query_text = obj["query"],
                doc_text   = obj["text"],
                title      = obj.get("title", ""),
                relevance  = int(obj.get("relevance", 0)),
            ))
    log.info("Loaded %d query-doc pairs from %s", len(pairs), path)
    return pairs


# ─────────────────────────────────────────────────────────────
# Synthetic Data Generator  (for smoke-testing without real data)
# ─────────────────────────────────────────────────────────────

VOCAB = (
    "the quick brown fox jumps over lazy dog information retrieval ranking "
    "learning neural network model query document relevance score feature "
    "search engine text classification deep machine learning natural language "
    "processing transformer attention mechanism evaluation metric precision recall "
    "corpus index term frequency inverse document sparse dense vector embedding"
).split()


def _rand_text(rng: np.random.Generator, n: int = 50) -> str:
    return " ".join(rng.choice(VOCAB, size=n))


def generate_synthetic_dataset(
    n_queries: int = 500,
    docs_per_query: int = 100,
    seed: int = 0,
) -> tuple[list[QueryDocPair], dict[str, str]]:
    """
    Generate a plausible synthetic dataset for smoke-testing.
    Relevance labels are correlated with term overlap so that
    the ranker has a real signal to learn from.
    """
    rng = np.random.default_rng(seed)
    docs: dict[str, str] = {}
    pairs: list[QueryDocPair] = []

    for qidx in range(n_queries):
        q_terms = rng.choice(VOCAB, size=rng.integers(3, 7)).tolist()
        query   = " ".join(q_terms)
        qid     = f"q{qidx:06d}"

        for didx in range(docs_per_query):
            did = f"d{qidx:06d}_{didx:04d}"
            # Inject query terms with prob ∝ desired relevance
            rel = rng.choice([0, 1, 2, 3, 4], p=[0.35, 0.3, 0.2, 0.1, 0.05])
            n_inject = rel * 3
            base_text = _rand_text(rng, 60)
            injected  = (q_terms * (n_inject // len(q_terms) + 1))[:n_inject]
            rng.shuffle(injected)
            doc_text = base_text + " " + " ".join(injected)
            docs[did] = doc_text
            pairs.append(QueryDocPair(
                query_id   = qid,
                doc_id     = did,
                query_text = query,
                doc_text   = doc_text,
                title      = _rand_text(rng, 6),
                relevance  = rel,
            ))

    return pairs, docs


# ─────────────────────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────

class RankingPipeline:
    """End-to-end LambdaMART pipeline with offline evaluation."""

    def __init__(
        self,
        model_dir: str = "models/lambdamart",
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
    ):
        self.model_dir     = Path(model_dir)
        self.val_fraction  = val_fraction
        self.test_fraction = test_fraction
        self.bm25          = BM25()
        self.ranker        = LambdaMARTRanker()
        self.feature_eng: FeatureEngineer | None = None

    # ── main entry ──────────────────────────────────────────

    def run(
        self,
        pairs: list[QueryDocPair],
        docs: dict[str, str],
        skip_training: bool = False,
    ) -> dict[str, Any]:
        """Full pipeline: index → features → split → train → evaluate."""

        # 1. Build BM25 index
        self.bm25.index(docs)
        self.feature_eng = FeatureEngineer(self.bm25)

        # 2. Extract features + BM25 scores
        log.info("Extracting features for %d pairs …", len(pairs))
        t0 = time.perf_counter()
        X, y, qids = self._extract_features(pairs)
        log.info("Feature extraction done in %.1fs", time.perf_counter() - t0)

        # 3. Compute BM25 scores for baseline
        bm25_scores = np.array([
            self.bm25.score(p.query_text, p.doc_id) for p in pairs
        ])

        # 4. Train / test split (group-aware — queries don't leak across splits)
        (X_train, y_train, q_train,
         X_val,   y_val,   q_val,
         X_test,  y_test,  q_test,
         bm25_test) = self._split(X, y, qids, bm25_scores)

        # 5. Train LambdaMART
        if not skip_training:
            self.ranker.fit(X_train, y_train, q_train, X_val, y_val, q_val)
            self.ranker.save(self.model_dir)

        # 6. Evaluate on held-out test set
        lm_scores = self.ranker.predict(X_test)

        results_lm   = evaluate_ranking(q_test, y_test, lm_scores,   k=10)
        results_bm25 = evaluate_ranking(q_test, y_test, bm25_test,   k=10)

        ndcg_lm   = results_lm["ndcg@10"]
        ndcg_bm25 = results_bm25["ndcg@10"]
        delta_pct = (ndcg_lm - ndcg_bm25) / max(ndcg_bm25, 1e-9) * 100

        log.info("=" * 60)
        log.info("OFFLINE EVALUATION RESULTS")
        log.info("  BM25 baseline  NDCG@10 = %.4f", ndcg_bm25)
        log.info("  LambdaMART     NDCG@10 = %.4f", ndcg_lm)
        log.info("  Delta                  = %+.1f%%", delta_pct)
        log.info("  Target (≥18%% gain)     = %s",
                 "✓ ACHIEVED" if delta_pct >= 18.0 else "✗ BELOW TARGET")
        log.info("=" * 60)

        fi = self.ranker.feature_importance()
        log.info("Top-10 features:\n%s", fi.head(10).to_string(index=False))

        return {
            "bm25_ndcg10":     ndcg_bm25,
            "lambdamart_ndcg10": ndcg_lm,
            "delta_pct":       delta_pct,
            "target_met":      delta_pct >= 18.0,
            "n_test_queries":  results_lm["n_queries"],
            "feature_importance": fi,
        }

    # ── internals ───────────────────────────────────────────

    def _extract_features(
        self, pairs: list[QueryDocPair]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X    = np.zeros((len(pairs), 36), dtype=np.float32)
        y    = np.zeros(len(pairs), dtype=np.int32)
        qids = np.empty(len(pairs), dtype=object)

        for i, pair in enumerate(pairs):
            X[i]    = self.feature_eng.extract(pair)
            y[i]    = pair.relevance
            qids[i] = pair.query_id

        return X, y, qids

    def _split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        qids: np.ndarray,
        bm25_scores: np.ndarray,
    ):
        """Group-aware split so each query appears in exactly one partition."""
        unique_qids = np.unique(qids)
        rng = np.random.default_rng(42)
        rng.shuffle(unique_qids)

        n      = len(unique_qids)
        n_test = max(1, int(n * self.test_fraction))
        n_val  = max(1, int(n * self.val_fraction))

        test_qids  = set(unique_qids[:n_test])
        val_qids   = set(unique_qids[n_test: n_test + n_val])
        train_qids = set(unique_qids[n_test + n_val:])

        def mask(qid_set):
            return np.array([q in qid_set for q in qids])

        tr = mask(train_qids)
        vl = mask(val_qids)
        te = mask(test_qids)

        return (
            X[tr], y[tr], qids[tr],
            X[vl], y[vl], qids[vl],
            X[te], y[te], qids[te],
            bm25_scores[te],
        )


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LambdaMART ranking pipeline")
    parser.add_argument("--corpus",  help="Path to corpus TSV (doc_id \\t text)")
    parser.add_argument("--pairs",   help="Path to pairs JSONL")
    parser.add_argument("--model-dir", default="models/lambdamart")
    parser.add_argument("--smoke",   action="store_true",
                        help="Run synthetic smoke test (no real data needed)")
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--docs-per-query", type=int, default=100)
    args = parser.parse_args()

    pipeline = RankingPipeline(model_dir=args.model_dir)

    if args.smoke:
        log.info("Running synthetic smoke test (%d queries × %d docs) …",
                 args.n_queries, args.docs_per_query)
        pairs, docs = generate_synthetic_dataset(
            n_queries=args.n_queries,
            docs_per_query=args.docs_per_query,
        )
    else:
        if not args.corpus or not args.pairs:
            parser.error("--corpus and --pairs are required unless --smoke is set")
        docs  = load_corpus_from_tsv(args.corpus)
        pairs = load_pairs_from_jsonl(args.pairs)

    results = pipeline.run(pairs, docs)
    print(json.dumps(
        {k: v for k, v in results.items() if k != "feature_importance"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
