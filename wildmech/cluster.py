"""Bottom-up clustering of de-identified summaries (or scrubbed first user turns).

Default embedding is TF-IDF + SVD so the pipeline runs offline. For the paper, swap in a
sentence-embedding model run inside Leo's environment (set embedding: sentence_transformers).
Cluster names come from an LLM that only sees summaries, never raw text.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


def embed(texts: list[str], method: str = "tfidf_svd", dim: int = 100, seed: int = 0) -> np.ndarray:
    if method == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return normalize(model.encode(texts, show_progress_bar=False))
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    k = max(2, min(dim, X.shape[1] - 1, X.shape[0] - 1))
    return normalize(TruncatedSVD(n_components=k, random_state=seed).fit_transform(X))


def choose_k(Z: np.ndarray, k_grid=(10, 15, 20, 30, 40, 60), seed: int = 0, sample: int = 5000) -> tuple[int, dict]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(sample, len(Z)), replace=False)
    scores = {}
    n_distinct = len(np.unique(np.round(Z[idx], 6), axis=0))
    for k in k_grid:
        if k >= min(len(idx), n_distinct):
            continue
        lab = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(Z[idx])
        scores[k] = float(silhouette_score(Z[idx], lab))
    best = max(scores, key=scores.get)
    return best, scores


def cluster(texts: list[str], k: int | None = None, method: str = "tfidf_svd", seed: int = 0):
    Z = embed(texts, method=method, seed=seed)
    scores = {}
    if k is None:
        k, scores = choose_k(Z, seed=seed)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    return labels, Z, {"k": k, "silhouette_grid": scores}


def top_terms(texts: list[str], labels: np.ndarray, n: int = 8) -> dict[int, list[str]]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for c in np.unique(labels):
        mean = np.asarray(X[labels == c].mean(axis=0)).ravel()
        out[int(c)] = vocab[mean.argsort()[::-1][:n]].tolist()
    return out


NAME_SYSTEM = ("You name clusters of de-identified engineering questions. Output JSON only. "
               "Never include company, product, or part identifiers.")
NAME_USER = """Here are sample summaries from one cluster:
{samples}
Give: {{"name": "<= 8 words", "description": "<= 30 words", "suggested_l1": "<one code from: {codes}>"}}"""


def name_clusters(df: pd.DataFrame, label_col: str, text_col: str, backend, codes: list[str],
                  n_samples: int = 25, seed: int = 0) -> pd.DataFrame:
    rows = []
    for c, g in df.groupby(label_col):
        s = g[text_col].sample(min(n_samples, len(g)), random_state=seed).tolist()
        if backend.name == "rules":
            rows.append({label_col: c, "name": "", "description": "", "suggested_l1": None})
            continue
        out = backend.complete_json(NAME_SYSTEM, NAME_USER.format(samples="\n".join(f"- {x}" for x in s),
                                                                  codes=", ".join(codes)))
        out[label_col] = c
        rows.append(out)
    return pd.DataFrame(rows)
