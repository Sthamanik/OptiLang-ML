"""
train.py
Trains the full pipeline: clustering → classification → rank prediction.
Saves all fitted models and artefacts to models/.

Run after preprocess.py.

Saved artefacts
---------------
models/clustering_scaler.joblib
models/clustering_pca.joblib
models/clustering_medoids.joblib
models/le_pattern.joblib
models/le_severity.joblib
models/cluster_names_k{k}.json
models/classifier_{name}_k{k}.pkl
models/best_{name}_rank_predictor.joblib
data/executions_clustered.csv
data/holdout_programs.csv
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from scipy.stats import randint, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    pairwise_distances,
    silhouette_score,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET       = "priority_rank"

CLUSTER_NAMES = {
    0: "string_builder_refactor",
    1: "hoist_and_fold",
    2: "pure_constant_fold",
    3: "guard_clause_exit",
    4: "runtime_cost_reduction",
    5: "dead_code_elimination",
}

def _kmedoids(X: np.ndarray, k: int, max_iter: int = 100, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    D   = pairwise_distances(X, metric="euclidean")
    medoid_idx = rng.choice(len(X), size=k, replace=False)

    for _ in range(max_iter):
        labels = np.argmin(D[:, medoid_idx], axis=1)
        new_medoids = np.array([
            members[np.argmin(D[np.ix_(members := np.where(labels == c)[0], members)].sum(axis=1))]
            if len(members := np.where(labels == c)[0]) > 0 else medoid_idx[c]
            for c in range(k)
        ])
        if np.array_equal(sorted(new_medoids), sorted(medoid_idx)):
            break
        medoid_idx = new_medoids

    return np.argmin(D[:, medoid_idx], axis=1), medoid_idx


def _select_k(X_cluster_scaled: np.ndarray, random_state: int = 42) -> int:
    """Pick k via majority vote across silhouette, Davies-Bouldin, Calinski-Harabasz."""
    rng = np.random.RandomState(random_state)
    sub = X_cluster_scaled[rng.choice(len(X_cluster_scaled), size=min(2000, len(X_cluster_scaled)), replace=False)]

    scores = {k: {} for k in range(2, 11)}
    for k in range(2, 11):
        labels, _ = _kmedoids(sub, k=k, random_state=random_state)
        scores[k]["sil"] = silhouette_score(sub, labels)
        scores[k]["db"]  = davies_bouldin_score(sub, labels)
        scores[k]["ch"]  = calinski_harabasz_score(sub, labels)

    best_sil = max(scores, key=lambda k: scores[k]["sil"])
    best_db  = min(scores, key=lambda k: scores[k]["db"])
    best_ch  = max(scores, key=lambda k: scores[k]["ch"])

    final_k = Counter([best_sil, best_db, best_ch]).most_common(1)[0][0]
    print(f"k selection → sil:{best_sil} db:{best_db} ch:{best_ch} → FINAL_K={final_k}")
    return final_k


def run_clustering(X_raw: pd.DataFrame, meta: pd.DataFrame):
    program_ids = np.array(sorted(meta["program_id"].unique()))
    cluster_prgs, holdout_prgs = train_test_split(program_ids, test_size=0.20, random_state=RANDOM_STATE)

    idx_cluster = np.flatnonzero(meta["program_id"].isin(set(cluster_prgs)).values)
    idx_holdout = np.flatnonzero(meta["program_id"].isin(set(holdout_prgs)).values)

    X_c_raw, X_h_raw = X_raw.iloc[idx_cluster], X_raw.iloc[idx_holdout]

    scaler = StandardScaler()
    X_c_scaled = scaler.fit_transform(X_c_raw)
    X_h_scaled = scaler.transform(X_h_raw)

    cumvar  = np.cumsum(PCA(random_state=RANDOM_STATE).fit(X_c_scaled).explained_variance_ratio_)
    n_comps = int(np.argmax(cumvar >= 0.95)) + 1
    pca     = PCA(n_components=n_comps, random_state=RANDOM_STATE)
    X_c_pca = pca.fit_transform(X_c_scaled)
    X_h_pca = pca.transform(X_h_scaled)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "clustering_scaler.joblib"))
    joblib.dump(pca,    os.path.join(MODELS_DIR, "clustering_pca.joblib"))
    print(f"PCA: {n_comps} components ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

    # Select k and cluster
    FINAL_K = _select_k(X_c_pca, RANDOM_STATE)
    labels_c, medoid_idx = _kmedoids(X_c_pca, k=FINAL_K, random_state=RANDOM_STATE)
    medoid_vectors = X_c_pca[medoid_idx]
    joblib.dump(
        {
            "medoid_vectors": medoid_vectors,
            "cluster_labels": list(range(FINAL_K)),
            "feature_columns": list(X_raw.columns),
            "k": FINAL_K,
        },
        os.path.join(MODELS_DIR, "clustering_medoids.joblib"),
    )

    # Assign holdout rows to nearest medoid
    dists     = np.linalg.norm(X_h_pca[:, None, :] - medoid_vectors[None, :, :], axis=2)
    labels_h  = np.argmin(dists, axis=1)

    final_labels = np.empty(len(X_raw), dtype=int)
    final_labels[idx_cluster] = labels_c
    final_labels[idx_holdout] = labels_h

    sil = silhouette_score(X_c_pca, labels_c)
    print(f"Silhouette: {sil:.4f} | k={FINAL_K}")

    return final_labels, FINAL_K, idx_holdout, set(holdout_prgs)


def run_classification(X_raw: pd.DataFrame, meta: pd.DataFrame,
                       holdout_prgs: set, FINAL_K: int):
    le_pattern  = LabelEncoder().fit(meta["pattern"].values)
    le_severity = LabelEncoder().fit(meta["severity"].values)
    joblib.dump(le_pattern,  os.path.join(MODELS_DIR, "le_pattern.joblib"))
    joblib.dump(le_severity, os.path.join(MODELS_DIR, "le_severity.joblib"))

    y = le_pattern.transform(meta["pattern"].values)

    meta["_pid"] = meta["program_id"].astype(str)
    is_holdout   = meta["_pid"].isin(holdout_prgs)
    idx_test     = np.where(is_holdout)[0]
    pool_prgs    = meta.loc[~is_holdout, "_pid"].unique()

    train_prgs, val_prgs = train_test_split(pool_prgs, test_size=0.20, random_state=RANDOM_STATE)
    idx_train = np.where(meta["_pid"].isin(set(train_prgs)))[0]
    idx_val   = np.where(meta["_pid"].isin(set(val_prgs)))[0]

    X_train, y_train = X_raw.iloc[idx_train], y[idx_train]
    X_val,   y_val   = X_raw.iloc[idx_val],   y[idx_val]
    X_test,  y_test  = X_raw.iloc[idx_test],  y[idx_test]
    groups           = meta.loc[idx_train, "program_id"].values

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Random Forest
    rf_search = RandomizedSearchCV(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", RandomForestClassifier(class_weight="balanced",
                                                  random_state=RANDOM_STATE, n_jobs=-1))]),
        {"clf__n_estimators": randint(100, 500), "clf__max_depth": [None, 10, 20, 30],
         "clf__min_samples_leaf": randint(1, 10), "clf__max_features": ["sqrt", "log2"]},
        n_iter=10, cv=cv, scoring="f1_weighted", random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    rf_search.fit(X_train, y_train, groups=groups)

    # XGBoost
    xgb_search = RandomizedSearchCV(
        xgb.XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE,
                           n_jobs=-1, use_label_encoder=False),
        {"n_estimators": randint(100, 400), "max_depth": randint(3, 10),
         "learning_rate": [0.01, 0.05, 0.1, 0.2],
         "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0]},
        n_iter=30, cv=cv, scoring="f1_weighted", random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    xgb_search.fit(X_train, y_train, groups=groups)

    # Pick winner on val set
    rf_f1  = f1_score(y_val, rf_search.best_estimator_.predict(X_val),  average="weighted")
    xgb_f1 = f1_score(y_val, xgb_search.best_estimator_.predict(X_val), average="weighted")

    if rf_f1 >= xgb_f1:
        best_clf, name = rf_search.best_estimator_, "randomforest"
    else:
        best_clf, name = xgb_search.best_estimator_, "xgboost"

    test_f1 = f1_score(y_test, best_clf.predict(X_test), average="weighted")
    print(f"Classifier: {name} | val_f1={max(rf_f1, xgb_f1):.4f} | test_f1={test_f1:.4f}")

    model_path = os.path.join(MODELS_DIR, f"classifier_{name}_k{FINAL_K}.pkl")
    joblib.dump(best_clf, model_path)

    # Cluster names
    with open(os.path.join(MODELS_DIR, f"cluster_names_k{FINAL_K}.json"), "w") as f:
        json.dump(CLUSTER_NAMES, f, indent=2)

    return best_clf


def _build_cluster_frame(meta: pd.DataFrame, X_raw: pd.DataFrame) -> tuple:
    scaler = joblib.load(os.path.join(MODELS_DIR, "clustering_scaler.joblib"))
    scaled = pd.DataFrame(scaler.transform(X_raw), columns=X_raw.columns, index=X_raw.index).add_prefix("scaled_")

    row_level = meta.copy()
    row_level = pd.concat([row_level, scaled], axis=1)

    prog_counts = row_level.groupby("program_id")["suggestion_id"].count().rename("program_suggestion_count")

    cf = row_level.groupby(["program_id", "cluster"]).agg(
        cluster_suggestion_count=("suggestion_id", "count"),
        cluster_impact_sum=("impact_score", "sum"),
        cluster_impact_mean=("impact_score", "mean"),
    ).reset_index()
    cf = cf.merge(prog_counts.reset_index(), on="program_id")
    cf["cluster_share"] = cf["cluster_suggestion_count"] / cf["program_suggestion_count"]
    cf[TARGET] = (cf.groupby("program_id")["cluster_impact_sum"]
                    .rank(ascending=False, method="dense").astype(int))

    scaled_agg = row_level.groupby(["program_id", "cluster"])[scaled.columns].mean().reset_index()
    cf = cf.merge(scaled_agg, on=["program_id", "cluster"])

    archetype_map = (row_level.groupby("program_id")["pattern"]
                     .apply(frozenset).reset_index())
    archetype_map["archetype"] = archetype_map["pattern"].apply(lambda s: "_".join(sorted(s)))
    cf = cf.merge(archetype_map[["program_id", "archetype"]], on="program_id", how="left")

    # Feature selection — exclude leakage cols
    exclude = {
        "program_id", "cluster_impact_sum", "cluster_impact_mean",
        "archetype", TARGET,
    }
    scaled_safe = [c for c in scaled.columns
                   if not any(kw in c.lower() for kw in
                               ["impact", "score", "efficiency", "quality",
                                "maintainability", "improvement", "gain"])]
    feature_cols = [c for c in ["cluster", "cluster_suggestion_count",
                                  "cluster_share", "program_suggestion_count"] + scaled_safe
                    if c not in exclude and c in cf.columns]

    # Auto-drop any feature correlated > 0.85 with target
    corr = cf[feature_cols].corrwith(cf[TARGET]).abs()
    leaky = corr[corr > 0.85].index.tolist()
    feature_cols = [c for c in feature_cols if c not in leaky]

    return cf, feature_cols


def run_prediction(meta: pd.DataFrame, X_raw: pd.DataFrame):
    cf, feature_cols = _build_cluster_frame(meta, X_raw)

    archetypes = cf["archetype"].unique()
    train_arch, temp = train_test_split(archetypes, test_size=0.30, random_state=RANDOM_STATE)
    val_arch, test_arch = train_test_split(temp, test_size=0.50, random_state=RANDOM_STATE)

    X, y = cf[feature_cols], cf[TARGET].values
    X_train, y_train = X.loc[cf["archetype"].isin(train_arch)], y[cf["archetype"].isin(train_arch)]
    X_val,   y_val   = X.loc[cf["archetype"].isin(val_arch)],   y[cf["archetype"].isin(val_arch)]
    X_test,  y_test  = X.loc[cf["archetype"].isin(test_arch)],  y[cf["archetype"].isin(test_arch)]
    groups           = cf.loc[cf["archetype"].isin(train_arch), "archetype"].values

    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    reg_params = {
        "xgboost": (
            xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": randint(50, 100), "max_depth": [1],
             "learning_rate": [0.01], "subsample": [0.3],
             "colsample_bytree": [0.3], "min_child_weight": [100, 200],
             "reg_alpha": [10, 20], "reg_lambda": [20, 50]},
        ),
        "randomforest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": randint(50, 200), "max_depth": [1, 2],
             "min_samples_leaf": [100, 200, 500], "max_features": ["sqrt", "log2"]},
        ),
    }

    best_model, best_name, best_val_mae = None, None, float("inf")
    for name, (estimator, params) in reg_params.items():
        search = RandomizedSearchCV(estimator, params, n_iter=30, cv=cv,
                                    scoring="neg_mean_absolute_error",
                                    random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
        search.fit(X_train, y_train, groups=groups)
        val_mae = mean_absolute_error(y_val, search.best_estimator_.predict(X_val))
        print(f"Regressor {name}: val_mae={val_mae:.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model   = search.best_estimator_
            best_name    = name

    test_mae = mean_absolute_error(y_test, best_model.predict(X_test))
    rho      = spearmanr(y_test, best_model.predict(X_test)).statistic
    print(f"Regressor winner: {best_name} | test_mae={test_mae:.4f} | spearman_rho={rho:.4f}")

    joblib.dump(best_model, os.path.join(MODELS_DIR, f"best_{best_name}_rank_predictor.joblib"))
    return best_model


def main() -> None:
    X_raw = pd.read_csv(os.path.join(DATA_DIR, "executions_features_raw.csv"))
    meta  = pd.read_csv(os.path.join(DATA_DIR, "executions_meta.csv")).reset_index(drop=True)
    assert len(X_raw) == len(meta)

    print("\n── Clustering ──────────────────────────────")
    final_labels, FINAL_K, idx_holdout, holdout_prgs = run_clustering(X_raw, meta)
    meta["cluster"] = final_labels
    meta.to_csv(os.path.join(DATA_DIR, "executions_clustered.csv"), index=False)
    pd.DataFrame({"program_id": sorted(holdout_prgs)}).to_csv(
        os.path.join(DATA_DIR, "holdout_programs.csv"), index=False
    )

    print("\n── Classification ──────────────────────────")
    X_with_cluster = X_raw.copy()
    X_with_cluster["cluster"] = final_labels
    run_classification(X_with_cluster, meta, {str(p) for p in holdout_prgs}, FINAL_K)

    print("\n── Rank Prediction ─────────────────────────")
    run_prediction(meta, X_raw)

    print("\nAll models saved to models/")


if __name__ == "__main__":
    main()
