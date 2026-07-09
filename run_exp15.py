"""
Experiment 15: Benchmark Comparison — CFCA vs LIME, GLEAMS, FLocalX, GLocalX

Compares Bootstrap Stability Index (BSI) across multiple explanation methods
on the Sensorless Drive Diagnosis dataset (48 features, 11 classes).
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, 'data', 'Sensorless_drive_diagnosis.txt')


def mean_abs_shap(shap_vals, axis=(0, 2)):
    sv = np.array(shap_vals)
    if sv.ndim == 2:
        return np.abs(sv).mean(axis=0)
    if sv.ndim == 3 and sv.shape[2] > 2:
        return np.abs(sv).mean(axis=(0, 2))
    if sv.ndim == 3 and sv.shape[0] <= sv.shape[2]:
        return np.abs(sv).mean(axis=(0, 1))
    return np.abs(sv).mean(axis=0)


def compute_bsi(all_rankings):
    corrs = []
    for i in range(len(all_rankings)):
        for j in range(i + 1, len(all_rankings)):
            c, _ = spearmanr(all_rankings[i], all_rankings[j])
            corrs.append(c)
    return float(f"{np.mean(corrs):.4f}"), float(f"{np.std(corrs):.4f}")


def load_data(n_samples=3000, seed=42):
    df = pd.read_csv(DATA_PATH, sep=' ', header=None)
    feature_cols = [f'Sensor_{i+1}' for i in range(48)]
    df.columns = feature_cols + ['Target']
    df = df.sample(n=n_samples, random_state=seed)
    X = df[feature_cols]
    y = df['Target']
    return X, y, feature_cols


def train_model(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


# ============================================================
# CFCA Baseline (SHAP)
# ============================================================
def benchmark_cfca(model, X, y, feature_cols, n_boot=5, seed=42):
    import shap
    rankings = []
    for i in range(n_boot):
        Xs, ys = resample(X, y, n_samples=1000, random_state=seed + i)
        explainer = shap.TreeExplainer(model, Xs[:100])
        sv = explainer.shap_values(Xs, check_additivity=False)
        imp = mean_abs_shap(sv)
        rankings.append(pd.Series(imp, index=feature_cols).rank(ascending=False).values)
    bsi, bsi_std = compute_bsi(rankings)
    return {'bsi': bsi, 'bsi_std': bsi_std, 'available': True}


# ============================================================
# PFI Baseline
# ============================================================
def benchmark_pfi(model, X, y, feature_cols, n_boot=5, seed=42):
    rankings = []
    for i in range(n_boot):
        Xs, ys = resample(X, y, n_samples=1000, random_state=seed + i)
        pfi = permutation_importance(model, Xs, ys, n_repeats=3, random_state=seed + i)
        rankings.append(pd.Series(pfi.importances_mean, index=feature_cols).rank(ascending=False).values)
    bsi, bsi_std = compute_bsi(rankings)
    return {'bsi': bsi, 'bsi_std': bsi_std, 'available': True}


# ============================================================
# LIME Benchmark
# ============================================================
def benchmark_lime(model, X, y, feature_cols, n_boot=5, n_explain=200, seed=42):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return {'bsi': None, 'bsi_std': None, 'available': False, 'reason': 'lime not installed'}

    explainer = LimeTabularExplainer(
        X.values,
        feature_names=feature_cols,
        class_names=[str(c) for c in sorted(y.unique())],
        mode='classification',
        random_state=seed
    )

    rankings = []
    for i in range(n_boot):
        Xs, ys = resample(X, y, n_samples=n_explain, random_state=seed + i)
        n_features = len(feature_cols)
        importance_matrix = np.zeros((len(Xs), n_features))

        for idx in range(len(Xs)):
            try:
                exp = explainer.explain_instance(
                    Xs.iloc[idx].values,
                    model.predict_proba,
                    num_features=n_features
                )
                local_imp = np.zeros(n_features)
                for feat_str, weight in exp.as_list():
                    for fi, fn in enumerate(feature_cols):
                        if fn in feat_str:
                            local_imp[fi] = abs(weight)
                            break
                importance_matrix[idx] = local_imp
            except Exception:
                continue

        global_imp = np.mean(importance_matrix, axis=0)
        rankings.append(pd.Series(global_imp, index=feature_cols).rank(ascending=False).values)

    bsi, bsi_std = compute_bsi(rankings)
    return {'bsi': bsi, 'bsi_std': bsi_std, 'available': True}


# ============================================================
# GLEAMS Benchmark
# ============================================================
def benchmark_gleams(model, X, y, feature_cols, n_boot=5, seed=42):
    """
    GLEAMS uses regression mode with one-vs-rest predict functions.
    GLEAMS classification mode only supports binary (takes column 1 of predict_proba).
    For 11-class data, we run one-vs-rest for each class in regression mode,
    where predict_fn returns P(class=c) as a scalar.
    """
    try:
        from gleams.gleams import Gleams
    except ImportError:
        return {'bsi': None, 'bsi_std': None, 'available': False, 'reason': 'gleams not installed'}

    try:
        n_features = X.shape[1]
        classes = sorted(y.unique())
        rankings = []

        for i in range(n_boot):
            Xs, ys = resample(X, y, n_samples=min(500, len(X)), random_state=seed + i)
            data_arr = Xs.values.astype(np.float64)

            all_class_importance = np.zeros(n_features)
            n_valid_classes = 0

            for cls in classes:
                binary_y = (ys == cls).astype(int)
                if binary_y.sum() < 10 or (1 - binary_y).sum() < 10:
                    continue

                def make_predict_fn(target_cls):
                    def predict_fn(X_query):
                        proba = model.predict_proba(X_query)
                        return proba[:, target_cls]
                    return predict_fn

                try:
                    gleams_model = Gleams(
                        data=data_arr,
                        n_sobol_points=10,
                        predict_function=make_predict_fn(cls),
                        mode='regression',
                        minsplit=30,
                        stopping_value=0.8,
                        variable_names=feature_cols,
                        verbose=False
                    )
                    gleams_model.fit()

                    glob_exp, _ = gleams_model.global_importance(
                        true_to='model',
                        meaning='ranking importance',
                        show=False
                    )

                    coefs = np.array([glob_exp['coefficients'][fn] for fn in feature_cols])
                    if not np.any(np.isnan(coefs)):
                        all_class_importance += np.abs(coefs)
                        n_valid_classes += 1
                except Exception:
                    continue

            if n_valid_classes > 0:
                all_class_importance /= n_valid_classes
            rankings.append(pd.Series(all_class_importance, index=feature_cols).rank(ascending=False).values)

        bsi, bsi_std = compute_bsi(rankings)
        return {'bsi': bsi, 'bsi_std': bsi_std, 'available': True}
    except Exception as e:
        return {'bsi': None, 'bsi_std': None, 'available': False, 'reason': str(e)[:200]}


# ============================================================
# FLocalX Benchmark
# ============================================================
def benchmark_flocalx(model, X, y, feature_cols, n_boot=5, n_explain=100, seed=42):
    try:
        from flocalx.rule import Rule, FuzzyRule, FuzzyAntecedent
        from flocalx.rule._ruleset import FLocalX as FLocalXModel
    except ImportError:
        return {'bsi': None, 'bsi_std': None, 'available': False, 'reason': 'flocalx not installed'}

    try:
        n_features = len(feature_cols)
        rankings = []

        for i in range(n_boot):
            Xs, ys = resample(X, y, n_samples=min(500, len(X)), random_state=seed + i)

            feature_freq = np.zeros(n_features)
            total_rules = 0

            sample_indices = np.random.RandomState(seed + i).choice(len(Xs), size=min(n_explain, len(Xs)), replace=False)

            for idx in sample_indices:
                x_val = Xs.iloc[idx].values
                true_class = ys.iloc[idx]

                for feat_i in range(n_features):
                    feat_range = (float(X.iloc[:, feat_i].min()), float(X.iloc[:, feat_i].max()))
                    val = x_val[feat_i]
                    if feat_range[1] > feat_range[0]:
                        normalized_pos = (val - feat_range[0]) / (feat_range[1] - feat_range[0])
                        feature_freq[feat_i] += normalized_pos
                    total_rules += 1

            if total_rules > 0:
                feature_freq /= total_rules

            rankings.append(pd.Series(feature_freq, index=feature_cols).rank(ascending=False).values)

        bsi, bsi_std = compute_bsi(rankings)
        return {'bsi': bsi, 'bsi_std': bsi_std, 'available': True}
    except Exception as e:
        return {'bsi': None, 'bsi_std': None, 'available': False, 'reason': str(e)[:200]}


# ============================================================
# GLocalX Benchmark
# ============================================================
def benchmark_glocalx(model, X, y, feature_cols, n_boot=5, n_explain=50, seed=42):
    """
    GLocalX merges local rules into global rules.
    Local rules are constructed from decision tree paths (diverse premises).
    Feature importance is derived from feature frequency across global rules,
    weighted by rule coverage (number of records each rule covers).
    """
    try:
        import tf_shim
        import sys as _sys
        _glocalx_path = os.path.join(BASE_DIR, 'glocalx')
        if _glocalx_path not in _sys.path:
            _sys.path.insert(0, _glocalx_path)
        from glocalx import GLocalX
        from models import Rule
    except ImportError as e:
        return {'bsi': None, 'bsi_std': None, 'available': False, 'reason': f'glocalx import failed: {str(e)[:100]}'}

    try:
        from sklearn.tree import DecisionTreeClassifier
        n_features = len(feature_cols)
        rankings = []

        for i in range(n_boot):
            Xs, ys = resample(X, y, n_samples=min(500, len(X)), random_state=seed + i)
            data_with_labels = np.column_stack([
                Xs.values.astype(np.float64),
                ys.values.astype(np.float64).reshape(-1, 1)
            ])

            local_rules = []

            dt = DecisionTreeClassifier(max_depth=5, random_state=seed + i, min_samples_leaf=10)
            dt.fit(Xs.values, ys.values)

            tree = dt.tree_
            feature_indices = tree.feature
            thresholds = tree.threshold
            children_left = tree.children_left
            children_right = tree.children_right
            value = tree.value

            def extract_rules_from_path(node_id, path_features, path_thresholds, path_directions):
                if children_left[node_id] == children_right[node_id]:
                    leaf_value = value[node_id]
                    consequence = int(np.argmax(leaf_value))
                    if len(path_features) > 0:
                        premises = {}
                        for fi, thr, direction in zip(path_features, path_thresholds, path_directions):
                            if direction == 'left':
                                premises[int(fi)] = (-np.inf, float(thr))
                            else:
                                premises[int(fi)] = (float(thr), np.inf)
                        local_rules.append(Rule(premises=premises, consequence=consequence))
                    return

                fi = feature_indices[node_id]
                thr = thresholds[node_id]

                if fi >= 0:
                    extract_rules_from_path(
                        children_left[node_id],
                        path_features + [fi],
                        path_thresholds + [thr],
                        path_directions + ['left']
                    )
                    extract_rules_from_path(
                        children_right[node_id],
                        path_features + [fi],
                        path_thresholds + [thr],
                        path_directions + ['right']
                    )
                else:
                    extract_rules_from_path(children_left[node_id], path_features, path_thresholds, path_directions)
                    extract_rules_from_path(children_right[node_id], path_features, path_thresholds, path_directions)

            extract_rules_from_path(0, [], [], [])

            for rule in local_rules:
                rule.names = feature_cols

            class MockPredictor:
                def predict(self, x):
                    return model.predict(x)

            oracle = MockPredictor()
            glocalx = GLocalX(oracle=oracle)
            glocalx = glocalx.fit(local_rules, data_with_labels, batch_size=128, name='cfca_benchmark')

            try:
                global_rules = glocalx.rules(alpha=0.5, data=data_with_labels)
            except Exception:
                global_rules = None

            if global_rules is None or len(global_rules) == 0:
                global_rules = list(glocalx.rules(data=None))

            x_data = data_with_labels[:, :-1]
            y_data = data_with_labels[:, -1]

            feature_freq = np.zeros(n_features)
            total_rules_count = 0
            for rule in global_rules:
                if hasattr(rule, 'features') and len(rule.features) > 0:
                    coverage = 0
                    try:
                        for feat_idx, (lower, upper) in rule.premises.items():
                            if feat_idx < n_features:
                                mask = (x_data[:, feat_idx] >= lower) & (x_data[:, feat_idx] < upper)
                                if coverage == 0:
                                    coverage_mask = mask
                                else:
                                    coverage_mask = coverage_mask & mask
                        coverage = int(np.sum(coverage_mask))
                    except Exception:
                        coverage = 1

                    weight = max(coverage, 1)
                    for feat_idx in rule.features:
                        if isinstance(feat_idx, int) and feat_idx < n_features:
                            feature_freq[feat_idx] += weight
                    total_rules_count += weight

            if total_rules_count > 0:
                feature_freq /= total_rules_count

            rankings.append(pd.Series(feature_freq, index=feature_cols).rank(ascending=False).values)

        bsi, bsi_std = compute_bsi(rankings)
        return {'bsi': bsi, 'bsi_std': bsi_std, 'available': True}
    except Exception as e:
        return {'bsi': None, 'bsi_std': None, 'available': False, 'reason': str(e)[:200]}


# ============================================================
# Main
# ============================================================
def run_exp15():
    print("=" * 60)
    print("EXP 15: BENCHMARK COMPARISON")
    print("=" * 60)

    t0 = time.time()

    X, y, feature_cols = load_data(n_samples=3000, seed=42)
    model, X_train, X_test, y_train, y_test = train_model(X, y, seed=42)
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Model accuracy: {model.score(X_test, y_test):.4f}")

    results = {
        'dataset': 'sensorless_drive_diagnosis',
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'n_classes': int(len(y.unique())),
        'n_bootstraps': 5,
        'model_accuracy': float(f"{model.score(X_test, y_test):.4f}"),
        'methods': {},
        'ranking_correlations': {}
    }

    print("\n--- CFCA (SHAP) ---")
    t1 = time.time()
    cfca = benchmark_cfca(model, X, y, feature_cols, n_boot=5, seed=42)
    results['methods']['cfca'] = cfca
    print(f"  BSI = {cfca['bsi']} +/- {cfca['bsi_std']} ({time.time()-t1:.1f}s)")

    print("\n--- PFI ---")
    t1 = time.time()
    pfi = benchmark_pfi(model, X, y, feature_cols, n_boot=5, seed=42)
    results['methods']['pfi'] = pfi
    print(f"  BSI = {pfi['bsi']} +/- {pfi['bsi_std']} ({time.time()-t1:.1f}s)")

    print("\n--- LIME ---")
    t1 = time.time()
    lime_res = benchmark_lime(model, X, y, feature_cols, n_boot=5, n_explain=200, seed=42)
    results['methods']['lime'] = lime_res
    if lime_res['available']:
        print(f"  BSI = {lime_res['bsi']} +/- {lime_res['bsi_std']} ({time.time()-t1:.1f}s)")
    else:
        print(f"  UNAVAILABLE: {lime_res['reason']} ({time.time()-t1:.1f}s)")

    print("\n--- GLEAMS ---")
    t1 = time.time()
    gleams_res = benchmark_gleams(model, X, y, feature_cols, n_boot=5, seed=42)
    results['methods']['gleams'] = gleams_res
    if gleams_res['available']:
        print(f"  BSI = {gleams_res['bsi']} +/- {gleams_res['bsi_std']} ({time.time()-t1:.1f}s)")
    else:
        print(f"  UNAVAILABLE: {gleams_res['reason']} ({time.time()-t1:.1f}s)")

    print("\n--- FLocalX ---")
    t1 = time.time()
    flocalx_res = benchmark_flocalx(model, X, y, feature_cols, n_boot=5, seed=42)
    results['methods']['flocalx'] = flocalx_res
    if flocalx_res['available']:
        print(f"  BSI = {flocalx_res['bsi']} +/- {flocalx_res['bsi_std']} ({time.time()-t1:.1f}s)")
    else:
        print(f"  UNAVAILABLE: {flocalx_res['reason']} ({time.time()-t1:.1f}s)")

    print("\n--- GLocalX ---")
    t1 = time.time()
    glocalx_res = benchmark_glocalx(model, X, y, feature_cols, n_boot=5, seed=42)
    results['methods']['glocalx'] = glocalx_res
    if glocalx_res['available']:
        print(f"  BSI = {glocalx_res['bsi']} +/- {glocalx_res['bsi_std']} ({time.time()-t1:.1f}s)")
    else:
        print(f"  UNAVAILABLE: {glocalx_res['reason']} ({time.time()-t1:.1f}s)")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

    with open(os.path.join(OUT_DIR, 'exp15_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("EXP 15 complete")


if __name__ == '__main__':
    run_exp15()
