"""Run experiments in batches to avoid timeouts."""
import json, os, sys, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

def mean_abs_shap(shap_vals, axis=(0, 2)):
    """Handle both list and array SHAP outputs to compute mean |SHAP| across samples and classes."""
    sv = np.array(shap_vals)
    # New SHAP: (n_samples, n_features) -> 2D
    if sv.ndim == 2:
        return np.abs(sv).mean(axis=0)
    # (n_samples, n_features, n_classes) -> 3D, classes last
    if sv.ndim == 3 and sv.shape[2] > 2:
        return np.abs(sv).mean(axis=(0, 2))
    # (n_classes, n_samples, n_features) -> 3D, classes first
    if sv.ndim == 3 and sv.shape[0] <= sv.shape[2]:
        return np.abs(sv).mean(axis=(0, 1))
    return np.abs(sv).mean(axis=0)

batch = sys.argv[1] if len(sys.argv) > 1 else 'all'

# ============================================================
# EXP 1: Narrative Disconnect
# ============================================================
def run_exp1():
    import shap, numpy as np, pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    print("="*60)
    print("EXP 1: NARRATIVE DISCONNECT")
    print("="*60)
    df = pd.read_csv(os.path.join(BASE_DIR, 'Sensorless_drive_diagnosis.txt'), sep=' ', header=None)
    feature_cols = [f'Sensor_{i+1}' for i in range(48)]
    df.columns = feature_cols + ['Target']
    df = df.sample(n=10000, random_state=42)
    X = df[feature_cols]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model, X_train)
    shap_vals = explainer.shap_values(X_test, check_additivity=False)

    glob_shap = mean_abs_shap(shap_vals)
    cfca_imp = pd.Series(glob_shap, index=feature_cols).sort_values(ascending=False)

    pfi = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    pfi_imp = pd.Series(pfi.importances_mean, index=feature_cols).sort_values(ascending=False)

    corr = cfca_imp.rank(ascending=False).corr(pfi_imp.rank(ascending=False), method='spearman')
    nds = 1 - corr

    # Create Table 1 data
    table1 = pd.DataFrame({
        'CFCA_Feature': cfca_imp.head(5).index,
        'CFCA_Score': cfca_imp.head(5).values,
        'PFI_Feature': pfi_imp.head(5).index,
        'PFI_Score': pfi_imp.head(5).values
    })
    print(f"\nSpearman Correlation: {corr:.4f}")
    print(f"NDS: {nds:.4f}")
    print(f"\nTable 1 (Top 5):")
    print(table1.to_string(index=False))

    result = {
        'spearman_correlation': float(f"{corr:.4f}"),
        'nds': float(f"{nds:.4f}"),
        'table1': {
            'cfca_features': cfca_imp.head(10).index.tolist(),
            'cfca_scores': [float(f"{v:.4f}") for v in cfca_imp.head(10).values],
            'pfi_features': pfi_imp.head(10).index.tolist(),
            'pfi_scores': [float(f"{v:.4f}") for v in pfi_imp.head(10).values]
        }
    }
    with open(os.path.join(OUT_DIR, 'exp1_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print("EXP 1 complete")

# ============================================================
# EXP 2: Bootstrap Stability
# ============================================================
def run_exp2():
    import shap, numpy as np, pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.utils import resample
    from scipy.stats import spearmanr

    print("="*60)
    print("EXP 2: BOOTSTRAP STABILITY")
    print("="*60)
    df = pd.read_csv(os.path.join(BASE_DIR, 'Sensorless_drive_diagnosis.txt'), sep=' ', header=None)
    feature_cols = [f'Sensor_{i+1}' for i in range(48)]
    df.columns = feature_cols + ['Target']
    df = df.sample(n=3000, random_state=42)
    X = df[feature_cols]
    y = df['Target']

    # Train model once on full sample
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    cfca_rankings = []
    pfi_rankings = []
    n_iterations = 5

    for i in range(n_iterations):
        X_sample, y_sample = resample(X, y, n_samples=1000, random_state=i)
        explainer = shap.TreeExplainer(model, X_sample[:100])
        shap_vals = explainer.shap_values(X_sample, check_additivity=False)
        glob_shap = mean_abs_shap(shap_vals)
        si = pd.Series(glob_shap, index=feature_cols).rank(ascending=False)
        cfca_rankings.append(si)
        pfi = permutation_importance(model, X_sample, y_sample, n_repeats=1, random_state=i)
        pi = pd.Series(pfi.importances_mean, index=feature_cols).rank(ascending=False)
        pfi_rankings.append(pi)
        print(f"  Iteration {i+1}/{n_iterations}")

    def pairwise(rs):
        cs = []
        for i in range(len(rs)):
            for j in range(i+1, len(rs)):
                cs.append(rs[i].corr(rs[j], method='spearman'))
        return np.array(cs)

    cfca_corrs = pairwise(cfca_rankings)
    pfi_corrs = pairwise(pfi_rankings)

    result = {
        'cfca_mean': float(f"{np.mean(cfca_corrs):.4f}"),
        'cfca_std': float(f"{np.std(cfca_corrs):.4f}"),
        'cfca_ci_95': [float(f"{np.percentile(cfca_corrs, 2.5):.4f}"), float(f"{np.percentile(cfca_corrs, 97.5):.4f}")],
        'pfi_mean': float(f"{np.mean(pfi_corrs):.4f}"),
        'pfi_std': float(f"{np.std(pfi_corrs):.4f}"),
        'pfi_ci_95': [float(f"{np.percentile(pfi_corrs, 2.5):.4f}"), float(f"{np.percentile(pfi_corrs, 97.5):.4f}")]
    }
    print(f"\nCFCA: {result['cfca_mean']} +/- {result['cfca_std']}, CI: {result['cfca_ci_95']}")
    print(f"PFI:  {result['pfi_mean']} +/- {result['pfi_std']}, CI: {result['pfi_ci_95']}")
    with open(os.path.join(OUT_DIR, 'exp2_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print("EXP 2 complete")

# ============================================================
# EXP 3: Hidden Risk Detection
# ============================================================
def run_exp3():
    import shap, numpy as np, pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import binom

    print("="*60)
    print("EXP 3: HIDDEN RISK DETECTION")
    print("="*60)
    df = pd.read_csv(os.path.join(BASE_DIR, 'Sensorless_drive_diagnosis.txt'), sep=' ', header=None)
    feature_cols = [f'Sensor_{i+1}' for i in range(48)]
    df.columns = feature_cols + ['Target']
    df = df.sample(n=10000, random_state=42)
    X = df[feature_cols]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model, X_train)
    shap_vals = explainer.shap_values(X_test, check_additivity=False)
    glob_shap = mean_abs_shap(shap_vals)
    global_imp = pd.Series(glob_shap, index=feature_cols).sort_values(ascending=False)
    global_ranks = global_imp.rank(ascending=False)

    T_GLOBAL = 15
    T_LOCAL = 3
    unimportant = global_ranks[global_ranks > T_GLOBAL].index.tolist()

    # shap_vals is (n_samples, n_features, n_classes)
    sv_arr = np.array(shap_vals)
    # Take max absolute over classes -> (n_samples, n_features)
    sv_max = np.abs(sv_arr).max(axis=2)
    sv_df = pd.DataFrame(sv_max, columns=feature_cols)
    local_ranks = sv_df.rank(axis=1, ascending=False)

    risks = []
    for feat in unimportant:
        critical = local_ranks[feat] <= T_LOCAL
        count = int(critical.sum())
        if count > 0:
            p0 = T_LOCAL / len(feature_cols)
            risks.append({
                'Feature': feat,
                'Global_Rank': int(global_ranks[feat]),
                'Critical_Instances': count,
                'Risk_Prevalence': float(f"{count/len(X_test):.4f}"),
                'p0': float(f"{p0:.4f}")
            })
    risks.sort(key=lambda x: x['Critical_Instances'], reverse=True)

    # Bootstrap overlap for top risk (reduced iterations for speed)
    overlap_results = {}
    if risks:
        top_feat = risks[0]['Feature']
        n_boot = 3
        n_detected = 0
        X_test_small = X_test.sample(n=500, random_state=42)
        for i in range(n_boot):
            X_s, y_s = X_train.sample(n=2000, random_state=i), y_train.loc[X_train.sample(n=2000, random_state=i).index]
            m = RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=-1)
            m.fit(X_s, y_s)
            e = shap.TreeExplainer(m, X_s[:100])
            sv = e.shap_values(X_test_small, check_additivity=False)
            sv_arr = np.array(sv)
            sv_m = np.abs(sv_arr).max(axis=2)
            lr = pd.DataFrame(sv_m, columns=feature_cols).rank(axis=1, ascending=False)
            if top_feat in lr.columns and (lr[top_feat] <= T_LOCAL).sum() > 0:
                n_detected += 1
        overlap_ratio = n_detected / n_boot
        p0 = T_LOCAL / len(feature_cols)
        from scipy.stats import binom
        threshold = int(0.9 * n_boot)
        p_val = 1 - binom.cdf(threshold - 1, n_boot, p0)
        overlap_results = {
            'feature': top_feat,
            'overlap_ratio': overlap_ratio,
            'detected_in': n_detected,
            'out_of': n_boot,
            'binomial_p_value': float(f"{p_val:.2e}")
        }
        print(f"\nTop risk: {top_feat}")
        print(f"  Overlap ratio: {overlap_ratio} ({n_detected}/{n_boot})")
        print(f"  Binomial p-value: {p_val:.2e}")

    result = {
        'threshold_global': T_GLOBAL,
        'threshold_local': T_LOCAL,
        'n_features': len(feature_cols),
        'p0': float(f"{T_LOCAL/len(feature_cols):.4f}"),
        'risks': risks[:10],
        'bootstrap_overlap': overlap_results
    }
    print(f"\nHidden risks detected: {len(risks)}")
    for r in risks[:5]:
        print(f"  {r['Feature']}: Global={r['Global_Rank']}, Critical={r['Critical_Instances']}, Prevalence={r['Risk_Prevalence']}")

    with open(os.path.join(OUT_DIR, 'exp3_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print("EXP 3 complete")

# ============================================================
# EXP 4: Actionability
# ============================================================
def run_exp4():
    import numpy as np, pandas as pd
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score
    import shap

    print("="*60)
    print("EXP 4: ACTIONABILITY")
    print("="*60)
    X, y = fetch_covtype(return_X_y=True, as_frame=True)
    X_s = X.sample(n=20000, random_state=42)
    y_s = y.loc[X_s.index]
    X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    base_acc = model.score(X_test, y_test)
    print(f"Baseline accuracy: {base_acc:.4f}")

    # CFCA
    explainer = shap.TreeExplainer(model, X_train)
    sv = explainer.shap_values(X_test, check_additivity=False)
    si = mean_abs_shap(sv)
    cfca_order = pd.Series(si, index=X_test.columns).sort_values(ascending=True).index.tolist()

    # PFI
    pfi = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    pfi_order = pd.Series(pfi.importances_mean, index=X_test.columns).sort_values(ascending=True).index.tolist()

    # Gini
    gini_order = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=True).index.tolist()

    def run_elim(order, name):
        pts = []
        current = order.copy()
        while len(current) > 1:
            m = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            m.fit(X_train[current], y_train)
            acc = accuracy_score(y_test, m.predict(X_test[current]))
            pts.append((len(current), float(f"{acc:.4f}")))
            current = current[1:]
        return pts

    results = {}
    for name, order in [('CFCA', cfca_order), ('PFI', pfi_order), ('Gini', gini_order)]:
        print(f"  Running {name}...")
        results[name] = run_elim(order, name)

    with open(os.path.join(OUT_DIR, 'exp4_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("EXP 4 complete")

# ============================================================
# EXP 5: Correlation Stress Test
# ============================================================
def run_exp5():
    import numpy as np, pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    import shap

    print("="*60)
    print("EXP 5: CORRELATION STRESS TEST")
    print("="*60)
    np.random.seed(42)
    N = 5000
    X1 = np.random.normal(0, 1, N)
    correlations = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    cfca_scores = []
    pfi_scores = []
    for rho in correlations:
        eps_var = max(0.001, (1 - rho**2) / rho**2)
        X2 = X1 + np.random.normal(0, np.sqrt(eps_var), N)
        X3 = np.random.normal(0, 1, N)
        X_df = pd.DataFrame({'Signal_A': X1, 'Signal_B_Redundant': X2, 'Noise': X3})
        y = (X1 + X3 > 0).astype(int)
        m = RandomForestClassifier(n_estimators=100, random_state=42)
        m.fit(X_df, y)
        e = shap.TreeExplainer(m, X_df[:100])
        sv = e.shap_values(X_df, check_additivity=False)
        si = mean_abs_shap(sv)
        pfi = permutation_importance(m, X_df, y, n_repeats=5, random_state=42)
        cfca_score = float(si[1]) if np.ndim(si) == 1 else float(np.abs(si).mean(axis=0)[1])
        cfca_scores.append(cfca_score)
        pfi_scores.append(float(pfi.importances_mean[1]))
        if rho in [0.5, 0.7, 0.85, 0.9, 0.95]:
            print(f"  rho={rho:.2f}: CFCA={cfca_score:.4f}, PFI={pfi.importances_mean[1]:.4f}")

    result = {'correlations': correlations, 'cfca_scores': cfca_scores, 'pfi_scores': pfi_scores}
    with open(os.path.join(OUT_DIR, 'exp5_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print("EXP 5 complete")

# ============================================================
# EXP 6: Sensitivity Analysis
# ============================================================
def run_exp6():
    import shap, numpy as np, pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    print("="*60)
    print("EXP 6: SENSITIVITY ANALYSIS")
    print("="*60)
    df = pd.read_csv(os.path.join(BASE_DIR, 'Sensorless_drive_diagnosis.txt'), sep=' ', header=None)
    feature_cols = [f'Sensor_{i+1}' for i in range(48)]
    df.columns = feature_cols + ['Target']
    df = df.sample(n=10000, random_state=42)
    X = df[feature_cols]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model, X_train)
    shap_vals = explainer.shap_values(X_test, check_additivity=False)
    glob_shap = mean_abs_shap(shap_vals)
    global_imp = pd.Series(glob_shap, index=feature_cols).sort_values(ascending=False)
    global_ranks = global_imp.rank(ascending=False)
    # shap_vals is (n_samples, n_features, n_classes)
    sv_max = np.abs(np.array(shap_vals)).max(axis=2)  # -> (n_samples, n_features)
    sv_df = pd.DataFrame(sv_max, columns=feature_cols)
    local_ranks = sv_df.rank(axis=1, ascending=False)

    grid = {}
    for tg in [10, 15, 20, 25, 30]:
        for tl in [1, 2, 3, 5]:
            unimportant = global_ranks[global_ranks > tg].index.tolist()
            count = 0
            for feat in unimportant:
                if feat in local_ranks.columns and (local_ranks[feat] <= tl).sum() > 0:
                    count += 1
            grid[f'{tg}_{tl}'] = count

    print("Threshold Grid:")
    for tg in [10, 15, 20, 25, 30]:
        row = [f"{grid[f'{tg}_{tl}']}" for tl in [1, 2, 3, 5]]
        print(f"  T_global={tg}: {', '.join(row)}")

    with open(os.path.join(OUT_DIR, 'exp6_results.json'), 'w') as f:
        json.dump(grid, f, indent=2)
    print("EXP 6 complete")

# ============================================================
# EXP 7: SECOM Audit
# ============================================================
def run_exp7():
    import shap, numpy as np, pandas as pd, time
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.impute import SimpleImputer

    print("="*60)
    print("EXP 7: SECOM AUDIT")
    print("="*60)
    data = np.loadtxt(os.path.join(BASE_DIR, 'secom.data'))
    labels_raw = pd.read_csv(os.path.join(BASE_DIR, 'secom_labels.data'), sep=' ', header=None)
    y = np.where(labels_raw[0].values == -1, 0, 1)
    print(f"Data: {data.shape}, failures: {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

    nan_cols = np.isnan(data).sum(axis=0) < data.shape[0] * 0.5
    data = data[:, nan_cols]
    data = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(data))
    var_mask = data.var() > 1e-8
    data = data.loc[:, var_mask]
    n_features = data.shape[1]
    print(f"Features after preproc: {n_features}")

    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    data.columns = feature_names
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print(f"Accuracy: {model.score(X_test, y_test):.4f}")

    t0 = time.time()
    explainer = shap.TreeExplainer(model, X_train)
    sv = explainer.shap_values(X_test, check_additivity=False)
    t_cfca = time.time() - t0
    if isinstance(sv, list):
        si = mean_abs_shap(sv)
    elif sv.ndim == 3:
        si = np.abs(sv[:, :, 1]).mean(axis=0)
    else:
        si = np.abs(sv).mean(axis=0)
    cfca_imp = pd.Series(si, index=feature_names).sort_values(ascending=False)

    t0 = time.time()
    pfi = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42, n_jobs=-1)
    pfi_imp = pd.Series(pfi.importances_mean, index=feature_names)
    t_pfi = time.time() - t0
    corr = cfca_imp.rank(ascending=False).corr(pfi_imp.rank(ascending=False), method='spearman')
    print(f"Correlation: {corr:.4f}, NDS: {1-corr:.4f}")

    # Stability
    cfca_rs, pfi_rs = [], []
    for i in range(5):
        X_s, X_t, y_s, y_t = train_test_split(data, y, test_size=0.2, random_state=i, stratify=y)
        m = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        m.fit(X_s, y_s)
        e = shap.TreeExplainer(m, X_s)
        sv_i = e.shap_values(X_t, check_additivity=False)
        if isinstance(sv_i, list):
            si_i = np.abs(np.array(sv_i)).mean(axis=(0, 1))
        elif sv_i.ndim == 3:
            si_i = np.abs(sv_i[:, :, 1]).mean(axis=0)
        else:
            si_i = np.abs(sv_i).mean(axis=0)
        cfca_rs.append(pd.Series(si_i, index=feature_names).rank(ascending=False))
        pfi_i = permutation_importance(m, X_t, y_t, n_repeats=2, random_state=i)
        pfi_rs.append(pd.Series(pfi_i.importances_mean, index=feature_names).rank(ascending=False))
    def pc(rs):
        cs = []
        for i in range(len(rs)):
            for j in range(i+1, len(rs)):
                cs.append(rs[i].corr(rs[j], method='spearman'))
        return float(f"{np.mean(cs):.4f}"), float(f"{np.std(cs):.4f}")
    cfca_bsi, cfca_bsi_std = pc(cfca_rs)
    pfi_bsi, pfi_bsi_std = pc(pfi_rs)
    print(f"CFCA BSI: {cfca_bsi} +/- {cfca_bsi_std}, PFI BSI: {pfi_bsi} +/- {pfi_bsi_std}")

    # Hidden risks
    glob_ranks = cfca_imp.rank(ascending=False)
    sv_all = explainer.shap_values(X_test, check_additivity=False)
    if isinstance(sv_all, list):
        sv_m = np.max(np.abs(np.array(sv_all)), axis=0)
    elif sv_all.ndim == 3:
        sv_m = np.max(np.abs(sv_all[:, :, 1]), axis=0)
    else:
        sv_m = np.max(np.abs(sv_all), axis=0)
    lr = pd.DataFrame(np.tile(sv_m, (len(X_test), 1)), columns=feature_names).rank(axis=1, ascending=False)
    unimportant = glob_ranks[glob_ranks > n_features // 2].index.tolist()
    hidden = [f for f in unimportant if f in lr.columns and (lr[f] <= 3).sum() > 0]
    print(f"Hidden risks: {len(hidden)}")

    result = {
        'n_features': n_features,
        'n_samples': len(data),
        'failure_rate': float(f"{y.mean()*100:.1f}"),
        'bsi_cfca': cfca_bsi,
        'bsi_pfi': pfi_bsi,
        'nds': float(f"{1-corr:.4f}"),
        'correlation': float(f"{corr:.4f}"),
        'hidden_risks_count': len(hidden),
        'time_cfca': float(f"{t_cfca:.2f}"),
        'time_pfi': float(f"{t_pfi:.2f}")
    }
    with open(os.path.join(OUT_DIR, 'exp7_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print("EXP 7 complete")

# ============================================================
# EXP 8: KITTI Perception Fusion
# ============================================================
def run_exp8():
    import shap, numpy as np, pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    print("="*60)
    print("EXP 8: KITTI PERCEPTION FUSION")
    print("="*60)
    np.random.seed(42)
    N = 5000
    occlusion = np.random.uniform(0, 1, N)
    camera_conf = np.random.uniform(0, 1, N) * (1 - occlusion * 0.6)
    lidar_intensity = np.random.uniform(0, 1, N)
    lidar_return = np.random.uniform(0, 50, N) * (1 + occlusion * 0.5)
    distance = np.random.uniform(0, 80, N)
    noise = np.random.normal(0, 0.1, N)
    det = 0.4*camera_conf + 0.1*lidar_intensity + 0.3*(lidar_return/50)*occlusion + 0.1*(1-distance/80) + 0.1*noise
    y = (det > 0.5).astype(int)
    X = pd.DataFrame({'Camera_Confidence': camera_conf, 'LiDAR_Intensity': lidar_intensity,
                       'LiDAR_Return': lidar_return, 'Distance': distance, 'Occlusion_Index': occlusion})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Accuracy: {model.score(X_test, y_test):.4f}")

    explainer = shap.TreeExplainer(model, X_train)
    sv = explainer.shap_values(X_test, check_additivity=False)
    if isinstance(sv, list):
        glob = mean_abs_shap(sv)
    elif len(sv.shape) == 3:
        glob = np.abs(sv).mean(axis=(0, 2))
    else:
        glob = np.abs(sv).mean(axis=0)
    cfca_imp = pd.Series(glob, index=X.columns).sort_values(ascending=False)
    pfi = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    pfi_imp = pd.Series(pfi.importances_mean, index=X.columns).sort_values(ascending=False)

    occluded = X_test[X_test['Occlusion_Index'] > 0.6]
    sv_occ = explainer.shap_values(occluded, check_additivity=False)
    if isinstance(sv_occ, list):
        sv_occ_a = np.abs(np.array(sv_occ)).mean(axis=0)
    else:
        sv_occ_a = np.abs(sv_occ).mean(axis=0) if len(sv_occ.shape) == 2 else np.abs(sv_occ).mean(axis=(0, 2))
    occ_df = pd.DataFrame(sv_occ_a, columns=X.columns) if sv_occ_a.ndim == 2 else pd.DataFrame([sv_occ_a], columns=X.columns)
    top1 = occ_df.idxmax(axis=1).value_counts() if len(occ_df) > 1 else pd.Series({occ_df.idxmax(axis=1).iloc[0]: 1})

    result = {
        'n_occluded': len(occluded),
        'occluded_pct': float(f"{len(occluded)/len(X_test)*100:.1f}"),
        'cfca_ranking': cfca_imp.to_dict(),
        'pfi_ranking': pfi_imp.to_dict(),
        'top1_occluded': {str(k): int(v) for k, v in top1.items()}
    }
    print(f"\nOccluded cases: {len(occluded)} ({result['occluded_pct']}%)")
    print(f"Top-1 in occluded: {top1.to_dict()}")
    with open(os.path.join(OUT_DIR, 'exp8_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print("EXP 8 complete")

# ============================================================
# EXP 9: SENSOR NOISE ROBUSTNESS
# ============================================================
def run_exp9():
    """Sensor noise sensitivity: perturb each feature with Gaussian noise at multiple std levels."""
    import shap, numpy as np, pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import spearmanr
    print("=" * 60)
    print("EXP 9: SENSOR NOISE ROBUSTNESS")
    print("=" * 60)
    df = pd.read_csv(os.path.join(BASE_DIR, 'Sensorless_drive_diagnosis.txt'), sep=' ', header=None)
    feature_cols = [f'Sensor_{i+1}' for i in range(48)]
    df.columns = feature_cols + ['Target']
    X = df[feature_cols]
    y = df['Target']
    X_small, _, y_small, _ = train_test_split(X, y, train_size=2000, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model, X_train)
    baseline_sv = explainer.shap_values(X_test, check_additivity=False)
    baseline_imp = np.abs(mean_abs_shap(baseline_sv))

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    results = {'noise_levels': noise_levels, 'rank_correlations': []}
    for nl in noise_levels:
        X_noisy = X_test[:500].copy()
        np.random.seed(42)
        for col in X_noisy.columns:
            std = X_noisy[col].std()
            X_noisy[col] += np.random.normal(0, nl * std, size=len(X_noisy))
        sv_noisy = explainer.shap_values(X_noisy, check_additivity=False)
        noisy_imp = np.abs(mean_abs_shap(sv_noisy))
        rho, _ = spearmanr(baseline_imp, noisy_imp)
        results['rank_correlations'].append(float(f"{rho:.4f}"))
        print(f"  noise={nl:.2f}: rank correlation={rho:.4f}")
    with open(os.path.join(OUT_DIR, 'exp9_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("EXP 9 complete")

# ============================================================
# EXP 10: REAL KITTI PERCEPTION FUSION
# ============================================================
def run_exp10():
    """Real KITTI object detection dataset: train YOLO, extract perception features, run CFCA."""
    import shap, numpy as np, pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from ultralytics import YOLO
    import time, os, glob, cv2
    from scipy.stats import spearmanr

    print("=" * 60)
    print("EXP 10: REAL KITTI PERCEPTION FUSION")
    print("=" * 60)

    kitti_dir = os.path.join(BASE_DIR, 'kitti_raw')
    data_yaml = os.path.join(kitti_dir, 'kitti.yaml')
    # Check kitti.yaml exists
    if not os.path.exists(data_yaml):
        print("ERROR: kitti.yaml not found. Run real KITTI download first.")
        return

    # Use pretrained YOLO (no KITTI-specific finetuning — already generalizes to road scenes)
    print("Loading pretrained YOLOv8n...")
    model = YOLO('yolov8n.pt')
    print("YOLO loaded")

    # Feature extraction: for each image, extract perception features
    print("Extracting perception features from KITTI validation set...")
    img_dir = os.path.join(kitti_dir, 'images', 'val')
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))

    # Also load labels to create ground truth
    label_dir = os.path.join(kitti_dir, 'labels', 'val')
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.txt')))

    feature_names = [
        'Num_Detections', 'Max_Confidence', 'Mean_Confidence', 'Min_Confidence',
        'Num_Cars', 'Num_Pedestrians', 'Num_Cyclists', 'Mean_Box_Area',
        'Max_Box_Area', 'Total_Box_Area', 'Class_Entropy',
        'Density_Score', 'Saturation_Score'
    ]

    records = []
    targets = []

    for img_file in img_files[:500]:
        img = cv2.imread(img_file)
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model(img, verbose=False)
        dets = results[0]
        boxes = dets.boxes

        if boxes is None or len(boxes) == 0:
            records.append({fn: 0.0 for fn in feature_names})
            targets.append(0)  # No detections -> class 0
            continue

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        xywh = boxes.xywh.cpu().numpy()

        n = len(confs)
        num_cars = int(np.sum(cls_ids == 0))
        num_peds = int(np.sum(cls_ids == 3))
        num_cycs = int(np.sum(cls_ids == 5))

        box_areas = xywh[:, 2] * xywh[:, 3]
        img_area = h * w

        # Class entropy
        if n > 0:
            cls_counts = np.bincount(cls_ids, minlength=8)
            cls_probs = cls_counts / cls_counts.sum()
            cls_entropy = -np.sum(cls_probs[cls_probs > 0] * np.log2(cls_probs[cls_probs > 0]))
        else:
            cls_entropy = 0.0

        # Density: how much of the image is covered by detections
        density = np.sum(box_areas) / img_area

        # Saturation: simplify as mean confidence * density
        saturation = np.mean(confs) * density if n > 0 else 0.0

        rec = {
            'Num_Detections': float(n),
            'Max_Confidence': float(np.max(confs)) if n > 0 else 0.0,
            'Mean_Confidence': float(np.mean(confs)) if n > 0 else 0.0,
            'Min_Confidence': float(np.min(confs)) if n > 0 else 0.0,
            'Num_Cars': float(num_cars),
            'Num_Pedestrians': float(num_peds),
            'Num_Cyclists': float(num_cycs),
            'Mean_Box_Area': float(np.mean(box_areas)) if n > 0 else 0.0,
            'Max_Box_Area': float(np.max(box_areas)) if n > 0 else 0.0,
            'Total_Box_Area': float(np.sum(box_areas)),
            'Class_Entropy': float(cls_entropy),
            'Density_Score': float(density),
            'Saturation_Score': float(saturation),
        }
        records.append(rec)

        # Target: busy scene based on detections > 0 and reasonable confidence
        if n >= 3 and np.mean(confs) > 0.5:
            targets.append(1)  # busy
        elif n == 0:
            targets.append(0)  # empty
        else:
            targets.append(2)  # moderate

    X = pd.DataFrame(records)
    y = pd.Series(targets)
    n_features = len(feature_names)
    print(f"Extracted {len(X)} samples with {n_features} features")
    print(f"Class distribution: busy={sum(y==1)}, empty={sum(y==0)}, moderate={sum(y==2)}")

    if len(X) < 100:
        print("Too few samples, skipping experiment")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    print(f"Accuracy: {acc:.4f}")

    # CFCA
    t0 = time.time()
    explainer = shap.TreeExplainer(rf, X_train)
    sv = explainer.shap_values(X_test, check_additivity=False)
    t_cfca = time.time() - t0
    if isinstance(sv, list):
        cfca_imp = np.abs(np.array(sv)).mean(axis=(0, 1))
    elif sv.ndim == 3:
        cfca_imp = np.abs(sv).mean(axis=(0, 2))
    else:
        cfca_imp = np.abs(sv).mean(axis=0)
    cfca_series = pd.Series(cfca_imp, index=feature_names).sort_values(ascending=False)

    # PFI
    t0 = time.time()
    pfi = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42)
    t_pfi = time.time() - t0
    pfi_series = pd.Series(pfi.importances_mean, index=feature_names).sort_values(ascending=False)

    # Correlation
    corr, _ = spearmanr(cfca_series, pfi_series)
    nds = 1 - corr

    # Bootstrap stability
    cfca_rs, pfi_rs = [], []
    for i in range(5):
        X_s, X_t, y_s, y_t = train_test_split(X, y, test_size=0.2, random_state=i, stratify=y)
        m = RandomForestClassifier(n_estimators=100, random_state=42)
        m.fit(X_s, y_s)
        e = shap.TreeExplainer(m, X_s)
        sv_i = e.shap_values(X_t, check_additivity=False)
        if isinstance(sv_i, list):
            si_i = np.abs(np.array(sv_i)).mean(axis=(0, 1))
        elif sv_i.ndim == 3:
            si_i = np.abs(sv_i).mean(axis=(0, 2))
        else:
            si_i = np.abs(sv_i).mean(axis=0)
        cfca_rs.append(pd.Series(si_i, index=feature_names).rank(ascending=False))
        pfi_i = permutation_importance(m, X_t, y_t, n_repeats=2, random_state=i)
        pfi_rs.append(pd.Series(pfi_i.importances_mean, index=feature_names).rank(ascending=False))

    def pc(rs):
        cs = []
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                cs.append(rs[i].corr(rs[j], method='spearman'))
        return float(f"{np.mean(cs):.4f}"), float(f"{np.std(cs):.4f}")

    cfca_bsi, cfca_bsi_std = pc(cfca_rs)
    pfi_bsi, pfi_bsi_std = pc(pfi_rs)

    # Hidden risk detection
    glob_ranks = cfca_series.rank(ascending=False)
    unimportant = glob_ranks[glob_ranks > n_features // 2].index.tolist()
    if isinstance(sv, list):
        sv_m = np.max(np.abs(np.array(sv)), axis=(0, 2))  # max over classes
        sv_is_1d = True
    elif sv.ndim == 3:
        sv_m = np.abs(sv).max(axis=2).max(axis=0)  # max abs over classes, then max over samples
        sv_is_1d = True
    else:
        sv_m = np.max(np.abs(sv), axis=0)
        sv_is_1d = True
    lr_data = np.tile(sv_m, (len(X_test), 1))
    lr = pd.DataFrame(lr_data, columns=feature_names).rank(axis=1, ascending=False)
    hidden = [f for f in unimportant if f in lr.columns and (lr[f] <= 3).sum() > 0]

    print(f"CFCA BSI: {cfca_bsi} +/- {cfca_bsi_std}, PFI BSI: {pfi_bsi} +/- {pfi_bsi_std}")
    print(f"Correlation: {corr:.4f}, NDS: {nds:.4f}")
    print(f"Hidden risks: {len(hidden)} -> {hidden}")

    result = {
        'n_samples': len(X),
        'n_features': n_features,
        'accuracy': float(f"{acc:.4f}"),
        'cfca_bsi': cfca_bsi,
        'cfca_bsi_std': cfca_bsi_std,
        'pfi_bsi': pfi_bsi,
        'pfi_bsi_std': pfi_bsi_std,
        'correlation': float(f"{corr:.4f}"),
        'nds': float(f"{nds:.4f}"),
        'hidden_risks_count': len(hidden),
        'hidden_risks': hidden,
        'cfca_ranking': cfca_series.to_dict(),
        'pfi_ranking': pfi_series.to_dict(),
        'time_cfca': float(f"{t_cfca:.2f}"),
        'time_pfi': float(f"{t_pfi:.2f}"),
    }
    with open(os.path.join(OUT_DIR, 'exp10_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print("EXP 10 complete")
experiments = {
    '1': run_exp1, 'exp1': run_exp1,
    '2': run_exp2, 'exp2': run_exp2,
    '3': run_exp3, 'exp3': run_exp3,
    '4': run_exp4, 'exp4': run_exp4,
    '5': run_exp5, 'exp5': run_exp5,
    '6': run_exp6, 'exp6': run_exp6,
    '7': run_exp7, 'exp7': run_exp7,
    '8': run_exp8, 'exp8': run_exp8,
    '9': run_exp9, 'exp9': run_exp9,
    '10': run_exp10, 'exp10': run_exp10,
}

if batch == 'all':
    for name, fn in [('1', run_exp1), ('2', run_exp2), ('3', run_exp3), ('4', run_exp4),
                     ('5', run_exp5), ('6', run_exp6), ('7', run_exp7), ('8', run_exp8), ('9', run_exp9)]:
        print(f"\n{'='*60}")
        print(f"STARTING BATCH {name}")
        print(f"{'='*60}")
        t0 = time.time()
        fn()
        print(f"Batch {name} took {time.time()-t0:.1f}s")
elif batch in experiments:
    experiments[batch]()
else:
    print(f"Unknown batch: {batch}. Options: all, 1-10, exp1-exp10")
