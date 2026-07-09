import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)

# ================================================================
# Figure 1: Narrative Disconnect (CFCA vs PFI bar chart)
# ================================================================
def fig1_narrative_disconnect():
    d = load_json('exp1_results.json')
    t1 = d['table1']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, label, feats, scores, color in [
        (axes[0], 'CFCA (Ours)', t1['cfca_features'], t1['cfca_scores'], '#2c3e50'),
        (axes[1], 'Permutation Feature Importance (PFI)', t1['pfi_features'], t1['pfi_scores'], '#c0392b'),
    ]:
        y_pos = np.arange(len(feats))
        ax.barh(y_pos, scores, color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats)
        ax.invert_yaxis()
        ax.set_xlabel('Global Importance')
        ax.set_title(label, fontsize=12)
        for i, v in enumerate(scores):
            ax.text(v + 0.0005, i, f'{v:.4f}', va='center', fontsize=8)

    fig.suptitle(f'Figure 1: Narrative Disconnect — CFCA vs PFI (Spearman ρ={d["spearman_correlation"]:.4f}, NDS={d["nds"]:.4f})', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figure1_narrative_disconnect.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('Figure 1 saved')

# ================================================================
# Figure 2: Bootstrap Stability (bar chart with error bars)
# ================================================================
def fig2_stability():
    d = load_json('exp2_results.json')
    methods = ['CFCA (Ours)', 'Permutation Feature\nImportance (PFI)']
    means = [d['cfca_mean'], d['pfi_mean']]
    stds  = [d['cfca_std'], d['pfi_std']]
    colors = ['#2c3e50', '#c0392b']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, means, yerr=stds, capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_ylabel("Spearman's ρ (Rank Stability)")
    ax.set_title('Figure 2: Bootstrap Stability Comparison', fontsize=13)
    ax.set_ylim(0, 1.15)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', fontsize=10)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Threshold (ρ=0.7)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figure2_stability.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('Figure 2 saved')

# ================================================================
# Figure 3: Actionability (Recursive Feature Elimination curves)
# ================================================================
def fig3_actionability():
    d = load_json('exp4_results.json')
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {'CFCA': '#2c3e50', 'PFI': '#c0392b'}
    labels = {'CFCA': 'CFCA (Ours)', 'PFI': 'Permutation Feature Importance (PFI)'}

    for method in ['CFCA', 'PFI']:
        data = d[method]
        xs = [p[0] for p in data]
        ys = [p[1] for p in data]
        ax.plot(xs, ys, marker='o', markersize=4, linewidth=2, color=colors[method], label=labels[method])

    ax.set_xlabel('Number of Features Retained', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Figure 3: Actionability — Recursive Feature Elimination on Covertype Dataset', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(55, 0)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figure3_actionability.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('Figure 3 saved')

# ================================================================
# Figure 4: Class-Specific CFCA (Wine dataset)
# ================================================================
def fig4_class_specific():
    d = load_json('exp1_results.json')
    fig, ax = plt.subplots(figsize=(10, 6))
    cfca_feats = d['table1']['cfca_features']
    cfca_scores = d['table1']['cfca_scores']
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cfca_feats)))
    bars = ax.barh(range(len(cfca_feats)), cfca_scores, color=colors, alpha=0.8)
    ax.set_yticks(range(len(cfca_feats)))
    ax.set_yticklabels(cfca_feats)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP| (Global Importance)')
    ax.set_title('Figure 4: Class-Aggregated CFCA Global Importance — Sensorless Drive Diagnosis Dataset', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figure4_class_specific.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('Figure 4 saved')

# ================================================================
# Figure 5: Correlation Stress Test (CFCA vs PFI across rho)
# ================================================================
def fig5_correlation_stress():
    d = load_json('exp5_results.json')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d['correlations'], d['cfca_scores'], marker='o', linewidth=2, color='#2c3e50', label='CFCA (Ours)')
    ax.plot(d['correlations'], d['pfi_scores'], marker='s', linewidth=2, color='#c0392b', label='Permutation Feature Importance (PFI)')
    ax.set_xlabel('Pairwise Feature Correlation (|ρ|)')
    ax.set_ylabel('Mean Absolute Importance')
    ax.set_title('Figure 5: Correlation Stress Test — CFCA vs PFI Under Increasing Feature Redundancy', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figure5_correlation_stress.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('Figure 5 saved')

# ================================================================
# Figure 6: Sensitivity Analysis Heatmap
# ================================================================
def fig6_sensitivity_heatmap():
    d = load_json('exp6_results.json')
    t_globals = sorted(set(int(k.split('_')[0]) for k in d.keys()))
    t_locals  = sorted(set(int(k.split('_')[1]) for k in d.keys()))
    grid = np.zeros((len(t_globals), len(t_locals)), dtype=int)
    for i, tg in enumerate(t_globals):
        for j, tl in enumerate(t_locals):
            key = f'{tg}_{tl}'
            grid[i, j] = d.get(key, 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_xticks(range(len(t_locals)))
    ax.set_xticklabels([str(t) for t in t_locals])
    ax.set_yticks(range(len(t_globals)))
    ax.set_yticklabels([str(t) for t in t_globals])
    ax.set_xlabel('Local Threshold (T_local)')
    ax.set_ylabel('Global Threshold (T_global)')
    ax.set_title('Figure 6: Sensitivity Analysis — Hidden Risks Detected by Threshold', fontsize=12)
    for i in range(len(t_globals)):
        for j in range(len(t_locals)):
            ax.text(j, i, str(grid[i, j]), ha='center', va='center', color='black', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Number of Hidden Risks Detected')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figure6_sensitivity_heatmap.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('Figure 6 saved')

# ================================================================
# Figure 7: SECOM Semiconductor Audit
# ================================================================
def fig7_secom_audit():
    d = load_json('exp7_results.json')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: BSI comparison
    ax = axes[0]
    methods = ['CFCA', 'PFI']
    bs = [d['bsi_cfca'], d.get('bsi_pfi', 0)]
    colors = ['#2c3e50', '#c0392b']
    ax.bar(methods, bs, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel("Spearman's ρ (Rank Stability)")
    ax.set_title('Panel A: Bootstrap Stability')
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(bs):
        if not np.isnan(v):
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
        else:
            ax.text(i, 0.1, 'NaN', ha='center', fontsize=10, color='red')

    # Panel B: NDS and Correlation
    ax = axes[1]
    metrics = ['NDS', 'Correlation']
    vals = [d['nds'], d['correlation']]
    ax.bar(metrics, vals, color=['#e67e22', '#8e44ad'], alpha=0.8, edgecolor='black')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Score')
    ax.set_title(f'Panel B: NDS & Correlation (n={d["n_features"]} features)')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05 if v >= 0 else v - 0.08, f'{v:.4f}', ha='center', fontsize=10)

    fig.suptitle(f'Figure 7: SECOM Semiconductor Manufacturing Audit (Failure Rate: {d["failure_rate"]}%)', fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figure7_secom_audit.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('Figure 7 saved')

# ================================================================
# Figure 8: Real KITTI Sensor Fusion
# ================================================================
def fig8_kitti():
    d = load_json('exp10_results.json')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Feature importance ranking
    ax = axes[0]
    feats = list(d['cfca_ranking'].keys())
    cfca_v = [d['cfca_ranking'][f] for f in feats]
    pfi_v  = list(d['pfi_ranking'].values())
    x = np.arange(len(feats))
    w = 0.35
    ax.bar(x - w/2, cfca_v, w, color='#2c3e50', alpha=0.8, label='CFCA')
    ax.bar(x + w/2, pfi_v, w, color='#c0392b', alpha=0.8, label='PFI')
    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=20, fontsize=8)
    ax.set_ylabel('Global Importance')
    ax.set_title('Panel A: Feature Importance Comparison')
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # Panel B: BSI comparison
    ax = axes[1]
    methods = ['CFCA', 'PFI']
    bs = [d['cfca_bsi'], d['pfi_bsi']]
    colors = ['#2c3e50', '#c0392b']
    ax.bar(methods, bs, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel("Spearman's ρ (Rank Stability)")
    ax.set_title(f'Panel B: Stability & NDS={d["nds"]:.4f}')
    ax.set_ylim(0, 1.1)
    for i, v in enumerate(bs):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)

    fig.suptitle(f'Figure 8: Real KITTI Perception Fusion — {d["n_samples"]} images, YOLOv8n (Accuracy: {d["accuracy"]})', fontsize=11, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figure8_kitti.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('Figure 8 saved')

# ================================================================
# Figure 9: Sensor Noise Robustness
# ================================================================
def fig9_noise_robustness():
    path = os.path.join(RESULTS_DIR, 'exp9_results.json')
    if not os.path.exists(path):
        print('Figure 9 skipped: exp9_results.json not found')
        return
    d = json.load(open(path))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d['noise_levels'], d['rank_correlations'], marker='o', linewidth=2, color='#2c3e50')
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Stability Threshold (ρ=0.7)')
    ax.set_xlabel('Gaussian Noise Level (σ multiplier)')
    ax.set_ylabel("Spearman's ρ (Rank Correlation vs Clean)")
    ax.set_title('Figure 9: Sensor Noise Robustness — CFCA Stability Under Input Perturbation', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figure9_noise_robustness.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('Figure 9 saved')

# ================================================================
# Figure 10: Ablation Study — Bootstrap vs Full CFCA
# ================================================================
def fig10_ablation():
    d = load_json('exp11_results.json')
    stages = d['stages']
    nds = d['summary']['nds']

    bootstrap_bsi = stages['stage3_bootstrap']['bsi']
    bootstrap_std = stages['stage3_bootstrap']['bsi_std']
    cfca_bsi = stages['stage4_full_cfca']['bsi']
    cfca_std = stages['stage4_full_cfca']['bsi_std']
    hidden = stages['stage4_full_cfca']['n_hidden']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: BSI Comparison (Bootstrap vs Full CFCA)
    x = [0, 1]
    bsi = [bootstrap_bsi, cfca_bsi]
    errs = [bootstrap_std, cfca_std]
    colors = ['#e67e22', '#27ae60']
    bars = axes[0].bar(x, bsi, yerr=errs, capsize=8, color=colors, alpha=0.9, edgecolor='white', linewidth=2, width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Bootstrap\nAggregation', 'Full CFCA'], fontsize=11)
    axes[0].set_ylabel('Bootstrap Stability Index (BSI)', fontsize=11)
    axes[0].set_title('(a) Stability: Bootstrap vs Full CFCA', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, (v, s) in enumerate(zip(bsi, errs)):
        axes[0].text(i, v + s + 0.03, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    gain = cfca_bsi - bootstrap_bsi
    pct = (gain / bootstrap_bsi) * 100
    axes[0].annotate(f'+{gain:.3f}\n(+{pct:.1f}%)', xy=(1, cfca_bsi + cfca_std + 0.03),
                     xytext=(1.35, cfca_bsi + cfca_std + 0.12),
                     fontsize=9, fontweight='bold', color='#27ae60',
                     arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))
    axes[0].axhline(y=0.9, color='red', linestyle='--', alpha=0.4, label='Safety threshold (0.9)')
    axes[0].legend(fontsize=8, loc='lower right')

    # Panel B: Hidden Risks Detected
    axes[1].bar([0], [hidden], color='#27ae60', alpha=0.9, edgecolor='white', linewidth=2, width=0.5)
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(['Full CFCA'], fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title(f'(b) Hidden Risks Detected', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, hidden + 3)
    axes[1].text(0, hidden + 0.3, str(hidden), ha='center', fontsize=14, fontweight='bold')
    risk_names = stages['stage4_full_cfca']['hidden_risks']
    if risk_names:
        risk_text = ', '.join(risk_names[:5])
        if len(risk_names) > 5:
            risk_text += f' (+{len(risk_names)-5} more)'
        axes[1].text(0, hidden * 0.5, risk_text, ha='center', fontsize=7, style='italic', wrap=True)

    # Panel C: Ranking Disagreement (NDS)
    axes[2].bar([0], [nds], color='#e74c3c', alpha=0.9, edgecolor='white', linewidth=2, width=0.5)
    axes[2].set_xticks([0])
    axes[2].set_xticklabels(['CFCA vs PFI'], fontsize=11)
    axes[2].set_ylabel('Narrative Disconnect Score', fontsize=11)
    axes[2].set_title('(c) Ranking Disagreement', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    axes[2].text(0, nds + 0.03, f'{nds:.3f}', ha='center', fontsize=14, fontweight='bold')
    corr = d['summary']['cfca_pfi_correlation']
    axes[2].text(0, nds * 0.5, f'Spearman ρ = {corr:.3f}', ha='center', fontsize=9, style='italic')

    fig.suptitle(f'Figure 10: Ablation — Bootstrap Aggregation vs Full CFCA (NDS={nds:.3f})', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figure10_ablation.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('Figure 10 saved')

# ================================================================
# Figure 11: Benchmark Comparison (CFCA vs PFI vs LIME vs FLocalX)
# ================================================================
def fig11_benchmark():
    d = load_json('exp15_results.json')
    methods = d['methods']

    method_labels = []
    bsi_values = []
    bsi_stds = []
    colors = []

    color_map = {
        'cfca': '#27ae60',
        'pfi': '#95a5a6',
        'lime': '#3498db',
        'flocalx': '#e67e22',
        'gleams': '#9b59b6',
        'glocalx': '#e74c3c',
    }
    display_map = {
        'cfca': 'CFCA (Ours)',
        'pfi': 'PFI',
        'lime': 'LIME',
        'flocalx': 'FLocalX',
        'gleams': 'GLEAMS',
        'glocalx': 'GLocalX',
    }

    for key in ['cfca', 'pfi', 'lime', 'flocalx', 'gleams', 'glocalx']:
        m = methods.get(key, {})
        if m.get('available', False):
            method_labels.append(display_map[key])
            bsi_values.append(m['bsi'])
            bsi_stds.append(m['bsi_std'])
            colors.append(color_map[key])
        else:
            method_labels.append(display_map[key] + '\n(N/A)')
            bsi_values.append(0)
            bsi_stds.append(0)
            colors.append('#cccccc')

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(method_labels, bsi_values, yerr=bsi_stds, capsize=5,
                  color=colors, edgecolor='white', linewidth=1.2)

    ax.axhline(y=0.9, color='#e74c3c', linestyle='--', linewidth=1.5, label='Safety threshold (0.9)')
    ax.set_ylabel('Bootstrap Stability Index (BSI)', fontsize=12)
    ax.set_title(f'Figure 11: Benchmark Comparison — CFCA vs Alternative Methods\n'
                 f'(Sensorless Drive Diagnosis, {d["n_features"]} features, {d["n_classes"]} classes, '
                 f'model accuracy = {d["model_accuracy"]:.2%})', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=10)

    for bar, val, std in zip(bars, bsi_values, bsi_stds):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + std + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar, label in zip(bars, method_labels):
        if 'N/A' in label:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.05,
                    'N/A', ha='center', va='bottom', fontsize=9, fontstyle='italic', color='#666')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figure11_benchmark.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('Figure 11 saved')

# ================================================================
# RUN ALL
# ================================================================
if __name__ == '__main__':
    print('Generating all figures...')
    fig1_narrative_disconnect()
    fig2_stability()
    fig3_actionability()
    fig4_class_specific()
    fig5_correlation_stress()
    fig6_sensitivity_heatmap()
    fig7_secom_audit()
    fig8_kitti()
    fig9_noise_robustness()
    fig10_ablation()
    fig11_benchmark()
    print(f'\nAll figures saved to {OUT_DIR}/')
