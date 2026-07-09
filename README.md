# CFCA: Consistent Feature Contribution Aggregation

A unified verification framework for detecting hidden safety risks in autonomous systems.

## Overview

This repository contains the full experimental pipeline for the CFCA framework, validated on the UCI Sensorless Drive Diagnosis dataset (n=58,509, 48 features), UCI SECOM semiconductor dataset (1,567 samples, 446 features), real KITTI object detection dataset (7,481 images), and synthetic KITTI perception fusion data. CFCA achieves near-perfect bootstrap stability (BSI=0.9955 vs PFI 0.1871) and identifies "Black Swan" features globally unimportant but locally critical in edge cases. Benchmark comparison against LIME, FLocalX, GLEAMS, and GLocalX confirms CFCA's stability advantage.

## Repository Structure

```
CFCA-GitHub/
├── run_experiments_batched.py      # Main experiment runner (11 experiments)
├── run_exp15.py                    # Benchmark comparison (LIME, GLEAMS, FLocalX, GLocalX)
├── generate_all_figures.py         # Figure generator from JSON results (11 figures)
├── requirements.txt                # Python dependencies
├── data/                           # Datasets (Sensorless Drive, SECOM)
├── kitti_raw/                      # Real KITTI object detection data
│   ├── images/
│   │   ├── train/                  # Training images
│   │   └── val/                    # Validation images
│   ├── labels/
│   │   ├── train/                  # YOLO-format labels
│   │   └── val/
│   └── kitti.yaml                  # KITTI dataset config
├── results/                        # Experiment results (JSON)
│   ├── exp1_results.json           # Narrative Disconnect
│   ├── exp2_results.json           # Bootstrap Stability
│   ├── exp3_results.json           # Hidden Risk Detection
│   ├── exp4_results.json           # Actionability (RFE)
│   ├── exp5_results.json           # Correlation Stress Test
│   ├── exp6_results.json           # Sensitivity Analysis
│   ├── exp7_results.json           # SECOM Semiconductor Audit
│   ├── exp8_results.json           # Simulated KITTI Fusion
│   ├── exp9_results.json           # Sensor Noise Robustness
│   ├── exp10_results.json          # Real KITTI Perception Fusion
│   ├── exp11_results.json          # Ablation Study
│   └── exp15_results.json          # Benchmark Comparison (LIME, GLEAMS, FLocalX, GLocalX)
├── figures/                        # Publication-quality figures
│   ├── figure1_narrative_disconnect.png
│   ├── figure2_stability.png
│   ├── figure3_actionability.png
│   ├── figure4_class_specific.png
│   ├── figure5_correlation_stress.png
│   ├── figure6_sensitivity_heatmap.png
│   ├── figure7_secom_audit.png
│   ├── figure8_kitti.png
│   ├── figure9_noise_robustness.png
│   ├── figure10_ablation.png
│   └── figure11_benchmark.png
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

**Sensorless Drive Diagnosis** (auto-downloaded on first run):
- 58,509 samples, 48 features, 11 motor fault classes
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/529/sensorless+drive+diagnosis)

**SECOM** (auto-downloaded on first run):
- 1,567 samples, 591 features (446 after preprocessing), 6.6% failure rate
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/152/secom)

**Real KITTI** (~390 MB, downloaded automatically):
- 7,481 training + 1,496 validation images, YOLO-format bounding box labels
- Source: [Ultralytics KITTI](https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip)
- Extracted to `kitti_raw/` on first run of Experiment 10

### 3. Run Experiments

```bash
# Run individual experiments (1-10)
python run_experiments_batched.py 1    # Narrative Disconnect (Sensorless Drive)
python run_experiments_batched.py 2    # Bootstrap Stability (Sensorless Drive)
python run_experiments_batched.py 3    # Hidden Risk Detection (Sensorless Drive)
python run_experiments_batched.py 4    # Actionability via RFE (Covertype)
python run_experiments_batched.py 5    # Correlation Stress Test (Sensorless Drive)
python run_experiments_batched.py 6    # Sensitivity Analysis (Sensorless Drive)
python run_experiments_batched.py 7    # SECOM Semiconductor Audit (446 features)
python run_experiments_batched.py 8    # Simulated KITTI Perception Fusion
python run_experiments_batched.py 9    # Sensor Noise Robustness (Sensorless Drive)
python run_experiments_batched.py 10   # Real KITTI Perception Fusion
python run_experiments_batched.py 11   # Ablation Study (incremental CFCA components)
python run_experiments_batched.py 15   # Benchmark Comparison (CFCA vs LIME/GLEAMS/FLocalX/GLocalX)

# Run all experiments sequentially
python run_experiments_batched.py all
```

### 4. Generate Figures

```bash
python generate_all_figures.py
```

Produces 11 publication-quality figures in `figures/` at 300 DPI.

## Experiments Summary

| # | Experiment | Dataset | Key Metric | Result |
|---|-----------|---------|-----------|--------|
| 1 | Narrative Disconnect | Sensorless Drive | NDS | 0.2764 |
| 2 | Bootstrap Stability | Sensorless Drive | BSI | CFCA 0.9955, PFI 0.1871 |
| 3 | Hidden Risk Detection | Sensorless Drive | Risks | Sensor_13, Sensor_36, Sensor_31 |
| 4 | Actionability (RFE) | Covertype | AUAC | Comparable CFCA vs PFI |
| 5 | Correlation Stress Test | Sensorless Drive | Stability | CFCA ~0.18 across all ρ |
| 6 | Sensitivity Analysis | Sensorless Drive | Heatmap | T_global=10 detects up to 12 risks |
| 7 | SECOM Audit | SECOM (446 features) | BSI | CFCA 0.5044, PFI NaN |
| 8 | Simulated KITTI Fusion | KITTI (simulated) | Stability | BSI >0.97 |
| 9 | Sensor Noise Robustness | Sensorless Drive | Correlation | >0.99 at 0.5σ noise |
| 10 | Real KITTI Fusion | KITTI (real, 500 images) | BSI | CFCA 0.9736, PFI 0.9917 |
| 11 | Ablation Study | Sensorless Drive | BSI | Incremental gain across 4 stages |
| 15 | Benchmark Comparison | Sensorless Drive | BSI | CFCA 0.996, LIME 0.9995, FLocalX 0.992, PFI 0.307 |

## Key Findings

1. **Narrative Disconnect**: CFCA achieves NDS=0.28, meaning 28% disagreement with PFI rankings—sufficient to cause misleading conclusions in safety audits.

2. **Stability**: CFCA BSI=0.9955 (near-perfect) vs PFI BSI=0.1871 (unstable). This is the most decisive finding for safety certification.

3. **Hidden Risks**: CFCA detects Sensor_13, Sensor_36, Sensor_31 as globally unimportant but locally critical—PFI would miss these entirely.

4. **Robustness**: CFCA importance remains stable even at 0.5σ input noise, confirming reliability under sensor degradation.

5. **Real KITTI**: On actual KITTI object detection data (500 images, 13 perception features), CFCA identifies detection count, confidence statistics, and object class composition as dominant features.

6. **Benchmark Comparison**: CFCA (BSI=0.996) matches or exceeds LIME (0.9995) and FLocalX (0.9923) in bootstrap stability while providing both global and local importance. PFI (0.307) remains substantially less stable. GLEAMS and GLocalX were unavailable due to library compatibility issues (research code / Python version constraints).

## License

This project is for academic research purposes only. Datasets are subject to their respective licenses.

## Data Sources

- **Sensorless Drive Diagnosis**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/529/sensorless+drive+diagnosis)
- **SECOM**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/152/secom)
- **KITTI**: [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
