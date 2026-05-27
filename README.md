# Classifier-Aware Temporal Evidence Decomposition for Explainable Time-Series Decisions

This repository contains the code used to reproduce the experiments for the manuscript:

**Classifier-Aware Temporal Evidence Decomposition for Explainable Time-Series Decisions**

The paper studies an intrinsic classifier-level evidence decomposition mechanism for time-series classifiers. The proposed model uses an evidence-gated state-space encoder and represents the classifier input as an additive sum of temporal evidence vectors:

$$
u = \sum_{t=1}^{T} e_t, \qquad e_t = \alpha_t z_t .
$$

This repository is intended as a clean reproducibility package for the submitted manuscript. It includes only the scripts used in the current paper version.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── process_datasets.py
├── run_main_accuracy_multiseed_aggregation_ablation.py
├── run_paired_statistical_tests.py
├── run_evidence_deletion_curves.py
├── run_ucr_xai_faithfulness_10datasets.py
├── run_qualitative_classifier_aware_profiles.py
├── Datasets/
└── Results/
```

The `Datasets/` and `Results/` directories can be created locally. Raw UCR datasets are not included in this repository.

---

## Scripts and Manuscript Mapping

| Manuscript item | Script |
|---|---|
| Table 2: aggregation ablation | `run_main_accuracy_multiseed_aggregation_ablation.py` |
| Table 3: paired statistical tests | `run_paired_statistical_tests.py` |
| Table 4 and Figure 2: evidence-norm deletion curves | `run_evidence_deletion_curves.py` |
| Table 5: XAI faithfulness against gradient baselines | `run_ucr_xai_faithfulness_10datasets.py` |
| Figure 3: qualitative classifier-aware evidence profiles | `run_qualitative_classifier_aware_profiles.py` |
| Dataset loading and preprocessing | `process_datasets.py` |

Earlier exploratory scripts for synthetic motif localization and perturbed-input robustness are intentionally not included in the main workflow because they are not part of the submitted manuscript.

---

## Environment

Recommended environment:

- Python 3.10 or later
- PyTorch
- NumPy
- pandas
- scikit-learn
- SciPy
- Matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` can be:

```text
numpy
pandas
scikit-learn
scipy
matplotlib
torch
```

For GPU support, install the PyTorch build appropriate for your CUDA version.

---

## Dataset Preparation

The scripts expect UCR-style train/test files where the first column is the class label and the remaining columns are time-series values.

The loader supports both flat and nested dataset layouts.

### Nested layout

```text
Datasets/
├── ECG5000/
│   ├── ECG5000_TRAIN
│   └── ECG5000_TEST
├── Wafer/
│   ├── Wafer_TRAIN
│   └── Wafer_TEST
└── ...
```

### Flat layout

```text
Datasets/
├── ECG5000_TRAIN
├── ECG5000_TEST
├── Wafer_TRAIN
├── Wafer_TEST
└── ...
```

Files with `.txt` extensions are also supported.

The ten UCR datasets used in the manuscript are:

```text
ECG5000
Wafer
ElectricDevices
FaceAll
PhalangesOutlinesCorrect
CricketX
SwedishLeaf
UWaveGestureLibraryX
Yoga
Earthquakes
```

Each time series is independently standardized by `process_datasets.py`.

---

## Reproducing the Main Results

### 1. Aggregation ablation: Table 2

This script compares `PlainSSM`, `NormGated`, and `Proposed_Unnormalized_Base` across the ten selected UCR datasets and five random seeds.

```bash
python run_main_accuracy_multiseed_aggregation_ablation.py --device cpu
```

To use GPU:

```bash
python run_main_accuracy_multiseed_aggregation_ablation.py --device cuda
```

Outputs are saved under `Results/`, including files similar to:

```text
multiseed_aggregation_ablation_detail_<timestamp>.csv
multiseed_aggregation_ablation_summary_<timestamp>.csv
```

---

### 2. Paired statistical tests: Table 3

After running the aggregation ablation, use the generated detail CSV as input.

```bash
python run_paired_statistical_tests.py --input-detail-csv Results/<aggregation_detail_csv>.csv
```

Example:

```bash
python run_paired_statistical_tests.py --input-detail-csv Results/multiseed_aggregation_ablation_detail_YYYYMMDD_HHMMSS.csv
```

This script computes dataset-level paired comparisons between `Proposed_Unnormalized_Base` and the matched baselines.

---

### 3. Evidence-norm deletion curves: Table 4 and Figure 2

This script evaluates representation-level deletion across evidence ratios using the evidence-norm score:

$$
s_t^{(\mathrm{norm})}=\|e_t\|_2=\|\alpha_t z_t\|_2 .
$$

```bash
python run_evidence_deletion_curves.py --device cpu
```

Outputs include detail and summary CSV files and average deletion-curve figures:

```text
evidence_deletion_curves_detail_<timestamp>.csv
evidence_deletion_curves_summary_<timestamp>.csv
evidence_deletion_curves_average_<timestamp>.pdf
evidence_deletion_curves_average_<timestamp>.png
```

---

### 4. XAI faithfulness against gradient baselines: Table 5

This is the main representation-level XAI comparison. It compares:

- random ranking;
- latent norm $\|z_t\|_2$;
- gate activation $\alpha_t$;
- evidence norm $\|e_t\|_2$;
- class-logit score $w_{\hat c}^{\top}e_t$;
- margin-logit score $(w_{\hat c}-w_{c'})^{\top}e_t$;
- gradient saliency;
- input x gradient;
- Integrated Gradients.

Run:

```bash
python run_ucr_xai_faithfulness_10datasets.py --device cpu --ig-steps 16
```

Outputs are saved under:

```text
Results_UCR_XAI_Faithfulness/
```

The most important output is the summary CSV generated by the script.

---

### 5. Qualitative classifier-aware profiles: Figure 3

This script generates qualitative classifier-aware evidence profiles for representative ECG5000 and ElectricDevices examples. The generated figures show:

- input time series;
- class-logit evidence score $w_{\hat c}^{\top}e_t$;
- margin-logit evidence score $(w_{\hat c}-w_{c'})^{\top}e_t$.

```bash
python run_qualitative_classifier_aware_profiles.py
```

Outputs are saved under a timestamped folder in `Results/`, for example:

```text
Results/classifier_aware_qualitative_profiles_<timestamp>/
```

---




## Method Summary

The proposed model uses a simple state-space encoder:

$$
h_t = A h_{t-1} + Bx_t
$$

$$
z_t = \tanh\left(W_z [h_t, x_t] + b_z\right)
$$

$$
\alpha_t = \sigma\left(w_g^{\top} z_t + b_g\right)
$$

$$
u = \sum_{t=1}^{T} \alpha_t z_t = \sum_{t=1}^{T} e_t,
\qquad e_t = \alpha_t z_t .
$$

The classifier-aware scores used in the paper are:

$$
s_t^{(\mathrm{class})} = w_{\hat c}^{\top} e_t
$$

$$
s_t^{(\mathrm{margin})} = (w_{\hat c} - w_{c'})^{\top} e_t
$$

Deletion and insertion are performed at the classifier-representation level by masking additive evidence terms $e_t$, not by perturbing the raw input sequence.

---

## Expected Runtime

Runtime depends on hardware, dataset size, and whether CPU or GPU is used.

The Integrated Gradients baseline is substantially slower than the intrinsic classifier-aware scores because it requires multiple gradient evaluations. The manuscript uses 16 integration steps for Integrated Gradients.

On a standard laptop CPU, the full set of experiments may take several hours.

---

## Citation

If you use this repository, please cite the corresponding manuscript:

```bibtex
@article{altay2026classifieraware,
  title={Classifier-Aware Temporal Evidence Decomposition for Explainable Time-Series Decisions},
  author={Altay, Tayip},
  journal={Manuscript under review},
  year={2026}
}
```

Please update the citation after publication.

---

## License

Add the license selected for this repository here.

Recommended source-code licenses:

- MIT License
- Apache 2.0 License

For source code, MIT or Apache 2.0 is usually more appropriate than a Creative Commons license.
