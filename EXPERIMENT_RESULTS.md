# Experiment Results

## 1. Overview

This project evaluates a five-class lung and colon cancer histopathology classification framework using DenseNet201 and EfficientNetV2L feature extraction, scale-aware late feature fusion, and CatBoost with Grey Wolf Optimizer (GWO) or Bayesian Optimization with Hyperband (BOHB).

The experiments distinguish the repeated controlled evaluation from the separate, single-run classifier-optimization experiments. Results from these protocols should not be treated as directly interchangeable.

## 2. Dataset and Evaluation Protocol

The experiments use the LC25000 dataset, containing 25,000 RGB histopathological images distributed across five classes, with 5,000 images per class.

The documented fixed partition is:

| Partition | Number of images | Percentage |
|---|---:|---:|
| Training | 17,500 | 70% |
| Validation | 3,750 | 15% |
| Held-out test | 3,750 | 15% |
| Total | 25,000 | 100% |

The controlled evaluation uses stratified five-fold cross-validation on the 21,250-image development pool, repeated across random seeds 42, 123, and 2026, producing 15 matched validation observations.

Feature-standardization parameters are estimated from each training fold and applied unchanged to its validation fold. The locked test set is excluded from fold generation and model selection.

## 3. Model Architecture and Feature Fusion

DenseNet201 and EfficientNetV2L are used as heterogeneous feature extractors.

| Component | Feature dimension |
|---|---:|
| DenseNet201 | 1,920 |
| EfficientNetV2L | 1,280 |
| Concatenated representation | 3,200 |

The scale-aware fusion method standardizes each backbone's feature block using training-partition statistics before concatenation.

### Feature-scale diagnostics

| Diagnostic | DenseNet201 | EfficientNetV2L |
|---|---:|---:|
| Mean absolute feature value | 0.191906 | 0.172698 |
| Mean L2 norm | 30.3247 | 6.7382 |
| Mean energy contribution | 93.78% | 6.22% |
| PCA dimensions retaining 95% variance | 919 | 420 |

The reported feature-energy ratio is approximately 22.98:1 in favour of DenseNet201 before standardization. These measurements motivate the investigation of backbone-wise feature-scale regulation.

## 4. Main Controlled Evaluation

The repeated controlled evaluation reports the following results for standardized DenseNet201–EfficientNetV2L fusion using a Softmax classification head.

| Metric | Reported result |
|---|---:|
| Accuracy | 96.54% ± 0.34% |
| F1-score | 0.9654 ± 0.0034 |
| Macro-AUROC | 0.9980 ± 0.0003 |

These values represent the reported 15-observation controlled evaluation, not the separate single-run GWO–CatBoost test.

The manuscript also reports that standardized fusion improves accuracy by 2.061 percentage points over unstandardized fusion under the stated comparison.

## 5. Single-Run Classifier Results

The supplementary material reports the following single-run evaluation metrics.

| Classifier | Evaluation state | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Softmax | Validation | 97.07% | 0.9707 | 0.9707 | 0.9705 |
| Softmax | Held-out test | 97.41% | 0.9742 | 0.9741 | 0.9740 |
| CatBoost | Training | 97.53% | 0.9753 | 0.9753 | 0.9753 |
| CatBoost | Validation | 95.55% | 0.9556 | 0.9555 | 0.9553 |
| CatBoost | Held-out test | 95.31% | 0.9533 | 0.9531 | 0.9528 |

These results are reported separately from the repeated cross-validation results and the GWO/BOHB optimization track.

## 6. GWO and BOHB Optimization Results

The supplementary material reports the following single-run held-out test results.

| Model | Optimizer | Test accuracy | Precision | AUROC |
|---|---|---:|---:|---:|
| DenseNet201 + EfficientNetV2L + CatBoost | GWO | 97.44% | 0.9744 | 0.9985 |
| DenseNet201 + EfficientNetV2L + CatBoost | BOHB | 96.19% | 0.9620 | Not reported in Table S4 |

The reported GWO result is higher than the BOHB result in this optimization track. However, these are single-run outcomes; they do not establish statistically significant superiority of GWO over BOHB.

## 7. Statistical Comparisons

The supplementary statistical analysis reports paired comparisons, confidence intervals, paired t-tests, Wilcoxon tests, Cohen's effect sizes, and Holm correction.

### Standardized fusion versus individual backbones

| Comparison | Metric | Mean difference | 95% confidence interval | Paired t-test p-value | Cohen's effect size |
|---|---|---:|---|---:|---:|
| Fusion − DenseNet201 | Accuracy | +0.007257 | [0.005623, 0.008891] | 5.05 × 10⁻⁷ | 2.248 |
| Fusion − DenseNet201 | F1 | +0.007254 | [0.005632, 0.008876] | 4.64 × 10⁻⁷ | 2.264 |
| Fusion − DenseNet201 | AUROC | +0.000682 | [0.000603, 0.000761] | 1.00 × 10⁻¹⁰ | 4.376 |
| Fusion − EfficientNetV2L | Accuracy | +0.092686 | [0.090459, 0.094913] | 3.77 × 10⁻²⁰ | 21.061 |
| Fusion − EfficientNetV2L | F1 | +0.092768 | [0.090528, 0.095008] | 4.04 × 10⁻²⁰ | 20.960 |
| Fusion − EfficientNetV2L | AUROC | +0.018626 | [0.018276, 0.018977] | 1.24 × 10⁻²¹ | 26.888 |

The supplementary table marks these comparisons as significant after Holm correction. The interpretation depends on the paired experimental observations and the stated validation protocol.

### Softmax versus CatBoost using the same fusion representation

| Metric | Softmax mean | CatBoost mean | Difference |
|---|---:|---:|---:|
| Accuracy | 0.965390 | 0.939048 | +0.026343 |
| Precision | 0.965476 | 0.939084 | +0.026393 |
| Recall | 0.965390 | 0.939048 | +0.026343 |
| F1-score | 0.965412 | 0.939011 | +0.026401 |
| AUROC | 0.998022 | 0.995049 | +0.002973 |

The supplementary analysis reports statistically significant paired differences for these metrics after correction. These paired comparisons should be interpreted separately from the single-run GWO and BOHB results.

## 8. Figures and Diagnostic Outputs

The supplementary material includes the following figures:

| Figure | Description |
|---|---|
| S1 | Training and validation accuracy trajectory |
| S2 | Softmax validation confusion matrix |
| S3 | Softmax held-out test confusion matrix |
| S4 | CatBoost training confusion matrix |
| S5 | CatBoost validation confusion matrix |
| S6 | CatBoost held-out test confusion matrix |
| S7–S9 | CatBoost one-versus-rest ROC curves for training, validation, and test |
| S10 | GWO test confusion matrix |
| S11 | GWO test ROC curves |
| S12 | BOHB test confusion matrix |
| S13 | BOHB test ROC curves |
| S14 | PCA effective-dimensionality diagnostic |
| S15 | Backbone feature-energy contribution before standardization |

Where available, the corresponding original plots should be included in the repository's `results/` directory with clear filenames and captions.

## 9. Recommended Repository Results Structure

```text
results/
├── model_comparison.csv
├── metrics_summary.csv
├── timing_summary.csv
├── confusion_matrix_softmax_test.png
├── confusion_matrix_catboost_test.png
├── confusion_matrix_gwo_test.png
├── confusion_matrix_bohb_test.png
├── roc_curves_gwo_test.png
├── roc_curves_bohb_test.png
└── feature_scale_diagnostics.png
```

These are suggested filenames, not confirmation that all files currently exist. Add only the files actually generated by the corresponding experiments.

## 10. Reproducibility and Limitations

For reproducibility, document the random seeds, data partitions, class mapping, preprocessing, model configurations, optimizer search spaces, selected hyperparameters, package versions, and computational environment.

Training and inference times should be reported only from recorded experiment logs or timing CSV files. They are not provided here because verified timing values were not available in the supplied manuscript excerpts.

The reported results are based on LC25000. Repeated internal validation does not replace independent external-dataset testing or prospective clinical evaluation. Potential source-image dependence should also be considered when interpreting performance on datasets constructed using image augmentation.

The results should therefore be presented as benchmark classification findings rather than evidence of clinical readiness.
