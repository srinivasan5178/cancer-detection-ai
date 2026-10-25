# Complete Image Classification Pipeline
## DenseNet201 + EfficientNetV2L Fusion with GWO-Tuned CatBoost

This README explains how to configure, run, and interpret the standalone notebook:

- `Complete_Fusion_Pipeline_Flow.ipynb`
- Companion script (if used): `complete_fusion_pipeline.py`

The notebook is a standalone workflow. It does not depend on cells from the previous notebook.

---

## 1. Pipeline overview

The workflow follows this sequence:

1. Configure the dataset and output paths.
2. Discover class folders and count available images.
3. Split original images into training, validation, and test partitions.
4. Oversample the training partition only.
5. Create image generators; apply augmentation only to training data.
6. Train the DenseNet201 and EfficientNetV2L feature branches and their fusion classifier.
7. Extract fusion embeddings from the training, validation, and test partitions.
8. Use Grey Wolf Optimization (GWO) to search CatBoost hyperparameters using validation Macro-F1.
9. Train the final CatBoost model with the selected hyperparameters.
10. Evaluate the model and save metrics, plots, timings, and model files.

**Important:** The script's source code has been syntax-checked, but a full training run has not been performed as part of preparing these files. Runtime, memory use, dataset compatibility, and final metrics must be confirmed in the user's environment.

---

## 2. Requirements

Recommended:
- Python 3.10 or 3.11, or a TensorFlow-supported Python version for the chosen environment.
- Jupyter Notebook or Google Colab.
- GPU runtime for practical training; DenseNet201 + EfficientNetV2L can require substantial memory.
- Enough disk space for copied split images, balanced training images, saved models, and plots.
- Internet access on first run if ImageNet pretrained weights need to be downloaded.

Python packages used by the pipeline:
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- CatBoost

In a notebook cell, install the additional packages:

```python
%pip install -q catboost scikit-learn matplotlib
```

TensorFlow is assumed to be installed by the selected environment. If it is not installed, install a TensorFlow version compatible with your Python and CUDA/runtime configuration before running the notebook.

---

## 3. Dataset folder structure

Set `DATASET_ROOT` to the directory containing the image-class folders. The code supports these structures.

### Option A — class folders directly under the dataset root

```text
dataset/
├── class_1/
│   ├── image001.jpg
│   └── image002.jpg
├── class_2/
│   ├── image003.jpg
│   └── image004.jpg
└── class_3/
    ├── image005.jpg
    └── image006.jpg
```

### Option B — one domain folder above class folders

```text
dataset/
├── domain_A/
│   ├── class_1/
│   └── class_2/
└── domain_B/
    ├── class_3/
    └── class_4/
```

When duplicate class names are found under different domain folders, the script prefixes a domain name to keep class labels unique.

Supported image extensions include JPG, JPEG, PNG, BMP, TIFF, and WEBP. Each class should contain enough images to populate the training, validation, and test partitions. The dataset should have at least two non-empty classes.

**Set the path:** Open the main code cell and update the configuration line:

```python
DATASET_ROOT = "/content/Project_Sample"
```

For Google Drive, mount Drive first and set this variable to the actual mounted dataset path. Do not assume the example path exists.

---

## 4. Running in Google Colab

1. Upload and open `Complete_Fusion_Pipeline_Flow.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type** and choose a GPU if available.
3. Verify the dataset directory and update `DATASET_ROOT`.
4. Run the dependency-installation cell.
5. Run the complete pipeline code cell.
6. Wait for model training, GWO search, CatBoost training, and evaluation to finish.
7. Inspect the output directory and download or copy the results you need.

The full pipeline may take a long time, especially with large datasets, pretrained backbones, and GWO trials. If you run out of memory, consider a smaller batch size, a smaller image size, fewer GWO agents/iterations, or a lighter backbone. Document any changed settings for reproducibility.

---

## 5. Configuration parameters

The consolidated script exposes settings near the top of the code cell:

| Parameter | Purpose |
|---|---|
| `DATASET_ROOT` | Input dataset folder; must be set correctly |
| `OUTPUT_ROOT` | Root folder for generated outputs |
| `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` | Data split proportions; default 70/15/15 |
| `IMAGE_SIZE` | Input image width and height; default 64 × 64 |
| `BATCH_SIZE` | Images per batch; default 8 |
| `EPOCHS` | Maximum fusion-model epochs; default 50 |
| `SEED` | Random seed for reproducibility |
| `GWO_AGENTS` | Number of candidate solutions in GWO |
| `GWO_ITERS` | Number of GWO iterations |
| `CATBOOST_ITERATIONS` | CatBoost boosting iterations |
| `FUSION_LEARNING_RATE` | Fusion-model optimizer learning rate |

The default image size of 64 × 64 is retained from the supplied notebook. It is small for transfer learning; compare alternatives such as 224 × 224 as a separate experiment if compute resources permit. Changing the image size increases compute and memory requirements.

---

## 6. Data split and leakage precautions

The pipeline splits the original image files before oversampling. Oversampling is applied to the training partition only; validation and test partitions are not oversampled.

This reduces one common source of leakage, but it does not guarantee independence if the dataset contains multiple images from the same patient, slide, specimen, video, or acquisition session. For those datasets, split by patient/source/group ID rather than randomly splitting individual images.

The validation set is used for GWO hyperparameter selection. The test set is held out for final evaluation and should not be used to tune hyperparameters or select the best model.

Copy-based oversampling duplicates existing training images; it does not create new independent observations. Augmentation also does not replace a genuinely independent test set.

---

## 7. Model design

### DenseNet201 and EfficientNetV2L

The image is passed through two pretrained CNN branches:
- DenseNet201
- EfficientNetV2L

Each branch uses global average pooling and a learned projection. The branch embeddings are concatenated and passed through dense, batch-normalization, and dropout layers to form a fused representation and class-probability output.

The script uses ImageNet pretrained weights. On first execution, those weights may need to be downloaded. Both networks are computationally expensive, and EfficientNetV2L may exceed the memory available on smaller GPUs.

### Fusion embedding and CatBoost

After training the fusion model, the code extracts the layer named `fusion_embedding` for each partition. CatBoost is trained on the training embeddings.

### Grey Wolf Optimization

GWO searches the configured ranges for:
- CatBoost learning rate
- CatBoost depth
- CatBoost L2 leaf regularization

The objective is **1 − validation Macro-F1**. The chosen parameters are then used to fit the final CatBoost model on training embeddings. The test partition is not used in the optimization objective.

---

## 8. Evaluation metrics

The script reports the following metrics for the training, validation, and test partitions:

- **Accuracy:** fraction of predictions that are correct.
- **Macro precision:** precision calculated per class and averaged equally across classes.
- **Macro recall:** recall calculated per class and averaged equally across classes.
- **Macro-F1:** F1 calculated per class and averaged equally across classes; useful when class counts differ.
- **Weighted precision, recall, and F1:** class metrics averaged using each class's support.
- **Macro AUROC (one-vs-rest):** average class discrimination measure based on one-vs-rest ROC curves, where it is defined.

The test metrics are the primary held-out results to report. Training performance alone is not evidence of generalization. A large gap between training and test performance can indicate overfitting or distribution differences.

ROC/AUROC may be undefined for a class if the relevant evaluation split does not contain both positive and negative examples. The code skips drawing a class ROC curve in that case and may report AUROC as unavailable.

---

## 9. Output directory and files

By default, outputs are written under:

```text
/content/fusion_pipeline_output/
├── split/
│   ├── train/
│   ├── val/
│   └── test/
├── balanced_train/
├── plots/
│   ├── confusion_matrix_train.png
│   ├── confusion_matrix_validation.png
│   ├── confusion_matrix_test.png
│   ├── roc_train.png
│   ├── roc_validation.png
│   ├── roc_test.png
│   └── fusion_training_curves.png
├── reports/
│   ├── class_mapping.json
│   ├── model_metrics.csv
│   └── timing_summary.csv
└── models/
    ├── best_fusion_model.keras
    ├── final_fusion_model.keras
    └── catboost_fusion_model.cbm
```

Files and exact output availability depend on whether the pipeline reaches each stage successfully.

### Key report files

- `reports/model_metrics.csv`: metrics for train, validation, and test.
- `reports/timing_summary.csv`: fusion training, feature extraction, GWO search, CatBoost training, and total pipeline runtime.
- `reports/class_mapping.json`: mapping between class names and numeric indices.
- `plots/`: confusion matrices, ROC curves, and training curves.
- `models/`: saved fusion and CatBoost models.

The split and balanced image folders use disk space. The script recreates the split folder and balanced training folder when it starts; do not point these output paths at your original dataset.

---

## 10. Troubleshooting

### `FileNotFoundError: Dataset folder not found`
Check `DATASET_ROOT`, mount Google Drive if required, and verify that the directory exists.

### No classes or images found
Confirm that the class folders are directly under the dataset root or one level below it. Confirm the files use supported image extensions.

### ImageNet weights cannot be downloaded
Check network access. If running offline, arrange for the required pretrained weights to be available in the environment or adapt the model to use locally available weights. Results from random initialization will not be directly comparable to pretrained runs.

### GPU out-of-memory
Reduce `BATCH_SIZE`, use a smaller input resolution, reduce the number of GWO trials, or use a lighter model. Record changes in the experiment notes.

### TensorFlow / CUDA errors
Use a TensorFlow build compatible with the active Python and GPU runtime. Restart the runtime after changing major packages.

### Training takes too long
Start with a small trial run to verify paths and pipeline compatibility. Reduce epochs and GWO agents/iterations for debugging, then restore the planned settings for the final experiment.

### Unexpectedly high test performance
Check duplicate images, patient/source overlap, preprocessing leakage, class-label errors, and whether test data influenced tuning. Consider group-based splitting and independent external testing.

### Output files are missing
Check the last error in the notebook output and confirm which pipeline stage completed. Files created before a failure may exist, but later reports and models will not be available.

---

## 11. Reproducibility and reporting checklist

Before reporting results:

- [ ] Record the dataset source, version, class names, and image counts.
- [ ] Record Python, TensorFlow, scikit-learn, and CatBoost versions.
- [ ] Record GPU/CPU hardware and runtime.
- [ ] Record all changed configuration values.
- [ ] Verify the train/validation/test split and source-level independence.
- [ ] Confirm the validation set alone was used for GWO tuning.
- [ ] Report independent test accuracy, macro precision, macro recall, Macro-F1, and AUROC where defined.
- [ ] Include confusion matrices and per-class precision/recall/F1.
- [ ] Keep the model files, CSV reports, plots, and class mapping together.
- [ ] Repeat experiments with multiple seeds if statistical variability is important.

---

## 12. Limitations

- The README describes the code structure and intended workflow; it does not certify that training has succeeded on a particular dataset.
- No performance values should be claimed until the notebook has been executed and the generated reports inspected.
- Random image-level splitting may be inappropriate for patient-, slide-, or source-grouped datasets.
- Copy-based oversampling can increase training exposure to minority images but does not add independent data.
- The default 64 × 64 resolution may limit fine-grained visual information.
- GWO search can be computationally expensive because each candidate trains a CatBoost model.
- A saved model is not a complete deployment package by itself; inference preprocessing and class-index mapping must also be retained.

---

## 13. Citation and licensing note

DenseNet201, EfficientNetV2, TensorFlow/Keras, scikit-learn, and CatBoost are third-party methods and libraries. If this pipeline is used in a paper or product, cite the relevant original methods and follow the licenses and terms for the software and dataset used. Verify the exact references and dataset permissions before publication.

---

## 14. Quick start summary

1. Open `Complete_Fusion_Pipeline_Flow.ipynb`.
2. Install dependencies.
3. Set `DATASET_ROOT`.
4. Run the complete pipeline code cell.
5. Review `reports/model_metrics.csv` and `reports/timing_summary.csv`.
6. Inspect test confusion matrix and ROC plot.
7. Report only metrics generated by a completed run.
