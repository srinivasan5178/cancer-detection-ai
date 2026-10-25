# Cancer Histopathology Classification with Feature Fusion

A five-class LC25000 image-classification project using DenseNet201 and EfficientNetV2L feature extraction, scale-aware late feature fusion, and CatBoost hyperparameter optimization with Grey Wolf Optimization (GWO). The project also documents a Softmax baseline and a BOHB optimization comparison.

> **Research note:** Reported metrics in `EXPERIMENT_RESULTS.md` come from the supplied manuscript/supplementary material and refer to different evaluation protocols. Do not compare them as if they were produced by one identical run. Re-run the pipeline and retain its generated outputs before claiming reproducibility.

## Repository contents

- `complete_fusion_pipeline.py` — standalone Python pipeline source.
- `requirements.txt` — Python dependencies.
- `DATASET_INSTRUCTIONS.md` — dataset layout, path setup, preprocessing, and leakage-control guidance.
- `EXPERIMENT_RESULTS.md` — reported evaluation results and reporting notes.
- `docs/DenseNet201_EfficientNetV2L_GWO_CatBoost_Workflow.png` — graphical workflow.
- `docs/README.md` — documentation directory note.
- `results/README.md` — what to place in the results directory.

An executable `.ipynb` notebook is not included in this repository snapshot. If you want a Colab notebook, upload the actual notebook file; do not rename the Python script to `.ipynb`.

## Dataset setup

The pipeline expects the five class folders to be under `DATASET_ROOT`:

```text
Project_Sample/
├── colon_aca/
├── colon_n/
├── lung_aca/
├── lung_n/
└── lung_scc/
```

Folder names must match the class names supported by the script. See [`DATASET_INSTRUCTIONS.md`](DATASET_INSTRUCTIONS.md) for additional folder-layout notes and data-leakage precautions.

The LC25000 dataset is not included. Obtain it from an authorized source and ensure that your use and redistribution comply with its terms.

## Installation

Recommended: Python 3.10–3.12 in a clean virtual environment. TensorFlow GPU support depends on the operating system, hardware, and compatible drivers.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure and run

1. Open `complete_fusion_pipeline.py`.
2. Find `DATASET_ROOT` and set it to the directory that directly contains the class folders.
3. Check the configured image size, training epochs, split ratios, random seed, and GWO search settings before running.
4. Run:

```bash
python complete_fusion_pipeline.py
```

For Windows paths, use a raw string, for example:

```python
DATASET_ROOT = r"C:\path\to\Project_Sample"
```

The first run may download pretrained ImageNet weights, so network access may be needed. Training can be computationally expensive; a compatible GPU is recommended.

## Evaluation and reproducibility

- Split original images before applying oversampling or augmentation.
- Fit preprocessing/standardization only on the training partition or training fold.
- Use validation data for model selection and optimizer tuning; keep the held-out test set independent until final evaluation.
- Where source/patient/specimen identifiers exist, use group-aware splitting to reduce leakage between partitions.
- Save the actual metrics CSVs, confusion matrices, ROC curves, selected hyperparameters, and timing logs produced by each run.
- Record software versions, seeds, split counts, and hardware.

See [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) for reported metrics and protocol distinctions. Results must be verified against the output files from the corresponding run before publication.

## Workflow

![DenseNet201–EfficientNetV2L–GWO–CatBoost workflow](docs/DenseNet201_EfficientNetV2L_GWO_CatBoost_Workflow.png)

## Limitations

This is a research classification pipeline, not a clinically validated diagnostic product. Performance on LC25000 alone does not establish external generalization or clinical readiness.

