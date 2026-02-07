Hybrid GWO + CatBoost + DenseNet201 + EfficientNetV2M
=======================================================
A hybrid deep learning and optimization pipeline for lung and colon cancer classification, combining:

1.DenseNet201 & EfficientNetV2M/V2L for feature extraction
2.Fusion model for multi-branch representation learning
3.Grey Wolf Optimizer (GWO) for hyperparameter tuning
4.CatBoost for robust classification
5.AUROC, confusion matrix, and F1-score for evaluation

Features
========
1.Data balancing: Oversampling per class to handle imbalance
2.Train/Val/Test split with reproducible ratios
3.Fusion CNN model combining DenseNet201 and EfficientNetV2M
4.Feature extraction for downstream ML classifiers
5.GWO optimization of CatBoost hyperparameters
6.Evaluation metrics: Accuracy, Precision, Recall, F1, AUROC
7.Visualization: Confusion matrices and ROC curves saved to ./plots

├── hybrid-gwo-catboost-densenet201-efficientnetv2m-v2l.py
├── plots/                # ROC curves & confusion matrices
├── Project_Sample/       # Original dataset (colon/, lung/)
├── Project_Sample_balanced/  # Oversampled dataset
├── Project_Sample_split_bal/ # Train/Val/Test splits

Usage
=====
1.Prepare dataset  

Run pipeline
============
2.Place colon and lung cancer image folders under Project_Sample/.
├── python hybrid-gwo-catboost-densenet201-efficientnetv2m-v2l.py

Outputs
=======
1.Balanced dataset in Project_Sample_balanced/
2.Train/Val/Test splits in Project_Sample_split_bal/
3.Best fusion model: best_fusion_model.h5
4.Optimized CatBoost model: catboost_fusion_model_macrof1_weights.cbm
5.Plots: ROC curves and confusion matrices in ./plots

Evaluation
==========
1.Confusion Matrix: Per-class performance visualization
2.Classification Report: Precision, Recall, F1-score
3.AUROC: Macro-average ROC curves for Train/Val/Test splits
4.Computational Time: Inference + AUROC timing

Configuration
============
1.Train/Val/Test ratio: RATIO = (0.7, 0.15, 0.15)
2.Batch size: 8
3.Epochs: 50
4.GWO agents/iterations: 6 each (tunable)

