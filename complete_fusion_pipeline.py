"""
Consolidated image-classification pipeline:
DenseNet201 + EfficientNetV2L fusion -> feature extraction -> GWO-tuned CatBoost.

Expected input:
  DATASET_ROOT/
    colon_aca/   *.jpg, *.png, ...
    colon_n/
    lung_aca/
    lung_n/
    lung_scc/

Also supports one level of domain folders such as:
  DATASET_ROOT/colon/<class_name>/
  DATASET_ROOT/lung/<class_name>/

Important:
- Original images are split first; oversampling is applied ONLY to training data.
- Validation and test sets remain untouched to reduce data leakage.
- Set DATASET_ROOT to the actual dataset folder before running.
- This script uses pretrained ImageNet weights and may require internet on first run.
"""

import os
import csv
import json
import time
import shutil
import random
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Concatenate, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201, EfficientNetV2L

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------- CONFIGURATION -------------------------
DATASET_ROOT = "/content/Project_Sample"  # CHANGE if your dataset is elsewhere
OUTPUT_ROOT = "/content/fusion_pipeline_output"
SPLIT_ROOT = os.path.join(OUTPUT_ROOT, "split")
BALANCED_TRAIN_ROOT = os.path.join(OUTPUT_ROOT, "balanced_train")
PLOTS_DIR = os.path.join(OUTPUT_ROOT, "plots")
REPORTS_DIR = os.path.join(OUTPUT_ROOT, "reports")
MODEL_DIR = os.path.join(OUTPUT_ROOT, "models")

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
IMAGE_SIZE = (64, 64)   # Retained from the supplied notebook; consider 224x224 for stronger transfer learning
BATCH_SIZE = 8
EPOCHS = 50
SEED = 42
GWO_AGENTS = 6
GWO_ITERS = 6
FUSION_LEARNING_RATE = 0.0003176
FUSION_DENSE_UNITS = 512
FUSION_DROPOUT = 0.20
CATBOOST_ITERATIONS = 500

for folder in (OUTPUT_ROOT, PLOTS_DIR, REPORTS_DIR, MODEL_DIR):
    os.makedirs(folder, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------- DATASET DISCOVERY -------------------------
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(path):
    return Path(path).suffix.lower() in VALID_EXTENSIONS

def clean_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def discover_classes(dataset_root):
    """Return {class_name: [image paths]} for flat or one-level domain/class layouts."""
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_root}\n"
            "Update DATASET_ROOT to the folder containing your class directories."
        )

    class_files = {}
    top_dirs = sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    )

    for top in top_dirs:
        top_path = os.path.join(dataset_root, top)
        child_dirs = sorted(
            d for d in os.listdir(top_path)
            if os.path.isdir(os.path.join(top_path, d))
        )
        if child_dirs:
            # Treat children as classes. If duplicate class names occur in separate
            # domains, prefix with the domain to keep labels unique.
            for child in child_dirs:
                class_name = child
                files = [
                    os.path.join(top_path, child, f)
                    for f in os.listdir(os.path.join(top_path, child))
                    if os.path.isfile(os.path.join(top_path, child, f))
                    and is_image_file(f)
                ]
                if class_name in class_files:
                    class_name = f"{top}_{child}"
                class_files[class_name] = files
        else:
            files = [
                os.path.join(top_path, f)
                for f in os.listdir(top_path)
                if os.path.isfile(os.path.join(top_path, f)) and is_image_file(f)
            ]
            class_files[top] = files

    class_files = {k: v for k, v in class_files.items() if v}
    if len(class_files) < 2:
        raise ValueError(
            f"Found fewer than 2 non-empty classes under {dataset_root}. "
            "Check the folder layout and image extensions."
        )

    print("\\nOriginal dataset class counts:")
    for cls, files in sorted(class_files.items()):
        print(f"  {cls}: {len(files)}")
    return dict(sorted(class_files.items()))

# ------------------------- LEAKAGE-CONTROLLED SPLIT -------------------------
def split_original_files(class_files, split_root, seed=SEED):
    """Split unique original files before any oversampling."""
    clean_folder(split_root)
    rng = random.Random(seed)

    for cls, paths in class_files.items():
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # Ensure a non-empty test set when class has at least 3 images.
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]

        for split_name, split_paths in (
            ("train", train_paths), ("val", val_paths), ("test", test_paths)
        ):
            out_dir = os.path.join(split_root, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for index, src in enumerate(split_paths):
                # Unique names retain source file name plus index to avoid collisions.
                dst = os.path.join(out_dir, f"{index:06d}_{os.path.basename(src)}")
                shutil.copy2(src, dst)

        print(
            f"{cls}: original={n}, train={len(train_paths)}, "
            f"val={len(val_paths)}, test={len(test_paths)}"
        )

def oversample_training_only(split_root, balanced_train_root):
    """Duplicate minority training images only; validation/test remain untouched."""
    train_root = os.path.join(split_root, "train")
    clean_folder(balanced_train_root)
    classes = sorted(
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    )
    counts = {
        cls: len([
            f for f in os.listdir(os.path.join(train_root, cls))
            if is_image_file(f)
        ])
        for cls in classes
    }
    if not counts or max(counts.values()) == 0:
        raise ValueError("Training split contains no images.")
    target = max(counts.values())
    rng = random.Random(SEED)

    for cls in classes:
        src_dir = os.path.join(train_root, cls)
        dst_dir = os.path.join(balanced_train_root, cls)
        os.makedirs(dst_dir, exist_ok=True)
        files = [
            os.path.join(src_dir, f) for f in os.listdir(src_dir)
            if is_image_file(f)
        ]
        if not files:
            raise ValueError(f"Class '{cls}' has no training images.")
        for i, src in enumerate(files):
            shutil.copy2(src, os.path.join(dst_dir, f"original_{i:06d}{Path(src).suffix.lower()}"))
        extra_needed = target - len(files)
        for j in range(extra_needed):
            src = rng.choice(files)
            dst = os.path.join(
                dst_dir, f"oversample_{j:06d}_{Path(src).stem}{Path(src).suffix.lower()}"
            )
            shutil.copy2(src, dst)
        print(f"Training balance {cls}: {len(files)} -> {target}")
    return classes

# ------------------------- DATA GENERATORS -------------------------
def create_data_generator(dataset_dir, class_names, batch_size, shuffle):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Split folder not found: {dataset_dir}")
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20 if shuffle else 0,
        width_shift_range=0.1 if shuffle else 0.0,
        height_shift_range=0.1 if shuffle else 0.0,
        shear_range=0.1 if shuffle else 0.0,
        zoom_range=0.1 if shuffle else 0.0,
        horizontal_flip=shuffle,
        fill_mode="nearest"
    )
    gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="sparse",
        classes=class_names,
        shuffle=shuffle,
        seed=SEED if shuffle else None
    )
    if gen.samples == 0:
        raise ValueError(f"No images found in {dataset_dir}")
    return gen

class PairGenerator(tf.keras.utils.Sequence):
    """Feeds the same image tensor to both branches of the fusion model."""
    def __init__(self, image_generator):
        super().__init__()
        self.image_generator = image_generator

    def __len__(self):
        return len(self.image_generator)

    def __getitem__(self, index):
        x, y = self.image_generator[index]
        return (x, x), y

    @property
    def samples(self):
        return self.image_generator.samples

    @property
    def classes(self):
        return self.image_generator.classes

    @property
    def class_indices(self):
        return self.image_generator.class_indices

# ------------------------- FUSION MODEL -------------------------
def create_base_model(base_model, output_dim=128, freeze_until=400):
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(output_dim, activation="relu")(x)
    return Model(inputs=base_model.input, outputs=x)

def build_fusion_model(num_classes):
    input_a = Input(shape=(*IMAGE_SIZE, 3), name="densenet_image")
    input_b = Input(shape=(*IMAGE_SIZE, 3), name="efficientnet_image")

    dense_base = DenseNet201(
        weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3)
    )
    efficient_base = EfficientNetV2L(
        weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3),
        include_preprocessing=False
    )

    branch_a = create_base_model(dense_base)
    branch_b = create_base_model(efficient_base)

    feat_a = branch_a(input_a)
    feat_b = branch_b(input_b)
    merged = Concatenate(name="feature_fusion")([feat_a, feat_b])
    x = BatchNormalization()(merged)
    x = Dense(FUSION_DENSE_UNITS, activation="relu")(x)
    x = Dropout(FUSION_DROPOUT)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu", name="fusion_embedding")(x)
    output = Dense(num_classes, activation="softmax", name="class_output")(x)

    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=FUSION_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def extract_features(model, pair_generator):
    embedding_model = Model(
        inputs=model.inputs,
        outputs=model.get_layer("fusion_embedding").output
    )
    features, labels = [], []
    # Iterate underlying generator directly to avoid Sequence adapter ambiguity.
    image_gen = pair_generator.image_generator
    for idx in range(len(image_gen)):
        x, y = image_gen[idx]
        f = embedding_model.predict([x, x], verbose=0)
        features.append(f)
        labels.append(y)
    if not features:
        raise ValueError("Feature extraction produced no batches.")
    return np.vstack(features), np.concatenate(labels).astype(int)

# ------------------------- GREY WOLF OPTIMIZATION -------------------------
def gwo(objective, bounds, num_agents=GWO_AGENTS, max_iter=GWO_ITERS):
    dim = len(bounds)
    positions = np.array([
        [np.random.uniform(low, high) for low, high in bounds]
        for _ in range(num_agents)
    ])
    alpha_pos = positions[0].copy()
    beta_pos = positions[min(1, num_agents - 1)].copy()
    delta_pos = positions[min(2, num_agents - 1)].copy()
    alpha_score = beta_score = delta_score = float("inf")

    for iteration in range(max_iter):
        for i in range(num_agents):
            score = objective(positions[i])
            if score < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = score, positions[i].copy()
            elif score < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = score, positions[i].copy()
            elif score < delta_score:
                delta_score, delta_pos = score, positions[i].copy()

        a = 2.0 - 2.0 * iteration / max(1, max_iter - 1)
        for i in range(num_agents):
            candidates = []
            for leader in (alpha_pos, beta_pos, delta_pos):
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - positions[i])
                candidates.append(leader - A * D)
            positions[i] = np.array([
                np.clip(np.mean([c[j] for c in candidates]), bounds[j][0], bounds[j][1])
                for j in range(dim)
            ])
        print(f"GWO iteration {iteration + 1}/{max_iter}: best macro-F1 objective={alpha_score:.5f}")
    return alpha_pos, alpha_score

# ------------------------- EVALUATION / FIGURES -------------------------
def safe_macro_auroc(y_true, probabilities, n_classes):
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    if y_bin.shape[1] != n_classes:
        return float("nan")
    try:
        return float(roc_auc_score(y_bin, probabilities, average="macro"))
    except ValueError:
        return float("nan")

def save_confusion_matrix(y_true, y_pred, class_names, split_name):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", values_format="d", xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"{split_name} Confusion Matrix")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{split_name.lower()}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def save_roc_plot(y_true, probabilities, class_names, split_name):
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    fig, ax = plt.subplots(figsize=(8, 6))
    auc_values = []
    for i, class_name in enumerate(class_names):
        # ROC is undefined if the split has no positive or no negative samples for a class.
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
        class_auc = auc(fpr, tpr)
        auc_values.append(class_auc)
        ax.plot(fpr, tpr, lw=1.8, label=f"{class_name} (AUC={class_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"{split_name} ROC Curves")
    ax.grid(alpha=0.25)
    if auc_values:
        ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"roc_{split_name.lower()}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return safe_macro_auroc(y_true, probabilities, n_classes), path

def score_model(y_true, y_pred, probabilities, split_name, class_names):
    result = {
        "split": split_name,
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_auroc_ovr": safe_macro_auroc(y_true, probabilities, len(class_names))
    }
    print(f"\\n{split_name} results")
    for key, value in result.items():
        if key not in ("split", "samples"):
            print(f"  {key}: {value:.4f}")
    print(classification_report(
        y_true, y_pred, labels=np.arange(len(class_names)),
        target_names=class_names, zero_division=0
    ))
    save_confusion_matrix(y_true, y_pred, class_names, split_name)
    _, roc_path = save_roc_plot(y_true, probabilities, class_names, split_name)
    print(f"Saved ROC: {roc_path}")
    return result

# ------------------------- MAIN PIPELINE -------------------------
def main():
    start_total = time.time()
    class_files = discover_classes(DATASET_ROOT)
    class_names = list(class_files.keys())

    print("\\nSTEP 1: Split unique original images before oversampling")
    split_original_files(class_files, SPLIT_ROOT)

    print("\\nSTEP 2: Balance training set only")
    oversample_training_only(SPLIT_ROOT, BALANCED_TRAIN_ROOT)

    print("\\nSTEP 3: Create generators")
    train_gen = PairGenerator(create_data_generator(
        BALANCED_TRAIN_ROOT, class_names, BATCH_SIZE, shuffle=True
    ))
    val_gen = PairGenerator(create_data_generator(
        os.path.join(SPLIT_ROOT, "val"), class_names, BATCH_SIZE, shuffle=False
    ))
    test_gen = PairGenerator(create_data_generator(
        os.path.join(SPLIT_ROOT, "test"), class_names, BATCH_SIZE, shuffle=False
    ))

    print(f"Class mapping: {train_gen.class_indices}")
    with open(os.path.join(REPORTS_DIR, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    print("\\nSTEP 4: Train DenseNet201 + EfficientNetV2L fusion model")
    model = build_fusion_model(num_classes=len(class_names))
    model.summary()
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, "best_fusion_model.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        )
    ]
    t0 = time.time()
    history = model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS,
        callbacks=callbacks, verbose=1
    )
    fusion_training_seconds = time.time() - t0
    model.save(os.path.join(MODEL_DIR, "final_fusion_model.keras"))

    # Save training curves.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get("accuracy", []), label="Train")
    axes[0].plot(history.history.get("val_accuracy", []), label="Validation")
    axes[0].set_title("Fusion Model Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend()
    axes[1].plot(history.history.get("loss", []), label="Train")
    axes[1].plot(history.history.get("val_loss", []), label="Validation")
    axes[1].set_title("Fusion Model Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fusion_training_curves.png"), dpi=300)
    plt.close(fig)

    print("\\nSTEP 5: Extract fusion embeddings")
    t0 = time.time()
    X_train, y_train = extract_features(model, train_gen)
    X_val, y_val = extract_features(model, val_gen)
    X_test, y_test = extract_features(model, test_gen)
    feature_extraction_seconds = time.time() - t0
    print("Feature shapes:", X_train.shape, X_val.shape, X_test.shape)

    print("\\nSTEP 6: GWO tunes CatBoost using validation macro-F1")
    classes = np.arange(len(class_names))
    class_weights_array = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_array)}

    def objective(params):
        learning_rate = float(params[0])
        depth = int(np.clip(round(params[1]), 4, 10))
        l2_leaf_reg = float(params[2])
        candidate = CatBoostClassifier(
            iterations=CATBOOST_ITERATIONS,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function="MultiClass",
            eval_metric="TotalF1",
            class_weights=class_weights,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False
        )
        candidate.fit(X_train, y_train)
        pred = np.asarray(candidate.predict(X_val)).reshape(-1).astype(int)
        return 1.0 - f1_score(y_val, pred, average="macro", zero_division=0)

    t0 = time.time()
    best_pos, best_objective = gwo(
        objective, bounds=[(0.01, 0.2), (4, 10), (1, 10)],
        num_agents=GWO_AGENTS, max_iter=GWO_ITERS
    )
    gwo_seconds = time.time() - t0
    best_lr = float(best_pos[0])
    best_depth = int(np.clip(round(best_pos[1]), 4, 10))
    best_l2 = float(best_pos[2])
    print(
        f"Best GWO parameters: learning_rate={best_lr:.5f}, "
        f"depth={best_depth}, l2_leaf_reg={best_l2:.4f}; "
        f"validation macro-F1 objective={1-best_objective:.4f}"
    )

    print("\\nSTEP 7: Fit final CatBoost model on training embeddings")
    cat_model = CatBoostClassifier(
        iterations=CATBOOST_ITERATIONS,
        learning_rate=best_lr,
        depth=best_depth,
        l2_leaf_reg=best_l2,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        class_weights=class_weights,
        random_seed=SEED,
        verbose=100,
        allow_writing_files=False
    )
    t0 = time.time()
    cat_model.fit(X_train, y_train)
    catboost_training_seconds = time.time() - t0
    cat_model.save_model(os.path.join(MODEL_DIR, "catboost_fusion_model.cbm"))

    print("\\nSTEP 8: Evaluate train, validation and independent test")
    results = []
    for split_name, X, y in (
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test)
    ):
        t0 = time.time()
        probabilities = cat_model.predict_proba(X)
        prediction_seconds = time.time() - t0
        predictions = np.argmax(probabilities, axis=1)
        result = score_model(y, predictions, probabilities, split_name, class_names)
        result["prediction_seconds"] = prediction_seconds
        results.append(result)

    metrics_path = os.path.join(REPORTS_DIR, "model_metrics.csv")
    fieldnames = list(results[0].keys())
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    timing = [
        {"stage": "fusion_model_training", "seconds": fusion_training_seconds},
        {"stage": "feature_extraction_all_splits", "seconds": feature_extraction_seconds},
        {"stage": "gwo_search", "seconds": gwo_seconds},
        {"stage": "catboost_training", "seconds": catboost_training_seconds},
        {"stage": "total_pipeline", "seconds": time.time() - start_total}
    ]
    timing_path = os.path.join(REPORTS_DIR, "timing_summary.csv")
    with open(timing_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "seconds"])
        writer.writeheader()
        writer.writerows(timing)

    print("\\nSaved outputs:")
    print(f"  Metrics CSV: {metrics_path}")
    print(f"  Timing CSV:  {timing_path}")
    print(f"  Plots:       {PLOTS_DIR}")
    print(f"  Models:      {MODEL_DIR}")
    print("\\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
