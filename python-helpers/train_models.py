#!/usr/bin/env python3
"""
Train the two-stage milk adulteration models and export pure-C inference code.

Stage 1: XGBoost binary classifier, target Is_Adulterated (0/1).
Stage 2: Random Forest classifier, target Adulterant_Detected, trained only on
         adulterated samples (Starch / Detergent / Water).

The supplied data is synthetic and assumption-driven. Metrics describe only
performance on this generated distribution, not real milk.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from xgboost import XGBClassifier

FEATURES = ["pH", "Temperature_C", "Turbidity_NTU"]
CLASS_NAMES = ["Normal", "Starch", "Detergent", "Water"]
ADULTERANT_NAMES = ["Starch", "Detergent", "Water"]
SEED = 20261011


def load_split(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a CSV and its aligned batch IDs from generation_metadata.csv."""
    csv_path = data_dir / f"milk_synthetic_{split}.csv"
    metadata_path = data_dir / "generation_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {csv_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}; batch-aware CV requires the metadata file."
        )

    frame = pd.read_csv(csv_path)
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.loc[metadata["split"] == split].sort_values("row_index")
    metadata = metadata.reset_index(drop=True)

    if len(frame) != len(metadata):
        raise ValueError(
            f"{split}: CSV has {len(frame)} rows but metadata has {len(metadata)}. "
            "Check that the CSV and metadata came from the same dataset generation."
        )
    missing = set(FEATURES + ["Is_Adulterated", "Adulterant_Detected"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    if frame[FEATURES].isna().any().any() or not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError(f"{csv_path} has missing or non-finite feature values.")
    if not np.array_equal(
        frame["Is_Adulterated"].astype(int).to_numpy(),
        (frame["Adulterant_Detected"].astype(str) != "Normal").astype(int).to_numpy(),
    ):
        raise ValueError(f"{split}: target columns are inconsistent.")
    return frame, metadata


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _json_tree_to_flat(tree_json, feature_names):
    """Flatten an XGBoost JSON tree into nodes with global integer child indices."""
    nodes = []

    def visit(node):
        idx = len(nodes)
        nodes.append(None)
        if "leaf" in node:
            nodes[idx] = {
                "feature": -1, "left": -1, "right": -1,
                "threshold": 0.0, "value": float(node["leaf"]),
            }
            return idx
        split = str(node["split"])
        if split in feature_names:
            feature = feature_names.index(split)
        elif split.startswith("f") and split[1:].isdigit():
            feature = int(split[1:])
        else:
            raise ValueError(f"Unknown XGBoost split feature {split!r}")
        children = {int(child["nodeid"]): child for child in node["children"]}
        yes = int(node["yes"])
        no = int(node["no"])
        left = visit(children[yes])
        right = visit(children[no])
        nodes[idx] = {
            "feature": feature, "left": left, "right": right,
            "threshold": float(node["split_condition"]), "value": 0.0,
        }
        return idx

    root = visit(tree_json)
    return nodes, root


def export_xgboost_c(model, output_c: Path, header_path: Path, feature_names):
    """Export XGBoost trees to compact standalone C; output[0]=P(normal), [1]=P(adulterated)."""
    import math
    booster = model.get_booster()
    tree_jsons = [json.loads(t) for t in booster.get_dump(dump_format="json")]
    all_nodes, roots = [], []
    for tree in tree_jsons:
        nodes, root = _json_tree_to_flat(tree, feature_names)
        offset = len(all_nodes)
        # Adjust child references to the global flat node array.
        for n in nodes:
            if n["feature"] >= 0:
                n["left"] += offset
                n["right"] += offset
        roots.append(root + offset)
        all_nodes.extend(nodes)

    config = json.loads(booster.save_config())
    raw_base = config["learner"]["learner_model_param"]["base_score"]
    if isinstance(raw_base, str):
        raw_base = raw_base.strip("[]").split(",")[0]
    base_prob = float(raw_base)
    if not (0.0 < base_prob < 1.0):
        raise ValueError(f"Unexpected logistic XGBoost base_score: {base_prob}")
    base_margin = math.log(base_prob / (1.0 - base_prob))

    c = [
        '#include "milk_xgb.h"', '#include <math.h>', '#include <stddef.h>', '',
        'typedef struct { int feature; int left; int right; double threshold; double value; } XgbNode;',
        'static const XgbNode XGB_NODES[] = {'
    ]
    for n in all_nodes:
        c.append('    {%d, %d, %d, %.17g, %.17g},' % (
            n["feature"], n["left"], n["right"], n["threshold"], n["value"]))
    c += ['};', f'static const int XGB_ROOTS[{len(roots)}] = {{' + ', '.join(map(str, roots)) + '};',
          f'static const int XGB_TREE_COUNT = {len(roots)};',
          f'static const double XGB_BASE_MARGIN = {base_margin:.17g};', '',
          'int milk_xgb_score(const double input[3], double output[2]) {',
          '    if (input == NULL || output == NULL) return -1;',
          '    for (int i = 0; i < 3; ++i) if (!isfinite(input[i])) return -2;',
          '    double margin = XGB_BASE_MARGIN;',
          '    for (int t = 0; t < XGB_TREE_COUNT; ++t) {',
          '        int node = XGB_ROOTS[t];',
          '        while (XGB_NODES[node].feature >= 0) {',
          '            const XgbNode *n = &XGB_NODES[node];',
          '            node = ((float)input[n->feature] < (float)n->threshold) ? n->left : n->right;',
          '        }',
          '        margin += XGB_NODES[node].value;',
          '    }',
          '    double p = 1.0 / (1.0 + exp(-margin));',
          '    output[0] = 1.0 - p;', '    output[1] = p;', '    return 0;', '}', '']
    output_c.write_text('\n'.join(c), encoding='utf-8')
    header_path.write_text("""#pragma once
#ifdef __cplusplus
extern "C" {
#endif
/* Inputs: pH, Temperature_C, Turbidity_NTU. Returns 0 on success.
 * output[0] = P(Normal), output[1] = P(Adulterated). No heap allocation.
 * Returns -1 for null pointers and -2 for non-finite inputs.
 */
int milk_xgb_score(const double input[3], double output[2]);
#ifdef __cplusplus
}
#endif
""", encoding='utf-8')


def export_random_forest_c(model, output_c: Path, header_path: Path, feature_names):
    """Export sklearn RandomForestClassifier to portable C arrays and a no-heap scorer."""
    classes = [str(x) for x in model.classes_]
    if len(classes) != 3:
        raise ValueError(f"Expected 3 RF classes, got {classes}")
    nodes = []
    roots = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        offset = len(nodes)
        roots.append(offset)
        for i in range(tree.node_count):
            feature = int(tree.feature[i])
            left = int(tree.children_left[i])
            right = int(tree.children_right[i])
            if feature >= 0:
                nodes.append({
                    "feature": feature, "left": offset + left, "right": offset + right,
                    "threshold": float(tree.threshold[i]), "probs": [0.0, 0.0, 0.0],
                })
            else:
                values = np.asarray(tree.value[i][0], dtype=float).reshape(-1)
                total = float(values.sum())
                probs = (values / total).tolist() if total > 0 else [1.0 / 3.0] * 3
                # Align the tree output to model.classes_ order. sklearn trees use
                # the classifier's class order; explicitly pad defensively.
                probs = (probs + [0.0] * 3)[:3]
                nodes.append({
                    "feature": -1, "left": -1, "right": -1,
                    "threshold": 0.0, "probs": probs,
                })

    c = [
        '#include "milk_rf.h"', '#include <math.h>', '#include <stddef.h>', '',
        'typedef struct { int feature; int left; int right; double threshold; double p0; double p1; double p2; } RfNode;',
        'static const RfNode RF_NODES[] = {'
    ]
    for n in nodes:
        p0, p1, p2 = n["probs"]
        c.append('    {%d, %d, %d, %.17g, %.17g, %.17g, %.17g},' % (
            n["feature"], n["left"], n["right"], n["threshold"], p0, p1, p2))
    c += ['};', f'static const int RF_ROOTS[{len(roots)}] = {{' + ', '.join(map(str, roots)) + '};',
          f'static const int RF_TREE_COUNT = {len(roots)};', '',
          'int milk_rf_score(const double input[3], double output[3]) {',
          '    if (input == NULL || output == NULL) return -1;',
          '    for (int i = 0; i < 3; ++i) if (!isfinite(input[i])) return -2;',
          '    output[0] = output[1] = output[2] = 0.0;',
          '    for (int t = 0; t < RF_TREE_COUNT; ++t) {',
          '        int node = RF_ROOTS[t];',
          '        while (RF_NODES[node].feature >= 0) {',
          '            const RfNode *n = &RF_NODES[node];',
          '            node = ((double)(float)input[n->feature] <= n->threshold) ? n->left : n->right;',
          '        }',
          '        output[0] += RF_NODES[node].p0;',
          '        output[1] += RF_NODES[node].p1;',
          '        output[2] += RF_NODES[node].p2;',
          '    }',
          '    for (int k = 0; k < 3; ++k) output[k] /= (double)RF_TREE_COUNT;',
          '    return 0;', '}', '']
    output_c.write_text('\n'.join(c), encoding='utf-8')
    header_path.write_text("""#pragma once
#ifdef __cplusplus
extern "C" {
#endif
/* Inputs: pH, Temperature_C, Turbidity_NTU. Returns 0 on success.
 * Outputs are class probabilities in model_metadata.json stage2.class_order.
 * No heap allocation. Returns -1 for null pointers and -2 for non-finite inputs.
 */
int milk_rf_score(const double input[3], double output[3]);
#ifdef __cplusplus
}
#endif
""", encoding='utf-8')


def verify_c_exports(out_dir: Path, xgb_model, rf_model, X_test, max_samples=100):
    """Compile generated C with a host compiler and compare C scores to Python."""
    import ctypes
    import shutil
    import subprocess
    import tempfile
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        return {"status": "skipped", "reason": "No C compiler (cc/gcc/clang) found"}
    esp = out_dir / "esp32"
    with tempfile.TemporaryDirectory() as tmp:
        lib = Path(tmp) / "libmilk_models.so"
        command = [compiler, "-shared", "-fPIC", "-O2", str(esp / "milk_xgb.c"),
                   str(esp / "milk_rf.c"), "-lm", "-o", str(lib)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except Exception as exc:
            return {"status": "failed", "reason": str(exc)}
        if result.returncode != 0:
            return {"status": "failed", "compiler_stderr": result.stderr[-4000:]}
        dll = ctypes.CDLL(str(lib))
        Double3 = ctypes.c_double * 3
        Double2 = ctypes.c_double * 2
        dll.milk_xgb_score.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        dll.milk_xgb_score.restype = ctypes.c_int
        dll.milk_rf_score.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        dll.milk_rf_score.restype = ctypes.c_int
        subset = X_test.iloc[:min(max_samples, len(X_test))]
        xgb_python = xgb_model.predict_proba(subset)[:, 1]
        rf_python = rf_model.predict_proba(subset)
        max_xgb_error = 0.0
        max_rf_error = 0.0
        for i, row in enumerate(subset.to_numpy(dtype=float)):
            x = Double3(*[float(v) for v in row])
            out2 = Double2()
            out3 = Double3()
            if dll.milk_xgb_score(x, out2) != 0 or dll.milk_rf_score(x, out3) != 0:
                return {"status": "failed", "reason": f"C inference returned error at row {i}"}
            max_xgb_error = max(max_xgb_error, abs(float(out2[1]) - float(xgb_python[i])))
            max_rf_error = max(max_rf_error, max(abs(float(out3[j]) - float(rf_python[i, j])) for j in range(3)))
        tolerance = 1e-6
        status = "passed" if max(max_xgb_error, max_rf_error) <= tolerance else "failed"
        return {"status": status, "samples_checked": len(subset),
                "max_xgboost_probability_absolute_error": max_xgb_error,
                "max_random_forest_probability_absolute_error": max_rf_error,
                "tolerance": tolerance,
                "compiler": Path(compiler).name,
                "note": "Host compiler parity test; still compile and test on ESP-IDF target."}


def choose_binary_threshold(y_true: np.ndarray, p_positive: np.ndarray, min_specificity: float = 0.5) -> dict:
    """
    Select threshold using validation F2. F2 weights recall more than precision.
    If tied, choose the threshold with greater recall, then greater specificity.
    """
    candidates = np.unique(np.concatenate(([0.001, 0.5, 0.999], p_positive)))
    rows = []
    for threshold in candidates:
        pred = (p_positive >= threshold).astype(int)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        tn = int(np.sum((y_true == 0) & (pred == 0)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append((float(f2), float(recall), float(specificity), float(precision), float(threshold)))
    # Prefer F2 while enforcing a minimum normal-milk specificity, to prevent a
    # degenerate always-adulterated classifier. If the validation probabilities
    # cannot meet the floor, fall back to all thresholds and make that explicit.
    eligible = [r for r in rows if r[2] >= min_specificity]
    used_fallback = not eligible
    pool = eligible if eligible else rows
    best = max(pool, key=lambda r: (r[0], r[1], r[2], r[3]))
    return {
        "threshold": best[4],
        "validation_f2": best[0],
        "validation_recall": best[1],
        "validation_specificity": best[2],
        "validation_precision": best[3],
        "selection_rule": "maximize F2 (beta=2) among thresholds meeting minimum validation specificity; ties prefer recall, specificity, precision",
        "minimum_validation_specificity": float(min_specificity),
        "specificity_constraint_fallback_used": used_fallback,
    }


def metric_report(y_true, y_pred, labels, target_names) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=target_names,
            output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing the three CSV splits and generation_metadata.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("milk_model_output"))
    parser.add_argument("--xgb-iterations", type=int, default=40,
                        help="RandomizedSearchCV candidates for XGBoost (default: 40)")
    parser.add_argument("--rf-iterations", type=int, default=40,
                        help="RandomizedSearchCV candidates for Random Forest (default: 40)")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-specificity", type=float, default=0.5,
                        help="Minimum normal-class specificity allowed when choosing the validation threshold (default: 0.5)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    if not 0.0 <= args.min_specificity <= 1.0:
        parser.error("--min-specificity must be between 0 and 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    deployment_dir = args.out_dir / "esp32"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    train, train_meta = load_split(args.data_dir, "train")
    valid, valid_meta = load_split(args.data_dir, "validation")
    test, test_meta = load_split(args.data_dir, "test")

    # Guard against accidental batch leakage. The provided split is designed to
    # have disjoint batch IDs, and CV below groups by batch as well.
    batch_sets = {
        name: set(meta["batch_id"].astype(str))
        for name, meta in [
            ("train", train_meta), ("validation", valid_meta), ("test", test_meta)
        ]
    }
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = batch_sets[a] & batch_sets[b]
        if overlap:
            raise ValueError(f"Batch leakage: {a}/{b} share IDs: {sorted(overlap)[:5]}")

    X_train = train[FEATURES]
    X_valid = valid[FEATURES]
    X_test = test[FEATURES]
    yb_train = train["Is_Adulterated"].astype(int).to_numpy()
    yb_valid = valid["Is_Adulterated"].astype(int).to_numpy()
    yb_test = test["Is_Adulterated"].astype(int).to_numpy()
    groups_binary = train_meta["batch_id"].astype(str).to_numpy()

    print("Dataset sizes:", {"train": len(train), "validation": len(valid), "test": len(test)})
    print("Feature order:", FEATURES)
    print("Batch-disjoint splits verified.")

    # ---------------- Stage 1: XGBoost binary detector ----------------
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=1,
        verbosity=0,
    )
    xgb_space = {
        "n_estimators": [100, 160, 220, 300, 400, 500],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.015, 0.025, 0.04, 0.06, 0.08, 0.12],
        "min_child_weight": [1, 2, 4, 6, 8],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "gamma": [0.0, 0.05, 0.15, 0.3],
        "reg_alpha": [0.0, 0.001, 0.01, 0.1, 0.5],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
        "scale_pos_weight": [0.5, 0.75, 1.0, 1.25, 1.5],
    }
    cv_binary = StratifiedGroupKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seed
    )
    xgb_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_space,
        n_iter=args.xgb_iterations,
        scoring="balanced_accuracy",
        cv=cv_binary,
        refit=True,
        random_state=args.seed,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )
    xgb_search.fit(X_train, yb_train, groups=groups_binary)
    xgb_model = xgb_search.best_estimator_

    # Tune decision threshold on validation only; do not use test data to choose it.
    p_valid_binary = xgb_model.predict_proba(X_valid)[:, 1]
    threshold_info = choose_binary_threshold(yb_valid, p_valid_binary, args.min_specificity)
    binary_threshold = float(threshold_info["threshold"])

    p_test_binary = xgb_model.predict_proba(X_test)[:, 1]
    pred_test_binary = (p_test_binary >= binary_threshold).astype(int)
    binary_metrics = metric_report(yb_test, pred_test_binary, [0, 1], ["Normal", "Adulterated"])
    binary_metrics.update({
        "threshold": binary_threshold,
        "validation_threshold_selection": threshold_info,
        "best_group_cv_balanced_accuracy": float(xgb_search.best_score_),
        "best_params": xgb_search.best_params_,
        "test_probability_mean_by_true_class": {
            str(c): float(p_test_binary[yb_test == c].mean()) for c in [0, 1]
        },
    })
    xgb_model.save_model(str(args.out_dir / "xgboost_binary.json"))
    import joblib
    joblib.dump(xgb_model, args.out_dir / "xgboost_binary.joblib")

    # ---------------- Stage 2: Random Forest adulterant ID ----------------
    tr_ad = train.loc[train["Is_Adulterated"].astype(int) == 1].copy()
    va_ad = valid.loc[valid["Is_Adulterated"].astype(int) == 1].copy()
    te_ad = test.loc[test["Is_Adulterated"].astype(int) == 1].copy()
    tr_ad_meta = train_meta.loc[train["Is_Adulterated"].astype(int).to_numpy() == 1]
    y3_train = tr_ad["Adulterant_Detected"].astype(str).to_numpy()
    y3_valid = va_ad["Adulterant_Detected"].astype(str).to_numpy()
    y3_test = te_ad["Adulterant_Detected"].astype(str).to_numpy()
    groups_rf = tr_ad_meta["batch_id"].astype(str).to_numpy()

    rf = RandomForestClassifier(random_state=args.seed, n_jobs=1)
    rf_space = {
        "n_estimators": [200, 300, 400, 500, 650, 800],
        "max_depth": [None, 4, 6, 8, 10, 14, 18],
        "min_samples_split": [2, 4, 8, 12, 20],
        "min_samples_leaf": [1, 2, 3, 4, 6, 8],
        "max_features": ["sqrt", 0.75, 1.0],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "criterion": ["gini", "entropy", "log_loss"],
        "bootstrap": [True, False],
    }
    cv_rf = StratifiedGroupKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seed + 1
    )
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_space,
        n_iter=args.rf_iterations,
        scoring="balanced_accuracy",
        cv=cv_rf,
        refit=True,
        random_state=args.seed + 1,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )
    rf_search.fit(tr_ad[FEATURES], y3_train, groups=groups_rf)
    rf_model = rf_search.best_estimator_

    pred_valid_rf = rf_model.predict(va_ad[FEATURES])
    pred_test_rf = rf_model.predict(te_ad[FEATURES])
    rf_metrics = metric_report(
        y3_test, pred_test_rf, ADULTERANT_NAMES, ADULTERANT_NAMES
    )
    rf_metrics.update({
        "best_group_cv_balanced_accuracy": float(rf_search.best_score_),
        "best_params": rf_search.best_params_,
        "validation_metrics": metric_report(
            y3_valid, pred_valid_rf, ADULTERANT_NAMES, ADULTERANT_NAMES
        ),
    })
    joblib.dump(rf_model, args.out_dir / "random_forest_adulterant.joblib")

    # ---------------- Full cascade evaluation on untouched test split ----------------
    # First stage chooses whether the sample is normal. Only samples predicted
    # adulterated are passed to the second-stage model.
    cascade_predictions = []
    for idx, row in test.iterrows():
        p_ad = float(p_test_binary[idx])
        if p_ad < binary_threshold:
            cascade_predictions.append("Normal")
        else:
            class_probs = rf_model.predict_proba(X_test.iloc[[idx]])[0]
            cascade_predictions.append(str(rf_model.classes_[int(np.argmax(class_probs))]))
    cascade_predictions = np.asarray(cascade_predictions, dtype=object)
    y4_test = test["Adulterant_Detected"].astype(str).to_numpy()
    cascade_metrics = metric_report(y4_test, cascade_predictions, CLASS_NAMES, CLASS_NAMES)
    cascade_metrics["class_order"] = CLASS_NAMES
    cascade_metrics["note"] = (
        "Full two-stage pipeline on the untouched test split. A false-negative at "
        "stage 1 prevents stage 2 from identifying the adulterant."
    )

    # Save reports and training metadata.
    reports = {
        "warning": (
            "Synthetic, assumption-driven data only. These metrics do not establish "
            "real-world accuracy on milk or sensor hardware."
        ),
        "features_in_exact_order": FEATURES,
        "stage1_binary_xgboost": binary_metrics,
        "stage2_random_forest_adulterant_only": rf_metrics,
        "full_cascade_four_class_test": cascade_metrics,
        "class_mapping": {
            "binary": {"0": "Normal", "1": "Adulterated"},
            "random_forest_classes_in_model_order": [str(c) for c in rf_model.classes_],
            "cascade_classes": CLASS_NAMES,
        },
        "data_rows": {"train": len(train), "validation": len(valid), "test": len(test)},
        "batch_counts": {k: len(v) for k, v in batch_sets.items()},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(args.out_dir / "metrics.json", reports)

    # Model metadata for firmware integration.
    try:
        import xgboost
        xgb_version = xgboost.__version__
    except Exception:
        xgb_version = "unknown"
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "unknown"
    deployment_metadata = {
        "warning": "Synthetic-data model; not validated for real milk.",
        "feature_order": FEATURES,
        "input_type": "double",
        "expected_input_units": {
            "pH": "pH sensor output",
            "Temperature_C": "degrees Celsius",
            "Turbidity_NTU": (
                "nominal synthetic feature scale; not validated true NTU. "
                "Must be recalibrated against the actual sensor."
            ),
        },
        "stage1": {
            "function": "milk_xgb_score",
            "source": "milk_xgb.c",
            "header": "milk_xgb.h",
            "output": "two class probabilities/scores, class index 0=Normal, 1=Adulterated",
            "adulterated_probability_index": 1,
            "positive_threshold": binary_threshold,
            "model": "XGBoost binary classifier",
        },
        "stage2": {
            "function": "milk_rf_score",
            "source": "milk_rf.c",
            "header": "milk_rf.h",
            "output": "three class probabilities/scores in the order below",
            "class_order": [str(c) for c in rf_model.classes_],
            "model": "Random Forest trained on adulterated samples only",
        },
        "versions": {"xgboost": xgb_version, "scikit_learn": sklearn_version},
    }
    write_json(deployment_dir / "model_metadata.json", deployment_metadata)

    # Generate a C header for the tuned threshold and exact class-index mapping,
    # so firmware does not rely on manually copied constants.
    firmware_class_codes = {"Normal": 0, "Starch": 1, "Detergent": 2, "Water": 3}
    config_lines = [
        "#pragma once",
        "/* Generated by train_models.py. Do not edit by hand. */",
        f"#define MILK_XGB_ADULTERATED_THRESHOLD {binary_threshold:.17g}",
    ]
    for i, cls in enumerate([str(c) for c in rf_model.classes_]):
        config_lines.append(f"#define MILK_RF_OUTPUT_CLASS_{i} {firmware_class_codes[cls]} /* {cls} */")
    (deployment_dir / "milk_model_config.h").write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    # ESP32 export: standalone, allocation-free C. No m2cgen or Python runtime
    # is required on the device.
    export_xgboost_c(xgb_model, deployment_dir / "milk_xgb.c",
                     deployment_dir / "milk_xgb.h", FEATURES)
    export_random_forest_c(rf_model, deployment_dir / "milk_rf.c",
                           deployment_dir / "milk_rf.h", FEATURES)
    c_export_verification = verify_c_exports(args.out_dir, xgb_model, rf_model, X_test)
    write_json(deployment_dir / "c_export_verification.json", c_export_verification)
    if c_export_verification.get("status") == "failed":
        raise RuntimeError("Generated C failed host compilation/parity test; see esp32/c_export_verification.json")

    # Save feature importances where available.
    importances = {
        "features": FEATURES,
        "xgboost_gain": {
            k: float(v) for k, v in xgb_model.get_booster().get_score(importance_type="gain").items()
        },
        "random_forest": {
            k: float(v) for k, v in zip(FEATURES, rf_model.feature_importances_)
        },
    }
    write_json(args.out_dir / "feature_importance.json", importances)

    # Export a representative sample for device-side smoke testing.
    sample = {
        "features": {name: float(test.iloc[0][name]) for name in FEATURES},
        "expected_stage1_probability_adulterated": float(p_test_binary[0]),
        "threshold": binary_threshold,
        "expected_stage1_prediction": int(pred_test_binary[0]),
        "expected_stage2_probabilities": [
            float(v) for v in rf_model.predict_proba(X_test.iloc[[0]])[0]
        ],
        "expected_stage2_class_order": [str(c) for c in rf_model.classes_],
        "expected_cascade_prediction": str(cascade_predictions[0]),
    }
    write_json(deployment_dir / "smoke_test_sample.json", sample)

    print("\n=== TEST RESULTS (synthetic distribution only) ===")
    print("Stage 1 XGBoost binary detector")
    print(classification_report(yb_test, pred_test_binary, target_names=["Normal", "Adulterated"], zero_division=0))
    print("Stage 2 Random Forest (adulterated rows only)")
    print(classification_report(y3_test, pred_test_rf, labels=ADULTERANT_NAMES, zero_division=0))
    print("Full cascade")
    print(classification_report(y4_test, cascade_predictions, labels=CLASS_NAMES, zero_division=0))
    print(f"\nArtifacts written to: {args.out_dir.resolve()}")
    print(f"ESP32 C exports written to: {deployment_dir.resolve()}")
    print("IMPORTANT: validate C output and sensor calibration before relying on device predictions.")


if __name__ == "__main__":
    main()

