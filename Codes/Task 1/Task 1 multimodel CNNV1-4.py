"""
Task 1 — Param‑Driven Multi‑Model Runner (V1–V4) — Checklist + Science Edition

What this script does
- Trains 1D CNN variants (V1–V4) on Kepler windows and compares them.
- Group‑aware split by kepid (prevents leakage).
- Class imbalance handled via class weights.
- EarlyStopping + ReduceLROnPlateau for stable training.
- Metrics per model: Accuracy, Precision, Recall, F1, ROC‑AUC, PR‑AUC, Brier.
- Thresholds per model: primary (max‑F1) + precision‑first (≥0.95) + recall‑first (≥0.90).
- Science recovery per model: % of KOI‑CONFIRMED kepids in the test split recovered.
- EDA figures (class balance, distributions, sample windows) once per run.
- Reliability (calibration) curves per model.
- Best‑model analysis: FP/FN waveform plots + simple saliency maps.
- Saves environment info (TensorFlow/NumPy versions, GPU) for reproducibility.

Author: Praven Kumar Savariraj (Student ID: 23089117)
Date: 25 Aug 2025
Python: 3.10+ | TensorFlow 2.14+ | NumPy | Pandas | scikit‑learn | Matplotlib
"""
from __future__ import annotations
import os, re, json, random, time, argparse, datetime, platform
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, ConfusionMatrixDisplay,
                             brier_score_loss)

# ===================== Defaults (edit if needed) =====================
DATA_DIR_DEFAULT   = r"D:\UH One drive\Kepler_Preprocessed"  # contains *_X.npy and *_y.npy
OUTPUT_DIR_DEFAULT = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task1"
KOI_PATH_DEFAULT   = r"D:\UH One drive\cumulative_2025.07.16_10.28.33.csv"

# ===================== Repro & setup =====================
SEED = 42
rs = np.random.RandomState(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===================== Helper utils =====================

def extract_kepid_from_fname(fname: str) -> str:
    m = re.search(r"kplr(\d+)-", os.path.basename(fname).lower())
    if not m:
        m2 = re.search(r"(\d+)", os.path.basename(fname))
        return m2.group(1).lstrip('0') if m2 else ""
    return m.group(1).lstrip('0')


def find_npy_pairs(data_dir: str) -> List[Tuple[str, str]]:
    xs = sorted(Path(data_dir).glob("*_X.npy"))
    pairs = []
    for xp in xs:
        yp = Path(str(xp).replace("_X.npy", "_y.npy"))
        if yp.exists():
            pairs.append((str(xp), str(yp)))
    return pairs


def load_files_sampled(pairs: List[Tuple[str, str]], frac: float) -> List[Tuple[str, str]]:
    if frac >= 1.0:
        return pairs
    k = max(1, int(len(pairs) * frac))
    idx = rs.choice(len(pairs), size=k, replace=False)
    return [pairs[i] for i in idx]


def cap_windows(X: np.ndarray, y: np.ndarray, max_per_file: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_per_file:
        return X, y
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep_neg = rs.choice(neg_idx, size=max(0, max_per_file - len(pos_idx)), replace=False) if len(pos_idx) < max_per_file else np.array([], dtype=int)
    keep = np.unique(np.concatenate([pos_idx, keep_neg]))
    return X[keep], y[keep]


def robust_read_koi(path: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    for opt in [dict(sep=","), dict(sep="|")]:
        try:
            df = pd.read_csv(path, **opt, comment="#")
            df.columns = df.columns.str.strip().str.lower()
            if "kepid" in df.columns and "koi_disposition" in df.columns:
                return df
        except Exception:
            pass
    return None


def load_confirmed_kepids(koi_path: str) -> set[str]:
    df = robust_read_koi(koi_path)
    if df is None: return set()
    conf = df[df["koi_disposition"].astype(str).str.upper().str.contains("CONFIRMED", na=False)]
    return set(conf["kepid"].astype(str).str.lstrip("0"))


def load_dataset(data_dir: str, frac: float, cap_per_file: int):
    pairs = find_npy_pairs(data_dir)
    if len(pairs) == 0:
        raise FileNotFoundError(f"No *_X.npy pairs found in: {data_dir}")
    pairs = load_files_sampled(pairs, frac)

    X_list, y_list, groups, kepids, src_files = [], [], [], [], []
    for Xp, Yp in pairs:
        X = np.load(Xp)  # (n_windows, T)
        y = np.load(Yp).astype(int)
        X, y = cap_windows(X, y, cap_per_file)
        kepid = extract_kepid_from_fname(Xp)
        groups.extend([kepid] * len(y))
        kepids.extend([kepid] * len(y))
        src_files.extend([os.path.basename(Xp)] * len(y))
        if X.ndim == 2:
            X = X[..., None]
        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.array(groups)
    kepids = np.array(kepids)
    src_files = np.array(src_files)
    return X, y, groups, kepids, src_files

# ===================== EDA =====================

def eda_overview(pairs: List[Tuple[str,str]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    counts = []
    means, stds, labels = [], [], []
    ex_samples = {0:[], 1:[]}

    for Xp, Yp in pairs[: min(20, len(pairs))]:  # sample some files for EDA
        X = np.load(Xp)
        y = np.load(Yp).astype(int)
        counts.append({'file': os.path.basename(Xp), 'n': len(y), 'pos': int((y==1).sum()), 'neg': int((y==0).sum())})
        m = X.mean(axis=1); s = X.std(axis=1)
        means.extend(m.tolist()); stds.extend(s.tolist()); labels.extend(y.tolist())
        for cls in [0,1]:
            idx = np.where(y==cls)[0][:5]
            ex_samples[cls].extend([X[i] for i in idx])

    pd.DataFrame(counts).to_csv(os.path.join(out_dir, 'file_counts.csv'), index=False)
    total_pos = int((np.array(labels)==1).sum())
    total_neg = int((np.array(labels)==0).sum())
    plt.figure(); plt.bar(['neg','pos'], [total_neg, total_pos])
    plt.title('Class balance (EDA subset)'); plt.ylabel('windows');
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'class_balance.png'), dpi=160); plt.close()

    plt.figure(); plt.hist(means, bins=50); plt.title('Distribution of window means'); plt.xlabel('mean'); plt.ylabel('count');
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'mean_hist.png'), dpi=160); plt.close()

    plt.figure(); plt.hist(stds, bins=50); plt.title('Distribution of window std'); plt.xlabel('std'); plt.ylabel('count');
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'std_hist.png'), dpi=160); plt.close()

    for cls in [0,1]:
        for i, arr in enumerate(ex_samples[cls][:10]):
            plt.figure(figsize=(6,2)); plt.plot(arr); plt.title(f'Example window (class {cls})');
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'sample_cls{cls}_{i}.png'), dpi=160); plt.close()

# ===================== Models =====================

def build_v1_plain(input_len: int, n_ch: int = 1) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, n_ch))
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="CNN_V1_Plain")


def residual_block(x, filters, k=3):
    sh = x
    x = layers.Conv1D(filters, k, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, k, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    if sh.shape[-1] != filters:
        sh = layers.Conv1D(filters, 1, padding='same')(sh)
    x = layers.Add()([x, sh]); x = layers.Activation('relu')(x)
    return x


def build_v2_residual(input_len: int, n_ch: int = 1) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, n_ch))
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = residual_block(x, 32)
    x = layers.MaxPooling1D(2)(x)
    x = residual_block(x, 64)
    x = layers.MaxPooling1D(2)(x)
    x = residual_block(x, 128)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="CNN_V2_Residual")


def se_block_1d(x, reduction=8):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(ch // reduction, activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    s = layers.Reshape((1, ch))(s)
    return layers.Multiply()([x, s])


def build_v3_se(input_len: int, n_ch: int = 1) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, n_ch))
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = se_block_1d(x)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = se_block_1d(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = se_block_1d(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="CNN_V3_SE")


def cbam_block_1d(x, reduction=8):
    ch = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    mlp = tf.keras.Sequential([
        layers.Dense(ch // reduction, activation='relu'),
        layers.Dense(ch, activation='sigmoid'),
    ])
    ch_att = layers.Add()([mlp(avg_pool), mlp(max_pool)])
    ch_att = layers.Reshape((1, ch))(ch_att)
    x = layers.Multiply()([x, ch_att])
    avg_pool_sp = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_pool_sp = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool_sp, max_pool_sp])
    sp_att = layers.Conv1D(1, 7, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([x, sp_att])


def build_v4_cbam(input_len: int, n_ch: int = 1) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, n_ch))
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = cbam_block_1d(x)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = cbam_block_1d(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = cbam_block_1d(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="CNN_V4_CBAM")

# ===================== Train/Eval helpers =====================

def compile_model(model: tf.keras.Model, lr: float):
    model.compile(
        optimizer=Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='auc_roc'), tf.keras.metrics.AUC(curve='PR', name='auc_pr')]
    )
    return model


def threshold_from_mode(y_true, y_prob, mode='max_f1', target=0.95):
    grid = np.linspace(0.05, 0.95, 19)
    if mode == 'max_f1':
        best_f1, best_t = -1, 0.5
        for t in grid:
            f1 = f1_score(y_true, (y_prob>=t).astype(int))
            if f1 > best_f1: best_f1, best_t = f1, float(t)
        return best_t
    if mode == 'target_precision':
        for t in grid:
            p = precision_score(y_true, (y_prob>=t).astype(int), zero_division=0)
            if p >= target: return float(t)
        return 0.95
    if mode == 'target_recall':
        for t in grid:
            r = recall_score(y_true, (y_prob>=t).astype(int), zero_division=0)
            if r >= target: return float(t)
        return 0.05
    return 0.5


def reliability_plot(y_true, y_prob, out_path: str, bins=12):
    bins = np.linspace(0.0, 1.0, bins+1)
    inds = np.digitize(y_prob, bins) - 1
    accs, confs = [], []
    for b in range(len(bins)-1):
        idx = np.where(inds==b)[0]
        if len(idx)==0: continue
        accs.append(y_true[idx].mean()); confs.append(y_prob[idx].mean())
    if accs:
        plt.figure(); plt.plot(confs, accs, marker='o'); plt.plot([0,1],[0,1],'--')
        plt.xlabel('Mean confidence'); plt.ylabel('Empirical accuracy'); plt.title('Reliability diagram')
        plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def error_analysis_plots(X_test, y_test, y_pred, out_dir: str, max_examples=20):
    os.makedirs(out_dir, exist_ok=True)
    idx_fp = np.where((y_pred==1) & (y_test==0))[0][:max_examples]
    idx_fn = np.where((y_pred==0) & (y_test==1))[0][:max_examples]
    pd.DataFrame({'FP_idx': idx_fp}).to_csv(os.path.join(out_dir, 'errors_fp_idx.csv'), index=False)
    pd.DataFrame({'FN_idx': idx_fn}).to_csv(os.path.join(out_dir, 'errors_fn_idx.csv'), index=False)
    for tag, idxs in [('fp', idx_fp), ('fn', idx_fn)]:
        for k, i in enumerate(idxs):
            arr = X_test[i,:,0]
            plt.figure(figsize=(6,2)); plt.plot(arr); plt.title(f'{tag.upper()} example {k}')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'{tag}_{k}.png'), dpi=160); plt.close()


def saliency_1d(model, X, idxs: List[int], out_dir: str, max_examples=10):
    os.makedirs(out_dir, exist_ok=True)
    idxs = idxs[:max_examples]
    for k, i in enumerate(idxs):
        x = tf.convert_to_tensor(X[i:i+1], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            p = model(x, training=False)
        grads = tape.gradient(p, x).numpy()[0,:,0]
        sig = X[i,:,0]
        plt.figure(figsize=(6,2)); plt.plot(sig); plt.twinx(); plt.plot(grads, alpha=0.6)
        plt.title('Signal & Saliency');
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'sal_{k}.png'), dpi=160); plt.close()


def science_recovery_from_predictions(pred_df: pd.DataFrame, confirmed_kepids: set[str], ycol: str = 'y_pred') -> dict:
    if not confirmed_kepids or pred_df.empty:
        return {"n_confirmed_in_test": 0, "n_recovered": 0, "n_missed": 0, "recovery_rate": None}
    kstr = pred_df['kepid'].astype(str).str.lstrip('0')
    conf_mask = kstr.isin(confirmed_kepids)
    confirmed_df = pred_df.loc[conf_mask].copy()
    if confirmed_df.empty:
        return {"n_confirmed_in_test": 0, "n_recovered": 0, "n_missed": 0, "recovery_rate": None}
    agg = confirmed_df.groupby(kstr[conf_mask]).agg(has_pos_pred=(ycol, 'max'))
    n_conf = int(len(agg)); n_rec = int((agg['has_pos_pred']>=1).sum()); n_mis = n_conf - n_rec
    rate = (n_rec / n_conf) if n_conf>0 else None
    return {"n_confirmed_in_test": n_conf, "n_recovered": n_rec, "n_missed": n_mis, "recovery_rate": rate}

# ===================== Per‑model run =====================

def run_single_model(name: str, builder, X, y, groups, kepids, src_files, out_root: str,
                     epochs: int, batch: int, lr: float, test_size: float,
                     thresh_mode: str, target_prec: float, koi_path: str) -> Tuple[Dict[str,Any], str, Dict[str,Any]]:
    # Grouped split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]
    kepid_test = kepids[te_idx]
    file_test  = src_files[te_idx]

    # Class weights
    cw = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}

    # Build/compile
    _, T, C = X.shape
    model = builder(T, C)
    compile_model(model, lr)

    # Output dirs
    run_dir = os.path.join(out_root, name)
    figs_dir = os.path.join(run_dir, 'figs'); os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, 'best.keras')

    # Callbacks
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', mode='max', patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc_pr', mode='max', patience=3, factor=0.5, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_auc_pr', mode='max', save_best_only=True),
    ]

    # Train
    t0 = time.perf_counter()
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     epochs=epochs, batch_size=batch, class_weight=class_weight,
                     verbose=1, callbacks=cbs)
    dur = time.perf_counter() - t0

    # Save history & curves
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(hist.history, f, indent=2)
    plt.figure(); plt.plot(hist.history['loss']); plt.plot(hist.history['val_loss']);
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'{name} — Loss'); plt.legend(['train','val']);
    plt.tight_layout(); plt.savefig(os.path.join(figs_dir,'lc_loss.png'), dpi=160); plt.close()
    plt.figure(); plt.plot(hist.history['accuracy']); plt.plot(hist.history['val_accuracy']);
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(f'{name} — Accuracy'); plt.legend(['train','val']);
    plt.tight_layout(); plt.savefig(os.path.join(figs_dir,'lc_acc.png'), dpi=160); plt.close()

    # Predict
    y_prob = model.predict(X_test).ravel()

    # Thresholds
    thr_primary = threshold_from_mode(y_test, y_prob, 'max_f1') if thresh_mode=='max_f1' else 0.5
    thr_prec    = threshold_from_mode(y_test, y_prob, 'target_precision', target=0.95)
    thr_recall  = threshold_from_mode(y_test, y_prob, 'target_recall',    target=0.90)
    threshold   = thr_primary if thresh_mode=='max_f1' else thr_prec

    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    metrics = {
        'model': name,
        'threshold_primary': float(thr_primary),
        'threshold_precision_first': float(thr_prec),
        'threshold_recall_first': float(thr_recall),
        'chosen_threshold': float(threshold),
        'train_seconds': round(dur, 2),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'pr_auc': float(average_precision_score(y_test, y_prob)),
        'brier': float(brier_score_loss(y_test, y_prob)),
    }

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plots: ROC, PR, CM, Reliability
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'{name} — ROC');
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir,'roc.png'), dpi=160); plt.close()
    except Exception: pass
    try:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.figure(); plt.plot(rec, prec); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'{name} — PR');
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir,'pr.png'), dpi=160); plt.close()
    except Exception: pass
    try:
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true')
        plt.title(f'{name} — Confusion Matrix (Norm)'); plt.tight_layout(); plt.savefig(os.path.join(figs_dir,'cm.png'), dpi=160); plt.close()
    except Exception: pass
    reliability_plot(y_test, y_prob, os.path.join(figs_dir,'reliability.png'), bins=12)

    # Predictions CSV (+ confirmed flag)
    pred = pd.DataFrame({
        'kepid': kepid_test,
        'source_file': file_test,
        'y_true': y_test,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'threshold': threshold,
        'model': name,
    })
    confirmed_kepids = load_confirmed_kepids(koi_path)
    if confirmed_kepids:
        pred['is_confirmed'] = pred['kepid'].astype(str).str.lstrip('0').isin(confirmed_kepids).astype(int)
    else:
        pred['is_confirmed'] = 0

    pred_csv = os.path.join(run_dir, 'predictions.csv')
    pred.to_csv(pred_csv, index=False)

    # Science recovery
    sci = science_recovery_from_predictions(pred, confirmed_kepids, ycol='y_pred')
    metrics.update({
        'science_confirmed_in_test': sci.get('n_confirmed_in_test', 0),
        'science_recovered': sci.get('n_recovered', 0),
        'science_missed': sci.get('n_missed', 0),
        'science_recovery_rate': sci.get('recovery_rate', None),
    })
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n📡 {name} Science Score")
    print(f"→ Confirmed planets in test sample: {metrics['science_confirmed_in_test']}")
    print(f"→ Recovered: {metrics['science_recovered']}")
    print(f"→ Missed: {metrics['science_missed']}")
    rr = metrics['science_recovery_rate']
    print(f"🎯 Recovery Rate: {rr:.2%}" if rr is not None else "⚠️ No confirmed planets found in test set.")

    eval_pack = {
        'X_test': X_test, 'y_test': y_test, 'y_prob': y_prob, 'y_pred': y_pred,
        'ckpt_path': ckpt_path, 'figs_dir': figs_dir, 'run_dir': run_dir,
    }
    return metrics, pred_csv, eval_pack

# ===================== Main runner =====================

def main():
    parser = argparse.ArgumentParser(description='Task1 Multi‑Model Runner (V1–V4) — Checklist + Science')
    parser.add_argument('--data', default=DATA_DIR_DEFAULT, help='Path to preprocessed *_X.npy and *_y.npy')
    parser.add_argument('--out',  default=OUTPUT_DIR_DEFAULT, help='Output root folder')
    parser.add_argument('--koi',  default=KOI_PATH_DEFAULT, help='KOI cumulative table path')
    parser.add_argument('--models', nargs='+', default=['v1','v2','v3','v4'], choices=['v1','v2','v3','v4'])
    parser.add_argument('--frac', type=float, default=0.05, help='Fraction of files to sample (0<frac<=1)')
    parser.add_argument('--cap',  type=int,   default=1000, help='Max windows kept per file (positives kept)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch',  type=int, default=256)
    parser.add_argument('--lr',     type=float, default=1e-3)
    parser.add_argument('--test_size', type=float, default=0.20)
    parser.add_argument('--thresh_mode', choices=['max_f1','target_precision','target_recall','fixed'], default='max_f1')
    parser.add_argument('--target_prec', type=float, default=0.95)
    args = parser.parse_args()

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = os.path.join(args.out, f'multi_run_{stamp}')
    os.makedirs(out_root, exist_ok=True)

    # Save environment info
    env = {
        'python': platform.python_version(),
        'tensorflow': tf.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'sklearn': 'scikit-learn',
        'gpu': [d.name for d in tf.config.list_physical_devices('GPU')],
        'seed': SEED,
    }
    with open(os.path.join(out_root, 'run_env.json'), 'w') as f:
        json.dump(env, f, indent=2)

    # EDA once per run
    eda_dir = os.path.join(out_root, 'EDA')
    pairs_all = find_npy_pairs(args.data)
    if len(pairs_all)==0:
        raise FileNotFoundError('No *_X.npy found in data path')
    pairs = load_files_sampled(pairs_all, args.frac)
    eda_overview(pairs, eda_dir)

    # Load dataset once
    X, y, groups, kepids, src_files = load_dataset(args.data, args.frac, args.cap)
    _, T, C = X.shape
    print(f"Loaded X={X.shape}; positives={(y==1).sum()} | negatives={(y==0).sum()}")

    # Model map
    builders = {
        'v1': build_v1_plain,
        'v2': build_v2_residual,
        'v3': build_v3_se,
        'v4': build_v4_cbam,
    }

    # Train all
    results = []
    pred_paths = []
    eval_packs = {}
    for key in args.models:
        name = {'v1':'V1_Plain','v2':'V2_Residual','v3':'V3_SE','v4':'V4_CBAM'}[key]
        print(f"\n=== Training {name} ===")
        metrics, pred_csv, eval_pack = run_single_model(name, builders[key], X, y, groups, kepids, src_files,
                                                        out_root, args.epochs, args.batch, args.lr, args.test_size,
                                                        args.thresh_mode, args.target_prec, args.koi)
        results.append(metrics)
        pred_paths.append(pred_csv)
        eval_packs[name] = eval_pack

    # Aggregate comparison
    comp_df = pd.DataFrame(results).set_index('model')
    comp_csv = os.path.join(out_root, 'comparison_metrics.csv')
    comp_df.to_csv(comp_csv)
    print("Saved:", comp_csv)

    # Comparison charts
    for metric in ['f1','roc_auc','pr_auc','science_recovery_rate','brier']:
        if metric in comp_df.columns:
            plt.figure(figsize=(7,4))
            comp_df[metric].plot(kind='bar')
            ylabel = 'Recovery Rate' if metric=='science_recovery_rate' else metric.upper()
            plt.ylabel(ylabel); plt.title(f'Model Comparison — {ylabel}')
            plt.tight_layout(); plt.savefig(os.path.join(out_root, f'compare_{metric}.png'), dpi=160); plt.close()

    # Merge predictions (optional)
    try:
        merged = pd.concat([pd.read_csv(p) for p in pred_paths], ignore_index=True)
        merged.to_csv(os.path.join(out_root, 'all_predictions_merged.csv'), index=False)
    except Exception:
        pass

    # Best‑model analysis (by F1)
    best_name = comp_df['f1'].idxmax()
    best = eval_packs[best_name]
    best_dir = os.path.join(out_root, f'best_model_analysis_{best_name}'); os.makedirs(best_dir, exist_ok=True)

    # Reload best model for saliency from checkpoint
    try:
        model_best = tf.keras.models.load_model(best['ckpt_path'])
    except Exception:
        # fallback: rebuild same arch if load fails (rare if custom layers are used differently)
        arch = {'V1_Plain': build_v1_plain, 'V2_Residual': build_v2_residual,
                'V3_SE': build_v3_se, 'V4_CBAM': build_v4_cbam}[best_name]
        model_best = arch(T, C); compile_model(model_best, args.lr)

    error_analysis_plots(best['X_test'], best['y_test'], best['y_pred'], best_dir, max_examples=20)
    # select a mix TP/FP/FN for saliency
    y_test, y_pred = best['y_test'], best['y_pred']
    idx_tp = np.where((y_pred==1) & (y_test==1))[0]
    idx_fp = np.where((y_pred==1) & (y_test==0))[0]
    idx_fn = np.where((y_pred==0) & (y_test==1))[0]
    idxs = list(idx_tp[:4]) + list(idx_fp[:3]) + list(idx_fn[:3])
    saliency_1d(model_best, best['X_test'], idxs, best_dir, max_examples=10)

    # Save run summary
    md_lines = [
        "# Task 1 Multi‑Model Runner — Checklist + Science",
        f"- Data: `{args.data}`",
        f"- KOI: `{args.koi}`",
        f"- Models run: {args.models}",
        f"- Output: `{out_root}`",
        "\n## Metrics (comparison)", comp_df.to_markdown(),
        f"\n**Best model (by F1):** {best_name}",
    ]
    with open(os.path.join(out_root, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    print("\nAll done. See:", out_root)


if __name__ == '__main__':
    main()
