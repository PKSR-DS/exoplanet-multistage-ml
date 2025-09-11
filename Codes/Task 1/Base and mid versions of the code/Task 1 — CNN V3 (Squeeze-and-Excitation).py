# ==============================================
# Task 1 — CNN V3 (Squeeze-and-Excitation)
# - Prepared by: Praveen Kumar Savariraj
# - I reviewed, modified, tested, and take responsibility for this code.
# - Cite datasets: NASA Exoplanet Archive KOI Cumulative Table; Kepler light curves (preprocessed .npy).
# ==============================================

DATA_DIR = r"D:\UH One drive\Kepler_Preprocessed"
KOI_CANDIDATE_PATHS = [
    r"D:\UH One drive\cumulative_2025.07.16_10.28.33.csv",
    r"D:\UH One drive\Kepler_Preprocessed\cumulative_2025.07.16_10.28.33.csv",
    "cumulative_2025.07.16_10.28.33.csv"
]

EPOCHS=30; BATCH_SIZE=256; LEARNING_RATE=1e-3; TEST_SIZE=0.2; SEED=42
SAMPLE_FRAC=0.05; MAX_WINDOWS_PER_FILE=1000

import os, re, glob, random, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
warnings.filterwarnings("ignore")

# --- utils (same as V1) ---
def set_seeds(seed=SEED): random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
def try_load_koi_df():
    for p in KOI_CANDIDATE_PATHS:
        try:
            if os.path.exists(p):
                try: df=pd.read_csv(p)
                except: df=pd.read_csv(p, sep="|", comment="#", engine="python", skip_blank_lines=True)
                df.columns=df.columns.str.strip().str.lower(); print(f"[INFO] Loaded KOI: {p}"); return df
        except Exception as e: print(f"[WARN] KOI read failed for {p}: {e}")
    print("[INFO] KOI table not found — science score limited."); return None
def confirmed_kepid_set(koi_df):
    if koi_df is None: return set()
    if "koi_disposition" in koi_df.columns:
        m=koi_df["koi_disposition"].astype(str).str.upper().eq("CONFIRMED")
        return set(pd.to_numeric(koi_df.loc[m,"kepid"], errors="coerce").dropna().astype(int))
    return set(pd.to_numeric(koi_df.get("kepid", pd.Series(dtype=int)), errors="coerce").dropna().astype(int))
_kepid_re=re.compile(r'kplr0*([0-9]+)', re.IGNORECASE)
def parse_kepid_from_name(n): 
    m=_kepid_re.search(n or ""); 
    return int(m.group(1)) if (m and m.group(1).isdigit()) else None
def find_pairs(d):
    x=sorted(glob.glob(os.path.join(d,"*_X.npy"))); pairs=[]
    for xp in x:
        base=xp[:-6]; y1,y2=base+"_y.npy", base+"_Y.npy"; yp=y1 if os.path.exists(y1) else (y2 if os.path.exists(y2) else None)
        if yp: pairs.append((xp,yp))
    if not pairs: raise FileNotFoundError(f"No *_X.npy in {d}")
    print(f"[INFO] Found {len(pairs)} star file pairs."); return pairs
def load_data(d, frac=SAMPLE_FRAC, cap=MAX_WINDOWS_PER_FILE):
    set_seeds(); Xs,ys,groups,files=[],[],[],[]
    for xp,yp in find_pairs(d):
        X=np.load(xp).astype("float32"); y=np.load(yp).astype("int8")
        if X.ndim==1: X=X.reshape(1,-1)
        if X.ndim==2: X=X[...,None]
        if (cap is not None) and (X.shape[0]>cap):
            idx=np.random.choice(np.arange(X.shape[0]), size=cap, replace=False); X,y=X[idx],y[idx]
        if frac<1.0 and X.shape[0]>1:
            n=max(1,int(X.shape[0]*frac)); idx=np.random.choice(np.arange(X.shape[0]), size=n, replace=False); X,y=X[idx],y[idx]
        base=os.path.basename(xp); kid=parse_kepid_from_name(base); grp=kid if kid is not None else base
        Xs.append(X); ys.append(y); groups.extend([grp]*len(y)); files.extend([base]*len(y))
    X=np.concatenate(Xs,axis=0); y=np.concatenate(ys,axis=0).astype(int); groups=np.array(groups); files=np.array(files)
    print(f"[INFO] Dataset windows: {X.shape[0]}, window_length: {X.shape[1]}"); return X,y,groups,files
def grouped_split(X,y,g,test_size=TEST_SIZE,seed=SEED):
    gss=GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed); return next(gss.split(X,y,groups=g))
def class_weights_from_labels(y):
    pos=float(y.mean()); 
    if pos<=0 or pos>=1: return None
    return {0:1.0, 1:float(max(1.0,(1-pos)/pos))}
def pick_threshold(y,p):
    best_t,best_f1=0.5,-1.0
    for t in np.linspace(0.2,0.8,25):
        pr=(p>=t).astype(int); _,_,f1,_=precision_recall_fscore_support(y,pr,average="binary",zero_division=0)
        if f1>best_f1: best_f1,best_t=f1,t
    return best_t,best_f1
def science_score(k,y,p,cset):
    if not cset: return set(),set(),set()
    df=pd.DataFrame({"kepid":k,"true":y,"pred":p}).dropna(); df["kepid"]=df["kepid"].astype(int); df=df[df["kepid"].isin(list(cset))]
    dfp=df[df["true"]==1]; 
    if dfp.empty: return set(),set(),set()
    g=dfp.groupby("kepid")["pred"].max(); rec=set(g[g==1].index); conf=set(g.index); miss=conf-rec; return rec,miss,conf
def aggregate_by_kepid(k,y,p,proba):
    df=pd.DataFrame({"kepid":k,"true":y,"pred":p,"proba":proba}).dropna(subset=["kepid"]); df["kepid"]=df["kepid"].astype(int)
    agg=df.groupby("kepid").agg(n_windows=("true","size"),
        tp_windows=("true",lambda s:int(((df.loc[s.index,"true"]==1)&(df.loc[s.index,"pred"]==1)).sum())),
        fn_windows=("true",lambda s:int(((df.loc[s.index,"true"]==1)&(df.loc[s.index,"pred"]==0)).sum())),
        fp_windows=("true",lambda s:int(((df.loc[s.index,"true"]==0)&(df.loc[s.index,"pred"]==1)).sum())),
        max_proba=("proba","max"), mean_proba=("proba","mean")).reset_index()
    agg["transit_detected"]=(agg["tp_windows"]>0).astype(int); return agg

# --- model (V3 SE) ---
def se_block(x, ratio=16):
    ch=int(x.shape[-1]); se=layers.GlobalAveragePooling1D()(x)
    se=layers.Dense(max(ch//ratio,1),activation="relu")(se)
    se=layers.Dense(ch,activation="sigmoid")(se)
    se=layers.Reshape((1,ch))(se)
    return layers.Multiply()([x,se])

def conv_se(x,filters,k=3,pool=True):
    x=layers.Conv1D(filters,k,padding="same")(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=se_block(x)
    if pool: x=layers.MaxPool1D(2)(x)
    return x

def build_model(window_size:int):
    inp=layers.Input(shape=(window_size,1))
    x=conv_se(inp,32,7)
    x=conv_se(x,64,5)
    x=conv_se(x,128,3,pool=False)
    x=layers.GlobalAveragePooling1D()(x); x=layers.Dense(128,activation="relu")(x); x=layers.Dropout(0.35)(x)
    out=layers.Dense(1,activation="sigmoid")(x)
    return models.Model(inp,out,name="CNN_V3_SE")

# --- train/eval/export ---
if __name__=="__main__":
    set_seeds(SEED)
    X,y,g,files=load_data(DATA_DIR,SAMPLE_FRAC,MAX_WINDOWS_PER_FILE)
    i_tr,i_te=grouped_split(X,y,g); Xtr,Xte,ytr,yte=X[i_tr],X[i_te],y[i_tr],y[i_te]
    ktest=np.array([parse_kepid_from_name(f) for f in files[i_te]])

    model=build_model(Xtr.shape[1])
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    cbs=[callbacks.EarlyStopping(patience=5,restore_best_weights=True,monitor="val_loss"),
         callbacks.ReduceLROnPlateau(patience=3,factor=0.5,verbose=1),
         callbacks.ModelCheckpoint("best_model_V3.h5",save_best_only=True,monitor="val_loss")]
    cw=class_weights_from_labels(ytr)
    model.fit(Xtr,ytr,validation_data=(Xte,yte),epochs=EPOCHS,batch_size=BATCH_SIZE,class_weight=cw,callbacks=cbs,verbose=2)

    proba=model.predict(Xte,batch_size=BATCH_SIZE).flatten()
    thr,bf1=pick_threshold(yte,proba); pred=(proba>=thr).astype(int)
    try: auc=roc_auc_score(yte,proba)
    except Exception: auc=np.nan
    print(f"Threshold: {thr:.3f} (F1={bf1:.3f})"); print(f"AUC: {auc:.4f}")
    print(pd.DataFrame(classification_report(yte,pred,output_dict=True,zero_division=0)).T)

    koi=try_load_koi_df(); conf=confirmed_kepid_set(koi)
    rec,miss,confset=science_score(ktest,yte,pred,conf)
    print("\n🔭 CNN V3 Science Score")
    print(f"→ Confirmed planets in test sample: {len(confset)}")
    print(f"→ Recovered: {len(rec)}"); print(f"→ Missed: {len(miss)}")
    if confset: print(f"🎯 Recovery Rate: {len(rec)/len(confset):.2%}")
    else: print("⚠️ No confirmed planets found in test set.")

    agg=aggregate_by_kepid(ktest,yte,pred,proba)
    kmap=dict(zip(koi.get("kepid",pd.Series(dtype=int)), koi.get("kepler_name",pd.Series(dtype=str)))) if koi is not None else {}
    agg["kepler_name"]=agg["kepid"].map(kmap) if kmap else np.nan
    agg["model_version"]="V3"
    agg.to_csv("task1_transit_predictions_V3.csv",index=False); print("[OK] Wrote task1_transit_predictions_V3.csv")
    allp="task1_transit_predictions.csv"
    if os.path.exists(allp):
        old=pd.read_csv(allp); all_df=pd.concat([old,agg],ignore_index=True).drop_duplicates(subset=["kepid","model_version"],keep="last")
        all_df.to_csv(allp,index=False)
    else: agg.to_csv(allp,index=False)
    print(f"[OK] Updated {allp}")


