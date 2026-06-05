"""
V9 Pipeline — Static + Daily sNEC Prediction
Rebuilt for desktop machine with synthetic/real data support.
Consistent with HPC v9 pipeline methodology.
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.insert(0, str(Path.home() / 'necprediction_nnrd'))
from global_variables import (
    mappings as MAPPING_EPISODES,
    mapping_daily_data as MAPPING_DAILY_DATA,
    mapping_baby_nec as MAPPING_BABY_NEC,
    features_to_include as FEATURES_TO_INCLUDE,
    daily_features_to_include as DAILY_FEATURES_TO_INCLUDE,
    CATEGORICAL_DAILY_FEATURES,
    Surgical_NEC,
)

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path.home() / 'synthetic_data'
RESULTS_DIR = Path.home() / 'results' / 'v9'
CACHE_DIR   = Path.home() / 'preprocessing_cache'
CACHE_FILE  = CACHE_DIR / 'v9_cache.pkl'

NEC_DAY_CUT  = 2       # exclude babies with sNEC onset <= this day
LOOKBACK     = 3       # days of daily data to use
OHE_MIN_FREQ = 5

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Column mappings ────────────────────────────────────────────────────────────
BABY_ID_COL  = MAPPING_EPISODES['EntityID']
NAT_ID_COL   = MAPPING_EPISODES['NationalIDBabyAnon']
GEST_COL     = MAPPING_EPISODES['GestationWeeks']
HOSP_COL     = MAPPING_EPISODES['ProviderNDAUCode']
BW_COL       = MAPPING_EPISODES['Birthweight']
ADMIT_COL    = MAPPING_EPISODES['AdmitTimeAnon']
EPNUM_COL    = MAPPING_EPISODES['EpisodeNumberBaby']

DAILY_ID_COL   = MAPPING_DAILY_DATA['EntityID']
DAILY_DATE_COL = MAPPING_DAILY_DATA['DayDateAnon']
DAILY_WT_COL   = MAPPING_DAILY_DATA['DayWorkingWeight']

NEC_ID_COL   = MAPPING_BABY_NEC['NationalIDBabyAnon']
SNEC_DOL_COL = MAPPING_BABY_NEC['earliest_SurgicalNEC_DOL']
SURG_COLS    = [MAPPING_BABY_NEC[k] for k in Surgical_NEC]

# Text columns to exclude (cause OHE explosion)
TEXT_COLS = [
    MAPPING_EPISODES.get('MethodsOfResuscitation', ''),
    MAPPING_EPISODES.get('DrugsInLabour', ''),
    MAPPING_EPISODES.get('ProblemsDuringPregnancy', ''),
    MAPPING_EPISODES.get('ProblemsMedicalMother', ''),
]

# Truly continuous daily features (get mean + last + slope)
CONTINUOUS_DAILY = [MAPPING_DAILY_DATA[f] for f in [
    'DayWorkingWeight', 'DayWeight', 'VolumeMilk', 'MaxBilirubin',
] if f in MAPPING_DAILY_DATA]

# Binary daily features (get max)
BINARY_DAILY = [MAPPING_DAILY_DATA[f] for f in [
    'MajorSurgeryDay', 'NitricOxide', 'ChestDrain', 'ReplogleTube',
    'DaySurfactantGiven', 'PulmonaryVasodilator', 'InotropesGiven',
    'Prostaglandin', 'RectalWashout', 'StomaInSitu',
    'FullExchangeTransfusion', 'PartialExchangeTransfusion',
    'CentralTone', 'Convulsions', 'NASTreatment',
    'SurgeryVPShunt', 'EEGCFAM', 'ROPSurgery', 'ROPScreen',
    'Phototherapy', 'GastroschisisSilo', 'ECMO',
] if f in MAPPING_DAILY_DATA]

MINS_DAY = 1440

def compute_slope(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2: return np.nan
    xm, ym = x[mask], y[mask]
    if np.all(xm == xm[0]): return np.nan
    slope, *_ = scipy_stats.linregress(xm, ym)
    return slope

def stratified_split(df, hosp_col, seed=42):
    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(seed)
    for h in df[hosp_col].dropna().unique():
        idx = df[df[hosp_col] == h].index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(0.7 * n)
        n_val   = int(0.1 * n)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train+n_val])
        test_idx.extend(idx[n_train+n_val:])
    return train_idx, val_idx, test_idx

if CACHE_FILE.exists():
    print("Loading cache...")
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    base_df       = cache['base_df']
    recent_wt_col = cache['recent_weight_col']
    print(f"Cache loaded: {base_df.shape}")
else:
    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading episodes...")
    ep = pq.read_table(DATA_DIR / 'episodes_anonymized.parquet').to_pandas()

    # Filter to first episode, <32 weeks
    if EPNUM_COL in ep.columns:
        ep = ep[ep[EPNUM_COL] == 1].copy()
    ep = ep[ep[GEST_COL] < 32].copy()
    ep = ep.drop(columns=[c for c in TEXT_COLS if c in ep.columns], errors='ignore')
    print(f"Eligible babies: {len(ep)}")

    # ── Load outcomes ──────────────────────────────────────────────────────────
    print("Loading outcomes...")
    nec = pq.read_table(DATA_DIR / 'baby_nec_v4_anonymized.parquet').to_pandas()
    nec['is_snec'] = nec[SURG_COLS].max(axis=1)
    snec_dol = nec[nec['is_snec'] == 1].set_index(NEC_ID_COL)[SNEC_DOL_COL].to_dict()
    print(f"sNEC babies: {len(snec_dol)}")

    # Map NationalID to EntityID
    nat_to_entity = ep.set_index(NAT_ID_COL)[BABY_ID_COL].to_dict()
    snec_dol_entity = {
        nat_to_entity[k]: v
        for k, v in snec_dol.items()
        if k in nat_to_entity
    }

    # Create outcome column
    ep['snec_dol'] = ep[BABY_ID_COL].map(snec_dol_entity)
    ep[SURG_COLS[0]] = (
        ep['snec_dol'].notna() &
        (ep['snec_dol'] > NEC_DAY_CUT)
    ).astype(int)

    # Exclude early onset
    ep = ep[~(ep['snec_dol'].notna() & (ep['snec_dol'] <= NEC_DAY_CUT))].copy()
    print(f"After exclusions: {len(ep)}, sNEC: {ep[SURG_COLS[0]].sum()}")

    # ── Load daily data ────────────────────────────────────────────────────────
    print("Loading daily data...")
    daily = pq.read_table(DATA_DIR / 'daily_data_anonymized.parquet').to_pandas()

    # Map entity ID
    daily[BABY_ID_COL] = daily[DAILY_ID_COL]
    daily = daily[daily[BABY_ID_COL].isin(ep[BABY_ID_COL])].copy()

    # Compute day of life
    admit_map = ep.set_index(BABY_ID_COL)[ADMIT_COL].to_dict()
    daily['admit_time'] = daily[BABY_ID_COL].map(admit_map)
    daily['dol'] = ((daily[DAILY_DATE_COL] - daily['admit_time']) / MINS_DAY).apply(
        lambda x: int(np.floor(x)) + 1 if pd.notna(x) else np.nan
    )
    daily = daily[(daily['dol'] >= 1) & (daily['dol'] <= LOOKBACK)].copy()
    print(f"Daily rows in window: {len(daily)}")

    # ── Aggregate daily features ───────────────────────────────────────────────
    print("Aggregating daily features...")
    daily_feat_cols = [c for c in daily.columns
                       if c in [MAPPING_DAILY_DATA[f] for f in DAILY_FEATURES_TO_INCLUDE
                                 if f in MAPPING_DAILY_DATA]
                       and c not in [DAILY_ID_COL, BABY_ID_COL, DAILY_DATE_COL, 'admit_time', 'dol']]

    cat_daily = [MAPPING_DAILY_DATA[f] for f in CATEGORICAL_DAILY_FEATURES
                 if f in MAPPING_DAILY_DATA and MAPPING_DAILY_DATA[f] in daily.columns]
    other_cont = [c for c in daily_feat_cols
                  if c not in BINARY_DAILY and c not in CONTINUOUS_DAILY and c not in cat_daily]

    agg_dfs = []

    # Binary: max
    bin_in = [c for c in BINARY_DAILY if c in daily.columns]
    if bin_in:
        a = daily.groupby(BABY_ID_COL)[bin_in].max()
        a.columns = [f'max__{c}' for c in bin_in]
        agg_dfs.append(a)

    # Continuous: mean + last + slope
    cont_in = [c for c in CONTINUOUS_DAILY if c in daily.columns]
    if cont_in:
        a_mean = daily.groupby(BABY_ID_COL)[cont_in].mean()
        a_mean.columns = [f'mean__{c}' for c in cont_in]
        agg_dfs.append(a_mean)

        a_last = daily.sort_values('dol').groupby(BABY_ID_COL)[cont_in].last()
        a_last.columns = [f'last__{c}' for c in cont_in]
        agg_dfs.append(a_last)

        def slope_fn(g):
            out = {}
            for col in cont_in:
                x = g['dol'].values.astype(float)
                y = g[col].values.astype(float)
                out[f'slope__{col}'] = compute_slope(x, y)
            return pd.Series(out)
        a_slope = daily.groupby(BABY_ID_COL).apply(slope_fn, include_groups=False)
        agg_dfs.append(a_slope)

    # Other continuous: mean
    oth_in = [c for c in other_cont if c in daily.columns]
    if oth_in:
        for col in oth_in:
            daily[col] = pd.to_numeric(daily[col], errors='coerce')
        a_oth = daily.groupby(BABY_ID_COL)[oth_in].mean()
        a_oth.columns = [f'mean__{c}' for c in oth_in]
        agg_dfs.append(a_oth)

    # Categorical: last
    cat_in = [c for c in cat_daily if c in daily.columns]
    if cat_in:
        a_cat = daily.sort_values('dol').groupby(BABY_ID_COL)[cat_in].last()
        a_cat.columns = [f'cat__{c}' for c in cat_in]
        agg_dfs.append(a_cat)

    if agg_dfs:
        agg_all = pd.concat(agg_dfs, axis=1).reset_index()
        agg_all = agg_all.rename(columns={'index': BABY_ID_COL})
        ep = ep.merge(agg_all, on=BABY_ID_COL, how='left')

    # Recent weight column
    recent_wt_col = f'last__{DAILY_WT_COL}' if f'last__{DAILY_WT_COL}' in ep.columns else None
    if recent_wt_col:
        ep['delta_weight'] = ep[recent_wt_col] - ep[BW_COL]

    base_df = ep.copy()
    del daily, ep; gc.collect()

    print(f"Final dataset: {base_df.shape}")
    print(f"sNEC prevalence: {base_df[SURG_COLS[0]].mean():.1%}")

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'base_df': base_df, 'recent_weight_col': recent_wt_col}, f)
    print("Cache saved.")

# ── Preprocessing ──────────────────────────────────────────────────────────────
print("\nPreprocessing...")
OUTCOME_COL = SURG_COLS[0]

drop_cols = [BABY_ID_COL, NAT_ID_COL, HOSP_COL, 'snec_dol', ADMIT_COL]
if EPNUM_COL in base_df.columns:
    drop_cols.append(EPNUM_COL)

train_idx, val_idx, test_idx = stratified_split(base_df, HOSP_COL)

base_df = base_df.reset_index(drop=True)
train_df = base_df.iloc[train_idx].reset_index(drop=True)
val_df   = base_df.iloc[val_idx].reset_index(drop=True)
test_df  = base_df.iloc[test_idx].reset_index(drop=True)

y_train = train_df[OUTCOME_COL].astype(int).values
y_val   = val_df[OUTCOME_COL].astype(int).values
y_test  = test_df[OUTCOME_COL].astype(int).values

feat_cols = [c for c in base_df.columns if c not in drop_cols + [OUTCOME_COL]]
tr = train_df[feat_cols].copy()
va = val_df[feat_cols].copy()
te = test_df[feat_cols].copy()
del train_df, val_df, test_df; gc.collect()

# OHE
cat_cols = tr.select_dtypes(exclude=[np.number]).columns.tolist()
if cat_cols:
    for col in cat_cols:
        for d in [tr, va, te]:
            d[col] = d[col].astype(str).replace(
                {'nan': 'missing', '<NA>': 'missing', 'None': 'missing'})
    enc = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist',
                        min_frequency=OHE_MIN_FREQ)
    enc.fit(tr[cat_cols])
    enc_names = enc.get_feature_names_out(cat_cols)
    for d in [tr, va, te]:
        enc_df = pd.DataFrame(enc.transform(d[cat_cols]),
                              columns=enc_names, index=d.index)
        d.drop(columns=cat_cols, inplace=True)
        for col in enc_df.columns:
            d[col] = enc_df[col].values

clean = lambda c: str(c).replace('[','').replace(']','').replace('<','').replace('>','')
tr.columns = [clean(c) for c in tr.columns]
va.columns = tr.columns
te.columns = tr.columns

# Missingness + imputation
num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
miss_cols = [col for col in num_cols if tr[col].isna().any()]
if miss_cols:
    miss_tr = pd.DataFrame({f'miss__{c}': tr[c].isna().astype(int) for c in miss_cols})
    miss_va = pd.DataFrame({f'miss__{c}': va[c].isna().astype(int) for c in miss_cols})
    miss_te = pd.DataFrame({f'miss__{c}': te[c].isna().astype(int) for c in miss_cols})
    tr = pd.concat([tr.reset_index(drop=True), miss_tr], axis=1)
    va = pd.concat([va.reset_index(drop=True), miss_va], axis=1)
    te = pd.concat([te.reset_index(drop=True), miss_te], axis=1)

# Drop all-NaN columns before imputation
all_nan_cols = [c for c in tr.columns if tr[c].isna().all()]
if all_nan_cols:
    print(f"Dropping {len(all_nan_cols)} all-NaN columns")
    tr = tr.drop(columns=all_nan_cols)
    va = va.drop(columns=all_nan_cols)
    te = te.drop(columns=all_nan_cols)

imp = SimpleImputer(strategy='median')
imp.fit(tr)
X_train = pd.DataFrame(imp.transform(tr), columns=tr.columns)
X_val   = pd.DataFrame(imp.transform(va), columns=va.columns)
X_test  = pd.DataFrame(imp.transform(te), columns=te.columns)
del tr, va, te; gc.collect()
print(f"Feature matrix: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

# ── Train and evaluate models ──────────────────────────────────────────────────
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

MODELS = {
    'Baseline': DummyClassifier(strategy='constant', constant=0),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_depth=3, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        random_state=42, n_jobs=-1, verbosity=0),
    'MLP': MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42),
}

results = {}
print("\nTraining models...")
for name, model in MODELS.items():
    print(f"  {name}...")
    try:
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        sw = np.array([cw[l] for l in y_train])
        try:
            model.fit(X_train, y_train, sample_weight=sw)
        except TypeError:
            model.fit(X_train, y_train)
    except Exception as e:
        print(f"    Error: {e}")
        continue

    for split, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
        try:
            proba = model.predict_proba(X)[:, 1]
            pred  = (proba >= 0.5).astype(int)
            auroc = roc_auc_score(y, proba)
            auprc = average_precision_score(y, proba)
            f1    = f1_score(y, pred, zero_division=0)
            tp    = ((pred==1) & (y==1)).sum()
            fp    = ((pred==1) & (y==0)).sum()
            fn    = ((pred==0) & (y==1)).sum()
            print(f"    {split}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, "
                  f"F1={f1:.4f}, TP={tp}, FP={fp}, FN={fn}")
            results[f'{name}_{split}'] = {
                'auroc': auroc, 'auprc': auprc, 'f1': f1,
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
            }
        except Exception as e:
            print(f"    Error evaluating {split}: {e}")

# Save results
import json
with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

results_df = pd.DataFrame([
    {'model': k.rsplit('_', 1)[0], 'split': k.rsplit('_', 1)[1], **v}
    for k, v in results.items()
])
results_df.to_csv(RESULTS_DIR / 'results.csv', index=False)

print("\nTest set results:")
print(results_df[results_df['split']=='test'][
    ['model','auroc','auprc','f1','tp','fp','fn']
].to_string(index=False))
print(f"\nResults saved to {RESULTS_DIR}")
print("Done.")
