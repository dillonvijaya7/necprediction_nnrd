"""
Daily Surgical NEC Prediction - Full NNRD Tables v4

- Combining daily records across all episodes per baby
- Filtering babies to <32 weeks gesttaion
- Use first 3 days from first admission (concatenate chronologically)
- Static features from first episode only
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from global_variables import (
    PATH_EPISODES, PATH_DAILY_DATA, PATH_ANON_BABY_NEC,
    MAPPING_EPISODES, MAPPING_DAILY_DATA, MAPPING_BABY_NEC,
    FEATURES_TO_INCLUDE_EPISODES, FEATURES_TO_EXCLUDE_EPISODES,
    FEATURES_TO_INCLUDE_DAILY, FEATURES_TO_EXCLUDE_DAILY,
    CATEGORICAL_DAILY_FEATURES, Surgical_NEC
)
from models import MODELS
from train_and_evaluation import train_model, evaluation_pipeline, save_results

RESULTS_DIR = '/rds/general/user/dv423/home/Neonatal_project/results/daily_surgical_nec_v4'
CACHE_FILE  = '/rds/general/user/dv423/home/Neonatal_project/preprocessing_cache/daily_surgical_nec_v4_cache.pkl'
MAX_GEST   = 32       # gestational age cutoff (weeks)
NEC_DAY_CUT = 2       # excluding babies whose earliest sNEC occurred on day of life <=2 
FIRST_N    = 3        # number of days from first admission to use as input window
MINS_DAY   = 1440     # minutes per day - used to convert admission time to day boundaries
OHE_MIN    = 5        # minimum frequency for one-hot encoding categories

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

# Decoding hashed column names from global_variables mappings
# All NNRD columns are stored as hashed lists - these resolve to the actual column name strings
baby_col   = MAPPING_EPISODES['NationalIDBabyAnon']  
entity_col = MAPPING_EPISODES['EntityID']              # unique episode identifier (one per admission)
epnum_col  = MAPPING_EPISODES['EpisodeNumberBaby']     # episode number (1 = first admission, chronological)
admit_col  = MAPPING_EPISODES['AdmitTimeAnon']         # admission time in minutes from birth
gest_col   = MAPPING_EPISODES['GestationWeeks']        
hosp_col   = MAPPING_EPISODES['ProviderNDAUCode']      # admitting hospital code (used for stratified splitting)
bw_col     = MAPPING_EPISODES['Birthweight']           

daily_entity = MAPPING_DAILY_DATA['EntityID']          # links daily records back to an episode
daily_date   = MAPPING_DAILY_DATA['DayDateAnon']       # day timestamp in minutes from midnight on birth day
daily_wt     = MAPPING_DAILY_DATA['DayWorkingWeight']  # clinically validated daily weight measurement

surg_nec_col   = MAPPING_BABY_NEC['Surgical_NEC']              # binary outcome: 1 = surgical NEC
earliest_dol   = MAPPING_BABY_NEC['earliest_SurgicalNEC_DOL']  # day of life when sNEC first occurred

# Resolving feature lists from string names to hashed column name strings
include_feats   = [MAPPING_EPISODES[f] for f in FEATURES_TO_INCLUDE_EPISODES if f in MAPPING_EPISODES]
exclude_feats   = [MAPPING_EPISODES[f] for f in FEATURES_TO_EXCLUDE_EPISODES if f in MAPPING_EPISODES]
daily_include   = [MAPPING_DAILY_DATA[f] for f in FEATURES_TO_INCLUDE_DAILY if f in MAPPING_DAILY_DATA]
categorical_daily = [MAPPING_DAILY_DATA[f] for f in CATEGORICAL_DAILY_FEATURES if f in MAPPING_DAILY_DATA]

gest_cats = ["All", "Very preterm", "Extremely preterm"]

def stratified_split(df, hosp_col, train_frac=0.7, val_frac=0.1, seed=42):
    """
    Hospital-stratified train/val/test split.
    Each hospital contributes babies to all three splits proportionally,
    so the test set reflects the full diversity of NICUs rather than
    being dominated by larger centres.
    """
    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(seed)
    for h in df[hosp_col].dropna().unique():
        idx = df[df[hosp_col] == h].index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(train_frac * n)
        n_val   = int(val_frac * n)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train+n_val])
        test_idx.extend(idx[n_train+n_val:])
    return train_idx, val_idx, test_idx

def clean_cols(df):
    """
    Remove special characters from column names that cause issues with
    scikit-learn (brackets and angle brackets from hashed column name format).
    """
    df.columns = [str(c).replace('[','').replace(']','').replace('<','').replace('>','') for c in df.columns]
    return df

def gest_cat(vals):
    """Assign gestational age category labels for subgroup analysis."""
    cats = []
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            cats.append("Unknown")
        elif v >= 37:
            cats.append("Full term")
        elif v >= 32:
            cats.append("Preterm")
        elif v >= 28:
            cats.append("Very preterm")   # 28-31 weeks
        else:
            cats.append("Extremely preterm")  # <28 weeks
    return np.array(cats)

# Load from cache incase any errors after prerpocessing
# so can skip straight to model training
if os.path.exists(CACHE_FILE):
    print(f"Loading cache {CACHE_FILE}")
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    df_base = cache['base_df']
    avg_wt_col = cache['avg_dayweight_col']
    recent_wt_col = cache['recent_weight_col']
    print(f"Cache loaded. Shape {df_base.shape}")
else:
    # Loading episodes and filtering to <32 weeks gestation
    ep = pq.read_table(PATH_EPISODES, use_pandas_metadata=False).to_pandas()
    print(f"Episodes shape {ep.shape}")

    # Filtering using EpisodeNumberBaby==1 (first admission) to get each baby's
    # gestational age
    first_ep_gest = ep[ep[epnum_col]==1][[baby_col, gest_col]]
    elig_babies = set(first_ep_gest[first_ep_gest[gest_col]<MAX_GEST][baby_col].values)
    ep = ep[ep[baby_col].isin(elig_babies)]
    print(f"Filtered <32wks: {len(elig_babies)} babies, {len(ep)} episodes")

    # Extracting first episode rows for static features
    # AdmitTimeAnon from episode 1 is used as the reference time for the 3-day window
    first_ep = ep[ep[epnum_col]==1].dropna(subset=[baby_col, entity_col, admit_col])
    first_admit = first_ep.set_index(baby_col)[admit_col].to_dict()  # {baby_id: first_admit_time_mins}

    first_ep = first_ep.drop(columns=[c for c in exclude_feats if c in first_ep.columns], errors='ignore')
    keep_cols = list(set([baby_col, hosp_col, bw_col, gest_col, surg_nec_col] +
                     [c for c in include_feats if c in first_ep.columns]))

    # Building a mapping of baby - all entity IDs across all admissions
    # needed to pull daily records from every episode (not just the first)
    all_entities = set(ep[entity_col])
    baby_to_entities = ep.groupby(baby_col)[entity_col].apply(list).to_dict()
    del ep; gc.collect()

    # Loading outcome labels and merging onto first episode
    nec_raw = pq.read_table(PATH_ANON_BABY_NEC, use_pandas_metadata=False).to_pandas()
    # Surgical NEC positive (1) if baby survived surgery, died from NEC, or
    # died from another cause after surgery 
    surg_cols = [MAPPING_BABY_NEC[k] for k in ['SurgicalNEC_Survived','SurgicalNEC_DiedNEC','SurgicalNEC_DiedNotNEC']]
    nec = nec_raw[[baby_col]+surg_cols+[earliest_dol]]
    del nec_raw; gc.collect()

    nec[surg_nec_col] = 0
    for c in surg_cols:
        nec.loc[nec[c]==1, surg_nec_col] = 1
    nec = nec[[baby_col, surg_nec_col, earliest_dol]]

    first_ep = first_ep.merge(nec, on=baby_col, how='left')
    first_ep[surg_nec_col] = first_ep[surg_nec_col].fillna(0).astype(int)
    del nec; gc.collect()

    # Excluding babies where sNEC occurred within the first 2 days of life
    # outcome would fall inside the input window (temporal leakage)
    early_nec = (first_ep[surg_nec_col]==1) & (first_ep[earliest_dol]<=NEC_DAY_CUT)
    first_ep = first_ep[~early_nec].copy()
    first_ep = first_ep.drop(columns=[earliest_dol], errors='ignore')

    first_ep = first_ep[[c for c in keep_cols if c in first_ep.columns]].copy()
    pop_babies = set(first_ep[baby_col])
    # Collecting all eligible entity IDs
    pop_entities = {eid for b in pop_babies for eid in baby_to_entities.get(b, [])}

    print(f"Episode features: {first_ep.shape}, total entities: {len(pop_entities)}")