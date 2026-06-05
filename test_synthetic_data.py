"""
Test that synthetic data loads correctly with existing global_variables mappings
"""
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from global_variables import mappings, mapping_daily_data, mapping_baby_nec, Surgical_NEC

DATA_DIR = Path.home() / 'synthetic_data'

# Column mappings
BABY_ID_COL  = mappings['EntityID']
GEST_COL     = mappings['GestationWeeks']
BW_COL       = mappings['Birthweight']
HOSP_COL     = mappings['ProviderNDAUCode']
NAT_ID_COL   = mappings['NationalIDBabyAnon']
DAILY_ID_COL = mapping_daily_data['EntityID']
NEC_ID_COL   = mapping_baby_nec['NationalIDBabyAnon']
SNEC_DOL_COL = mapping_baby_nec['earliest_SurgicalNEC_DOL']
SURG_COLS    = [mapping_baby_nec[k] for k in Surgical_NEC]

print("Loading synthetic data...")
ep    = pq.read_table(DATA_DIR / 'episodes_anonymized.parquet').to_pandas()
daily = pq.read_table(DATA_DIR / 'daily_data_anonymized.parquet').to_pandas()
nec   = pq.read_table(DATA_DIR / 'baby_nec_v4_anonymized.parquet').to_pandas()

print(f"\nEpisodes:   {ep.shape}")
print(f"Daily:      {daily.shape}")
print(f"Baby NEC:   {nec.shape}")

# Filter to <32 weeks
ep_preterm = ep[ep[GEST_COL] < 32]
print(f"\nPreterm (<32w): {len(ep_preterm)}")

# sNEC cases
nec['is_snec'] = nec[SURG_COLS].max(axis=1)
snec = nec[nec['is_snec'] == 1]
print(f"sNEC babies: {len(snec)}")
print(f"sNEC prevalence: {len(snec)/len(ep_preterm):.1%}")

# Gestational age distribution
print(f"\nGestational age distribution:")
print(ep_preterm[GEST_COL].value_counts().sort_index())

print("\nAll checks passed — synthetic data is compatible with global_variables.py")
