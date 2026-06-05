"""
Synthetic NNRD Data Generator
Generates synthetic data mimicking the NNRD structure for testing purposes.
Matches column names, data types, prevalence and distributions of real data.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
N_BABIES        = 10000
SNEC_PREVALENCE = 0.0319
MAX_GEST        = 32
OUTPUT_DIR      = Path.home() / 'synthetic_data'
SEED            = 42
rng             = np.random.default_rng(SEED)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Column mappings (hashed) ───────────────────────────────────────────────────
BABY_ID_COL     = '[13832, 3121, 2340, 9949]'
NATIONAL_ID_COL = '[1305, 9949, 2064, 19127, 1592, 9158]'
ENTITY_ID_COL   = '[5096, 8419, 1403, 1931, 1394]'
EPNUM_COL       = '[2064, 9949, 1592, 9158, 2064, 19127]'
GEST_COL        = '[144, 16144, 2116, 2924, 20059, 1116]'
HOSP_COL        = '[5096, 22650, 1197, 16769, 1592, 21986, 13040]'
BW_COL          = '[1760, 1320, 168, 10999]'
ADMIT_COL       = '[2064, 9949, 1592, 9158, 2064, 19127, 1592]'
SEX_COL         = '[1592, 9158, 2064, 19127]'
DAILY_ID_COL    = '[5096, 8419, 1403, 1931, 1394]'
DAILY_DATE_COL  = '[2295, 1708, 2149, 8057, 5822, 2861, 2349, 2109, 1179]'
DAILY_WT_COL    = '[141, 2924, 2924, 168, 1993, 2924, 3048, 2346]'
DAILY_MILK_COL  = '[2295, 3048, 12393, 1658, 3161, 1665]'
DAILY_BILI_COL  = '[2295, 2924, 1766, 4419, 2924, 6851, 10245]'

# NEC outcome columns
EARLIEST_DOL    = '[5041, 168, 17078, 20851, 22680, 1658, 168, 141, 13901]'
SURG_SURV_COL   = '[17078, 20851, 168, 26546, 1658]'
SURG_DIED_NEC   = '[17078, 20851, 26546, 1658, 168]'
SURG_DIED_OTHER = '[17078, 20851, 168, 1658, 26546]'
MED_NEC_DOL     = '[5041, 168, 3875, 22680, 1658, 168, 141, 13901]'

print("Generating synthetic NNRD data...")
print(f"N babies: {N_BABIES}, sNEC prevalence: {SNEC_PREVALENCE:.1%}")

# ── Generate babies ────────────────────────────────────────────────────────────
baby_ids      = np.arange(1, N_BABIES + 1)
national_ids  = np.arange(100001, 100001 + N_BABIES)
entity_ids    = np.arange(200001, 200001 + N_BABIES)
hospitals     = rng.choice([f'HOSP{i:03d}' for i in range(1, 182)], N_BABIES)
gest_ages     = rng.integers(23, MAX_GEST, N_BABIES)
birthweights  = np.clip(
    rng.normal(900, 300, N_BABIES) * (gest_ages / 28),
    300, 2500
).astype(int)
sexes         = rng.choice([1, 2], N_BABIES)
admit_times   = rng.integers(0, 1440, N_BABIES)  # minutes from midnight

# ── sNEC assignment ────────────────────────────────────────────────────────────
# Higher prevalence for EP, lower for VP — matching real data
snec_prob = np.where(gest_ages < 28, 0.0719, 0.0131)
is_snec   = rng.binomial(1, snec_prob).astype(bool)
print(f"sNEC babies: {is_snec.sum()} ({is_snec.mean():.1%})")

# sNEC onset day — median ~15 days, right skewed
snec_onset = np.where(
    is_snec,
    np.clip(rng.negative_binomial(3, 0.15, N_BABIES), 3, 120),
    np.nan
)

# NICU stay length
stay_length = np.where(
    is_snec,
    snec_onset + rng.integers(1, 30, N_BABIES),
    rng.integers(7, 90, N_BABIES)
).astype(float)
stay_length = np.where(np.isnan(stay_length), rng.integers(7, 90, N_BABIES), stay_length)

# ── Episodes table ─────────────────────────────────────────────────────────────
print("Generating episodes table...")

ep_df = pd.DataFrame({
    BABY_ID_COL:     baby_ids,
    NATIONAL_ID_COL: national_ids,
    ENTITY_ID_COL:   entity_ids,
    EPNUM_COL:       np.ones(N_BABIES, dtype=int),
    GEST_COL:        gest_ages,
    HOSP_COL:        hospitals,
    BW_COL:          birthweights,
    SEX_COL:         sexes,
    ADMIT_COL:       admit_times,
    # Additional static features
    '[1592, 9158, 2064]':   rng.integers(20, 45, N_BABIES),    # maternal age
    '[2064, 1592, 9158]':   rng.choice([0,1], N_BABIES, p=[0.7,0.3]),  # antenatal steroids
    '[9158, 2064, 1592]':   rng.choice([0,1], N_BABIES, p=[0.85,0.15]), # magnesium sulphate
    '[19127, 1592, 9158]':  rng.choice([0,1,2], N_BABIES),     # mode of delivery
    '[1592, 2064, 9158]':   rng.normal(36.5, 0.5, N_BABIES),   # admission temperature
    '[9158, 19127, 1592]':  rng.integers(100, 180, N_BABIES),  # heart rate
    '[2064, 19127, 9158]':  rng.integers(30, 80, N_BABIES),    # respiratory rate
    '[19127, 9158, 2064]':  rng.normal(95, 3, N_BABIES),       # oxygen saturation
    '[1592, 19127, 2064]':  rng.normal(4.5, 1.0, N_BABIES),    # blood glucose
    '[9158, 1592, 19127]':  rng.choice([0,1], N_BABIES, p=[0.6,0.4]),  # maternal diabetes
    '[2064, 9158, 19127]':  rng.choice([0,1], N_BABIES, p=[0.7,0.3]),  # maternal hypertension
    '[19127, 2064, 1592]':  rng.integers(1, 10, N_BABIES),     # IMD decile
    '[1592, 9158, 19127]':  rng.choice([0,1,2,3,4], N_BABIES), # apgar 1min
    '[9158, 19127, 1592]':  rng.choice([0,1,2,3,4,5,6,7,8,9,10], N_BABIES), # apgar 5min
    '[19127, 1592, 2064]':  rng.choice([0,1], N_BABIES, p=[0.8,0.2]),  # intubation at birth
})

pq.write_table(
    pa.Table.from_pandas(ep_df),
    OUTPUT_DIR / 'episodes_anonymized.parquet'
)
print(f"Episodes saved: {ep_df.shape}")

# ── Daily data table ───────────────────────────────────────────────────────────
print("Generating daily data table...")

daily_rows = []
for i, baby_id in enumerate(baby_ids):
    n_days = int(stay_length[i])
    entity_id = entity_ids[i]
    bw = birthweights[i]
    
    for day in range(1, n_days + 1):
        # Weight trajectory
        weight_change = rng.normal(0.5, 5) if day > 3 else rng.normal(-10, 5)
        daily_wt = max(200, bw + weight_change * day)
        
        daily_rows.append({
            DAILY_ID_COL:   entity_id,
            DAILY_DATE_COL: admit_times[i] + day * 1440,
            DAILY_WT_COL:   daily_wt,
            DAILY_MILK_COL: max(0, rng.normal(100, 30)),
            DAILY_BILI_COL: max(0, rng.normal(150, 50)),
            # Binary features
            '[2868, 1708, 23872, 1183, 2137, 4164]': rng.choice([0,1], p=[0.9,0.1]),   # major surgery
            '[27453, 11048, 2346, 8745, 2007]':      rng.choice([0,1], p=[0.95,0.05]), # nitric oxide
            '[20394, 2556, 2137, 11098]':             rng.choice([0,1], p=[0.85,0.15]), # chest drain
            '[1130, 3329, 12736, 1279, 2349, 2109, 1179]': rng.choice([0,1], p=[0.7,0.3]), # ventilator
            '[5096, 8419, 1403, 1931, 1394]':         rng.choice([0,1], p=[0.6,0.4]),  # CPAP
            '[11336, 5822, 1348, 2924, 10733, 3554]': rng.choice([0,1], p=[0.5,0.5]),  # antibiotics
            '[1457, 7903, 2240, 1179, 1708, 2875, 1358]': rng.choice([0,1], p=[0.8,0.2]), # inotropes
            '[8896, 2036, 1775, 26236, 1162, 1942, 4047, 1116, 17149]': rng.choice([0,1], p=[0.85,0.15]), # surfactant
            '[1970, 1942, 4798]':                     rng.choice([1,2,3,4], p=[0.3,0.3,0.2,0.2]), # care level
        })

daily_df = pd.DataFrame(daily_rows)
pq.write_table(
    pa.Table.from_pandas(daily_df),
    OUTPUT_DIR / 'daily_data_anonymized.parquet'
)
print(f"Daily data saved: {daily_df.shape}")

# ── Baby NEC outcome table ─────────────────────────────────────────────────────
print("Generating baby NEC outcome table...")

nec_df = pd.DataFrame({
    NATIONAL_ID_COL: national_ids,
    EARLIEST_DOL:    snec_onset,
    MED_NEC_DOL:     np.where(
        rng.binomial(1, 0.02, N_BABIES).astype(bool),
        rng.integers(3, 60, N_BABIES).astype(float),
        np.nan
    ),
    SURG_SURV_COL:   np.where(is_snec & (rng.random(N_BABIES) > 0.27), 1, 0),
    SURG_DIED_NEC:   np.where(is_snec & (rng.random(N_BABIES) < 0.15), 1, 0),
    SURG_DIED_OTHER: np.where(is_snec & (rng.random(N_BABIES) < 0.12), 1, 0),
})

pq.write_table(
    pa.Table.from_pandas(nec_df),
    OUTPUT_DIR / 'baby_nec_v4_anonymized.parquet'
)
print(f"Baby NEC saved: {nec_df.shape}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("SYNTHETIC DATA SUMMARY")
print("="*50)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Episodes:   {ep_df.shape}")
print(f"Daily data: {daily_df.shape}")
print(f"Baby NEC:   {nec_df.shape}")
print(f"\nsNEC prevalence: {is_snec.mean():.1%}")
print(f"EP (<28w):  {(gest_ages < 28).sum()} babies, sNEC: {is_snec[gest_ages < 28].mean():.1%}")
print(f"VP (28-31w): {(gest_ages >= 28).sum()} babies, sNEC: {is_snec[gest_ages >= 28].mean():.1%}")
print(f"\nFiles saved to {OUTPUT_DIR}")
print("Done.")
