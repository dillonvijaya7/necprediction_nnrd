Hello, this project focuses on developing Machine learning models for predicting surgical necrotising enterocolitis (sNEC) 
risk in preterm infants (<32 weeks of gestation) using the UK National Neonatal 
Research Database (NNRD). Specifically, i have developed two models - a static (birth-time only) model to understand performance benchmarks and a dynamic 
(static + first 3 days of daily clinical observations) model to underdstand the additive benefit of longitudinal modelling.

## Data

This project uses the UK NNRD, a national audit of clinical data from all NHS 
neonatal units in England and Wales. Access to the NNRD is managed by the 
Neonatal Data Analysis Unit (NDAU) at Imperial College London. Data cannot be 
shared publicly. To apply for access, visit:
https://www.imperial.ac.uk/neonatal-data-analysis-unit

## Requirements
```
pip install numpy pandas pyarrow scikit-learn xgboost
```

## Repository Structure

| File | Description |
|------|-------------|
| `run_daily_surgical_nec_v4.py` | Main pipeline — run this to reproduce results |
| `train_and_evaluation.py` | Model training, evaluation metrics, and results saving |
| `models.py` | Model definitions (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, MLP) |
| `global_variables.py` | Data paths and NNRD column mappings — update paths before running |
| `baby_nec_preprocessing.py` | Surgical NEC outcome label construction |
| `loading_data.py` | Parquet file loading utilities |
| `static_modelling.py` | Earlier static-only modelling pipeline |

## Configuration

Before running, update the data paths in `global_variables.py` to point to 
your local NNRD parquet files:
```python
PATH_EPISODES  = '/path/to/episodes_anonymized.parquet'
PATH_DAILY_DATA = '/path/to/daily_data_anonymized.parquet'
PATH_ANON_BABY_NEC = '/path/to/baby_nec_v4_anonymized.parquet'
```

## How to Run
```bash
python run_daily_surgical_nec_v4.py
```
 
Results are cached to disk after the first run and subsequent runs load from cache 
and skip straight to model training.

Results are saved to the directory specified by `RESULTS_DIR` in 
`run_daily_surgical_nec_v4.py`, which are organised by variant and gestational age category.
