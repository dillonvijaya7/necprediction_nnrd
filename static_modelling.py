from global_variables import PATH_ANON_NNRD_STATIC_AI, MAPPING_STATIC_AI, FEATURES_TO_INCLUDE
from models import FixedProbaClassifier, MODELS
from loading_data import load_parquet, reverse_mapping
import pandas as pd
import numpy as np
from baby_nec_preprocessing import preprocessing_baby_nec_dataset
from common_preprocessing_steps import get_train_val_test_indexes_split, encoding_cat_var, imputing_mean_missing_values,return_x_all_features,get_y_sets
from train_and_evaluation import train_and_evaluation_pipeline_all_models_per_gestation_category,save_results



def preprocessing_static_dataset():


    mapping_inverted = reverse_mapping(MAPPING_STATIC_AI)
    df_nnrd_stat = load_parquet(PATH_ANON_NNRD_STATIC_AI).rename(columns=mapping_inverted)
    df_nnrd_stat_filtered = df_nnrd_stat[FEATURES_TO_INCLUDE]

    numerical_cols = df_nnrd_stat_filtered.select_dtypes(include=[np.number]).columns.tolist()

    missingness_vectors = df_nnrd_stat_filtered[numerical_cols].drop(columns='Anon_ID').isna().astype(int).add_suffix('_missing')

    return pd.concat([df_nnrd_stat_filtered, missingness_vectors], axis=1)




def main():

    df_static_missingness = preprocessing_static_dataset()
    df_baby_nec = preprocessing_baby_nec_dataset()
    merged_df = df_static_missingness.merge(df_baby_nec, on='NationalIDBabyAnon', how="left")
    df_final = merged_df.drop(columns=["Anon_ID","NationalIDBabyAnon"])

    train_idx, val_idx, test_idx = get_train_val_test_indexes_split(df_final,'ProviderNDAUCode')
    train_df = df_final.iloc[train_idx]
    val_df = df_final.iloc[val_idx]
    test_df = df_final.iloc[test_idx]

    train_df_encoded, val_df_encoded, test_df_encoded = encoding_cat_var(train_df,val_df,test_df)
    train_df_encoded_imputed_none,val_df_encoded_imputed_none,test_df_encoded_imputed_none = imputing_mean_missing_values(train_df_encoded,val_df_encoded,test_df_encoded)
    x_train_all_features, x_val_all_features, x_test_all_features = return_x_all_features(train_df_encoded_imputed_none,val_df_encoded_imputed_none,test_df_encoded_imputed_none)
    y_train, y_val, y_test = get_y_sets(train_df_encoded_imputed_none,val_df_encoded_imputed_none,test_df_encoded_imputed_none,"SurgicalNEC")


    MODELS['Baseline_NEC_frequency'] = FixedProbaClassifier(p1 = y_train[y_train == 1].shape[0] / y_train.shape[0])

    all_results_per_gestation = train_and_evaluation_pipeline_all_models_per_gestation_category(MODELS,x_train_all_features,y_train,x_val_all_features,y_val,x_test_all_features,y_test)

    save_results(all_results_per_gestation['All'],MODELS,"results/static","all")
    save_results(all_results_per_gestation['Full term'],MODELS,"results/static","Full term")
    save_results(all_results_per_gestation['Preterm'],MODELS,"results/static","Preterm")
    save_results(all_results_per_gestation['Very preterm'],MODELS,"results/static","Very preterm")
    save_results(all_results_per_gestation['Extremely preterm'],MODELS,"results/static","Extremely preterm")

    return all_results_per_gestation