# This code is used for the static (birth-time) modelling to construct binarfy target variables for anyNEC, surgicalNEC amd medicalNEC 
from global_variables import NEC_categories, Surgical_NEC, Medical_NEC, PATH_ANON_BABY_NEC, MAPPING_BABY_NEC
from loading_data import load_parquet, reverse_mapping

def get_nec_target_variables(df):

    # AnyNEC
    df['AnyNEC'] = 0
    for col in NEC_categories:
        if col in df.columns:
            df.loc[df[col] == 1, 'AnyNEC'] = 1

    # SurgicalNEC
    df['SurgicalNEC'] = 0
    for col in Surgical_NEC:
        if col in df.columns:
            df.loc[df[col] == 1, 'SurgicalNEC'] = 1
    
    # MedicalNEC
    df['MedicalNEC'] = 0
    for col in Medical_NEC:
        if col in df.columns:
            df.loc[df[col] == 1, 'MedicalNEC'] = 1
    

    return df

def preprocessing_baby_nec_dataset():

    mapping_inverted = reverse_mapping(MAPPING_BABY_NEC)
    df_baby_nec = load_parquet(PATH_ANON_BABY_NEC).rename(columns=mapping_inverted)[['NationalIDBabyAnon'] + NEC_categories]