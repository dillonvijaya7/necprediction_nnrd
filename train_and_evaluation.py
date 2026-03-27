from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight


def train_model(model, train_x, train_y):

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_y), y=train_y)
    sample_weights = np.array([class_weights[int(label)] for label in train_y])

    try:
        model.fit(train_x, train_y, sample_weight=sample_weights)
    except TypeError:
        model.fit(train_x, train_y)

    return model


def test_model(model, x, y):

    y_proba = model.predict_proba(x)[:, 1]
    y_predict = model.predict(x)

    auroc = roc_auc_score(y, y_proba)
    auprc = average_precision_score(y, y_proba)
    f1 = f1_score(y, y_predict)
    cm = confusion_matrix(y, y_predict)

    return auroc, auprc, f1, cm


def get_features_importance(name_model, model_trained, columns):

    if name_model in ['Random Forest', 'XGBoost']:
        importances = pd.Series(model_trained.feature_importances_, index=columns)

    elif name_model == 'Logistic Regression':
        importances = pd.Series(model_trained.coef_[0], index=columns)

    else:
        importances = None

    return importances


def evaluation_pipeline(name_model, model_trained, val_x, val_y, test_x, test_y):

    val_auroc, val_auprc, val_f1_score, cm_val = test_model(model_trained, val_x, val_y)
    test_auroc, test_auprc, test_f1_score, cm_test = test_model(model_trained, test_x, test_y)

    print(f"Val AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}, F1-Score: {val_f1_score:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}, F1-Score: {test_f1_score:.4f}")

    metrics = [val_auroc, val_auprc, val_f1_score, test_auroc, test_auprc, test_f1_score]

    confusion_matrices = {
        "val": cm_val,
        "test": cm_test,
    }

    feature_importances = get_features_importance(name_model, model_trained, val_x.columns)

    return metrics, confusion_matrices, feature_importances


def train_and_evaluation_pipeline_all_models(models, train_x, train_y, val_x, val_y, test_x, test_y):

    metrics_all_models = pd.DataFrame(
        index=list(models.keys()),
        columns=["val_auroc", "val_auprc", "val_f1_score", "test_auroc", "test_auprc", "test_f1_score"]
    )
    dict_coeff_importances_all_models = dict()
    dict_confusion_matrices_all_models = dict()

    for name, model in models.items():
        print(f"{name}")

        model_trained = train_model(model, train_x, train_y)
        metrics, confusion_matrices, feature_importances = evaluation_pipeline(
            name, model_trained, val_x, val_y, test_x, test_y
        )

        metrics_all_models.loc[name] = metrics
        dict_confusion_matrices_all_models[name] = confusion_matrices
        dict_coeff_importances_all_models[name] = feature_importances

    return {
        "metrics_all_models": metrics_all_models,
        "confusion_matrices_all_models": dict_confusion_matrices_all_models,
        "coeff_importances_all_models": dict_coeff_importances_all_models,
    }


def gestation_age_cat(row):
    # Categorising gestational age (looks for a column containing 'GestationWeeks'
    # to handle hashed/cleaned column names)
    
    gest_val = None
    for col in row.index:
        if 'gestationweeks' in str(col).lower().replace(' ', '').replace('_', ''):
            gest_val = row[col]
            break

    if gest_val is None or (isinstance(gest_val, float) and np.isnan(gest_val)):
        return "Unknown"
    elif gest_val >= 37:
        return "Full term"
    elif gest_val >= 32:
        return "Preterm"
    elif gest_val >= 28:
        return "Very preterm"
    else:
        return "Extremely preterm"


def return_dfs_after_filter_gestation(x_df, y_df, value_to_filter):

    gest_age_all_categories = x_df.apply(gestation_age_cat, axis=1).reset_index(drop=True)
    indices = gest_age_all_categories[gest_age_all_categories == value_to_filter].index

    return x_df.iloc[indices], y_df.iloc[indices]


def train_and_evaluation_pipeline_all_models_per_gestation_category(
    models, x_train, y_train, x_val, y_val, x_test, y_test
):

    all_results = dict()
    gestation_age_categories = ["All", "Full term", "Preterm", "Very preterm", "Extremely preterm"]

    for gestation_age_category in gestation_age_categories:
        all_results[gestation_age_category] = dict()
        all_results[gestation_age_category]['metrics_all_models'] = pd.DataFrame(
            index=list(models.keys()),
            columns=["val_auroc", "val_auprc", "val_f1_score", "test_auroc", "test_auprc", "test_f1_score"]
        )
        all_results[gestation_age_category]['confusion_matrices_all_models'] = dict()
        all_results[gestation_age_category]['coeff_importances_all_models'] = dict()

    for name, model in models.items():
        print(f"{name}")

        model_trained = train_model(model, x_train, y_train)

        for gestation_age_category in gestation_age_categories:

            if gestation_age_category == "All":
                metrics, confusion_matrices, feature_importances = evaluation_pipeline(
                    name, model_trained, x_val, y_val, x_test, y_test
                )
            else:
                x_v, y_v = return_dfs_after_filter_gestation(x_val, y_val, gestation_age_category)
                x_t, y_t = return_dfs_after_filter_gestation(x_test, y_test, gestation_age_category)

                if len(y_v) == 0 or len(y_t) == 0:
                    print(f"  No samples for {gestation_age_category} — skipping.")
                    continue

                if len(np.unique(y_v)) < 2 or len(np.unique(y_t)) < 2:
                    print(f"  Only one class in {gestation_age_category} — skipping.")
                    continue

                metrics, confusion_matrices, feature_importances = evaluation_pipeline(
                    name, model_trained, x_v, y_v, x_t, y_t
                )

            all_results[gestation_age_category]['metrics_all_models'].loc[name] = metrics
            all_results[gestation_age_category]['confusion_matrices_all_models'][name] = confusion_matrices
            all_results[gestation_age_category]['coeff_importances_all_models'][name] = feature_importances

    return all_results


def confusion_matrix_to_df(double_array):
    return pd.DataFrame(
        double_array,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )


def save_results(all_results, models, name_folder, gestation_category):

    folder_path = os.path.join(name_folder, gestation_category)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    all_results['metrics_all_models'].to_csv(folder_path + "/results_metrics.csv", index=True)

    for model in list(models.keys()):

        # Confusion matrices
        cm_folder = os.path.join(name_folder, gestation_category, "confusion_matrices", model)
        if not os.path.exists(cm_folder):
            os.makedirs(cm_folder)

        if model in all_results['confusion_matrices_all_models']:
            confusion_matrix_to_df(
                all_results['confusion_matrices_all_models'][model]['val']
            ).to_csv(cm_folder + "/cm_val.csv", index=True)
            confusion_matrix_to_df(
                all_results['confusion_matrices_all_models'][model]['test']
            ).to_csv(cm_folder + "/cm_test.csv", index=True)

        # Feature importances
        imp_folder = os.path.join(name_folder, gestation_category, "coeff_importances", model)
        if not os.path.exists(imp_folder):
            os.makedirs(imp_folder)

        if model in all_results['coeff_importances_all_models']:
            if all_results['coeff_importances_all_models'][model] is not None:
                all_results['coeff_importances_all_models'][model].to_csv(
                    imp_folder + "/coeff_importances.csv", index=True
                )