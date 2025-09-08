# Clinical/ Integrated Data Model
# Saved prediction outputs for both integrated with imaging and non integrated models.
# This script trains and tests the XGBoost model while exporting its predictions as csv outputs for later data processing.

# Libraries Used:
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
import datetime
import os

warnings.filterwarnings("ignore")

# Filepaths for input and outputs
# Output filename extension for saved outputs
datapath_Boostdata = "/home/rlang/Documents/SantaClaraUniversity/AI_Covid_19/Summer/data/boostdata2_id.csv"
datapath_Imagingdata = "/home/rlang/Documents/SantaClaraUniversity/AI_Covid_19/Summer/data/all_fold_predictions.csv"
save_folder = "/home/rlang/Documents/SantaClaraUniversity/AI_Covid_19/Summer/data_xgboostresults"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_prefix = f"{timestamp}_XGBoostOutput"

# Load dataset csvs
data_boost = pd.read_csv(datapath_Boostdata)
pred_data = pd.read_csv(datapath_Imagingdata)

# XGBoost Model with Integrated Imaging Data
# First importing predicted LOS from imaging model
if not {"subjectID", "predicted_los"}.issubset(pred_data.columns):
    raise ValueError("Missing 'subjectID' or 'predicted_los' in predictions CSV")

# Merge and drop dublicate columns
# Match Patient id to Subject id
data_with_imaging = data_boost.merge(
    pred_data[["subjectID", "predicted_los"]],
    left_on="to_patient_id",
    right_on="subjectID",
    how="inner"
).drop(['subjectID', 'to_patient_id'], axis=1)

# Setting target and feature labels
# Length of stay is the target
X1, y1 = data_with_imaging.drop("length_of_stay", axis=1), data_with_imaging["length_of_stay"]
text_cols1 = X1.select_dtypes(exclude=np.number).columns.tolist()
for col in text_cols1:
    X1[col] = X1[col].astype("category")

# Setting XGBoost model parameters
params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "seed": 42
}

# 15 Fold K fold Cross Validation Settings
kf = KFold(n_splits=15, shuffle=True, random_state=42)
all_predictions_imaging = []

print("Starting 15-Fold Cross Validation (with imaging)...\n")

# Begin model training on folds
for fold, (train_idx, test_idx) in enumerate(kf.split(X1), start=1):
    X_train, X_test = X1.iloc[train_idx], X1.iloc[test_idx]
    y_train, y_test = y1.iloc[train_idx], y1.iloc[test_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    model = xgb.train(params, dtrain=dtrain, num_boost_round=1000,
                      early_stopping_rounds=5, evals=[(dtest, "eval")], verbose_eval=False)

    y_pred = model.predict(dtest)

    fold_df = pd.DataFrame({
        'PredictedStay': y_pred,
        'ActualStay': y_test.values
    })
    all_predictions_imaging.append(fold_df)

    # Print fold completed in terminal output
    print(f"Fold {fold} done.")

# Output model predictions into a csv for the integrated model
output_csv1 = os.path.join(save_folder, f"{output_prefix}_withImaging.csv")
pd.concat(all_predictions_imaging, ignore_index=True).to_csv(output_csv1, index=False)
# Print save path
print(f"\nSaved Predicted vs Actual LOS (with imaging) to: {output_csv1}")

# XGBoost model without integrated imaging data begins here
# Removing pateint Id column
data_boost_clean = data_boost.drop(columns=["to_patient_id"])
# Setting target and featues
X2, y2 = data_boost_clean.drop("length_of_stay", axis=1), data_boost_clean["length_of_stay"]
# Setting non-numbered columns to categorical type for XGBoost
text_cols2 = X2.select_dtypes(exclude=np.number).columns.tolist()
for col in text_cols2:
    X2[col] = X2[col].astype("category")

# Setting up 15 fold cross validation for non-image model
kf2 = KFold(n_splits=15, shuffle=True, random_state=42)
all_predictions_boost = []

print("\nStarting 15-Fold Cross Validation (boost data only)...\n")

# Begin model training on folds
for fold, (train_idx, test_idx) in enumerate(kf2.split(X2), start=1):
    X_train, X_test = X2.iloc[train_idx], X2.iloc[test_idx]
    y_train, y_test = y2.iloc[train_idx], y2.iloc[test_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    model = xgb.train(params, dtrain=dtrain, num_boost_round=1000,
                      early_stopping_rounds=5, evals=[(dtest, "eval")], verbose_eval=False)

    y_pred = model.predict(dtest)

    fold_df = pd.DataFrame({
        'PredictedStay': y_pred,
        'ActualStay': y_test.values
    })
    all_predictions_boost.append(fold_df)

    print(f"Fold {fold} done.")

# Save the output from the non image model to designtated filepath and print location
output_csv2 = os.path.join(save_folder, f"{output_prefix}_boostOnly.csv")
pd.concat(all_predictions_boost, ignore_index=True).to_csv(output_csv2, index=False)
print(f"\nSaved Predicted vs Actual LOS (boost only) to: {output_csv2}")
