# Integrated Imaging + Clinical XGBoost model for LOS prediction
#performs 15-fold CV, evaluates using Concordance Index (C-index), RMSE, ROC/AUC, and threshold-based metrics 

# Model Imports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error  # Importing for RMSE calculation
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
# Non essential imports
import warnings
import datetime
from scipy import stats

# Ignore warnings
warnings.filterwarnings("ignore")

# Custom concordance index function to get around issues with loading module
def concordance_index(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    n = 0
    n_concordant = 0
    n_tied = 0

    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                n += 1
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    n_concordant += 1
                elif y_pred[i] == y_pred[j]:
                    n_tied += 1

    return (n_concordant + 0.5 * n_tied) / n if n > 0 else 0.0

# Load the boostdata2_id.csv dataset
data = pd.read_csv("boostdata2_id.csv")

# Load predicted LOS data with patient_id
pred_data = pd.read_csv("/Users/hunter/Desktop/vit_checkpoints/all_fold_predictions.csv")

# Ensure required columns exist
if not {"subjectID", "predicted_los"}.issubset(pred_data.columns):
    raise ValueError("Missing 'subjectID' or 'predicted_los' in predictions CSV")

# Merge on patient_id (inner join ensures only matching rows are kept)
data = data.merge(pred_data[["subjectID", "predicted_los"]], left_on="to_patient_id", right_on="subjectID", how="inner")

# Remove 'patient_id' and 'subjectID' columns before proceeding with the model
data = data.drop(['subjectID', 'to_patient_id'], axis=1)

# Show first few rows of the merged dataset to ensure the new column is added
print(data.head())

# Define features and target
X, y = data.drop('length_of_stay', axis=1), data['length_of_stay']

# Convert categorical columns
textdata = X.select_dtypes(exclude=np.number).columns.tolist()
for col in textdata:
    X[col] = X[col].astype('category')

# XGBoost parameters
params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Use "hist" if no GPU
    "seed": 42
}

# Cross-validation
kf = KFold(n_splits=15, shuffle=True, random_state=42)
cindex_scores = []
rmse_scores = []  # List to store RMSE scores

print("Starting 5-Fold Cross Validation with C-index Evaluation...\n")

for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        early_stopping_rounds=5,
        evals=[(dtest, "eval")],
        verbose_eval=True
    )

    y_pred = model.predict(dtest)
    cidx = concordance_index(y_test.values, y_pred)
    cindex_scores.append(cidx)
    
    # Calculate RMSE for the fold
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

    print(f"Fold {fold} C-index: {cidx:.4f}, RMSE: {rmse:.4f}")

# Report average C-index and average RMSE
avg_cindex = np.mean(cindex_scores)
avg_rmse = np.mean(rmse_scores)
# Compute 95% confidence interval for C-index
sem_cindex = stats.sem(cindex_scores)  # Standard Error of the Mean
confidence = 0.95
h = sem_cindex * stats.t.ppf((1 + confidence) / 2, len(cindex_scores) - 1)
ci_lower = avg_cindex - h
ci_upper = avg_cindex + h

print(f"\nAverage C-index from 15-fold CV: {avg_cindex:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")
print(f"Average RMSE from 15-fold CV: {avg_rmse:.4f}")


# End timestamp
current_time = datetime.datetime.now()
print("Current date and time:", current_time)


# Store all predictions and labels
all_y_true = []
all_y_pred = []

for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    ...
    # After prediction
    y_pred = model.predict(dtest)

    all_y_true.extend(y_test.values)
    all_y_pred.extend(y_pred)

    ...

# Binary classification threshold (e.g., LOS > 5 days)
binary_y_true = [1 if val > 5 else 0 for val in all_y_true]
binary_y_pred = all_y_pred  # still regression values, good for ROC

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(binary_y_true, binary_y_pred)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkred', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')

# Calculate the best threshold based on Youden's index
best_idx = np.argmax(tpr - fpr)
best_threshold = thresholds[best_idx]

# Hollow marker
plt.scatter(
    fpr[best_idx],
    tpr[best_idx],
    facecolors='none', edgecolors='black', s=80, linewidths=2
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for LOS > 5 days')
plt.legend(loc="lower right")
plt.grid(True)

ax = plt.gca()
ax.set_aspect('equal')
ymin, ymax = ax.get_ylim()
ax.set_xlim(ymin, ymax)

plt.tight_layout()
plt.show()


# Predict using the best threshold (convert probabilities to binary class labels)
binary_y_pred_best_threshold = [1 if val > best_threshold else 0 for val in all_y_pred]

# Your original metrics
sensitivity = recall_score(binary_y_true, binary_y_pred_best_threshold)
specificity = recall_score(binary_y_true, binary_y_pred_best_threshold, pos_label=0)
accuracy = accuracy_score(binary_y_true, binary_y_pred_best_threshold)
precision = precision_score(binary_y_true, binary_y_pred_best_threshold)
f1 = f1_score(binary_y_true, binary_y_pred_best_threshold)

# Confusion matrix
conf_matrix = confusion_matrix(binary_y_true, binary_y_pred_best_threshold)
tn, fp, fn, tp = conf_matrix.ravel()

# Display original metrics
print("\nMetrics at Best Threshold:")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# ---- Add bootstrapped CIs here ----
# Bootstrap function
def bootstrap_ci(y_true, y_pred_binary, metric_func, n_bootstraps=1000, ci=0.95):
    rng = np.random.default_rng(seed=42)
    boot_metrics = []
    y_true = np.array(y_true)
    y_pred_binary = np.array(y_pred_binary)
    n = len(y_true)
    
    for _ in range(n_bootstraps):
        idxs = rng.choice(n, size=n, replace=True)
        boot_metrics.append(metric_func(y_true[idxs], y_pred_binary[idxs]))
    
    lower = np.percentile(boot_metrics, (1-ci)/2*100)
    upper = np.percentile(boot_metrics, (1+ci)/2*100)
    return lower, upper

# Compute 95% CIs for metrics
sensitivity_ci = bootstrap_ci(binary_y_true, binary_y_pred_best_threshold, recall_score)
specificity_ci = bootstrap_ci(binary_y_true, binary_y_pred_best_threshold, lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=0))
accuracy_ci = bootstrap_ci(binary_y_true, binary_y_pred_best_threshold, accuracy_score)
precision_ci = bootstrap_ci(binary_y_true, binary_y_pred_best_threshold, precision_score)
f1_ci = bootstrap_ci(binary_y_true, binary_y_pred_best_threshold, f1_score)

# Display metrics with 95% CI
print("\nMetrics at Best Threshold (with 95% CI):")
print(f"Sensitivity (Recall): {sensitivity:.4f} (CI: {sensitivity_ci[0]:.4f} - {sensitivity_ci[1]:.4f})")
print(f"Specificity: {specificity:.4f} (CI: {specificity_ci[0]:.4f} - {specificity_ci[1]:.4f})")
print(f"Accuracy: {accuracy:.4f} (CI: {accuracy_ci[0]:.4f} - {accuracy_ci[1]:.4f})")
print(f"Precision: {precision:.4f} (CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
print(f"F1-Score: {f1:.4f} (CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")

