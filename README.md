# Predicting COVID-19 Severity Using a Multimodal AI Model Validated Across Multiple Institutions

This repository contains code for predicting hospital **Length of Stay (LOS)** in COVID-19 patients using two complementary approaches:  

1. **Imaging Model (Chest X-ray (CXR))**  
   A deep learning–based regression model trained on chest radiograph (CXR) images.  
2. **Clinical/Integrated Model (XGBoost)**  
   A gradient-boosted decision tree model trained on structured clinical data, with an option to integrate predicted LOS from the imaging branch.  

The pipeline enables comparison of **imaging-only**, **clinical-only**, and **integrated imaging + clinical** models to evaluate predictive power in hospital outcome forecasting.

---

## Imaging Model (Chest X-ray (CXR))

This branch contains code for training and evaluating a Chest X-ray (CXR) regression model to predict hospital length of stay (LOS) from chest radiograph images of COVID-19 patients at Stony Brook University Hospital. The model integrates medical image preprocessing and stratified cross-validation,to evaluate the predictive power of deep learning in clinical outcome prediction.

**Dependencies:**  
- `Python 3.8+`  
- `torch`, `timm`, `torchxrayvision`  
- `scikit-learn`, `lifelines`, `opencv-python`, `matplotlib`, `scikit-image`, `nibabel`

---

## Clinical & Integrated Model (XGBoost)

This branch contains code for training and evaluating an XGBoost-based regression model to predict hospital length of stay (LOS) from structured clinical data of COVID-19 patients. The model supports two configurations, clinical only or integrated clinical and imaging. The pipeline performs 15-fold cross-validation, evaluates predictive accuracy using RMSE and Concordance Index (C-index), and exports model predictions for downstream analysis. In addition, survival analysis and hazard ratio estimation are implemented, including Kaplan–Meier–style curves with bootstrapped confidence intervals and Weibull parametric hazard modeling, to assess risk stratification based on predicted LOS.

**Dependencies:**  
- `pandas`, `numpy`, `xgboost`  
- `scikit-learn`, `scipy`  
- `matplotlib`  
- `lifelines`  
