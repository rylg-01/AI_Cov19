## Swin Transformer LOS Evaluation on COVID-19 and Healthy Patients

This repository contains scripts for independently evaluating pretrained Swin Transformer models for predicting hospital length of stay (LOS) from chest radiographs.  
Evaluations include:
- COVID patients split by severity (level 1 vs level 2)
- Healthy vs COVID patients

Metrics computed include precision, recall, F1-score, balanced accuracy, ROC AUC, confusion matrices, and threshold optimization.  
Scripts also generate plots for LOS distributions, accuracy vs threshold, and ROC curves.

# Requirements
Python 3.8+  
PyTorch  
timm  
OpenCV  
scikit-image  
pandas  
numpy  
matplotlib  
seaborn  
scikit-learn  

