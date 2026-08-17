# MediTwin AI — Review 3 Results
## 1. Tabular Risk Model — UCI Heart Disease (Cleveland)
- **Dataset**: UCI Heart Disease (Cleveland), 303 samples, 13 features
- **Target**: Binary cardiac risk (0 = no disease, 1 = disease)
- **Model**: RandomForest + GradientBoosting soft-voting ensemble
- **Accuracy** : 0.9016
- **Precision**: 0.8438
- **Recall**   : 0.9643
- **F1 Score** : 0.9
- **ROC-AUC**  : 0.9535
- **5-Fold CV ROC-AUC**: 0.8706 ± 0.0313

```
              precision    recall  f1-score   support

  No Disease       0.97      0.85      0.90        33
     Disease       0.84      0.96      0.90        28

    accuracy                           0.90        61
   macro avg       0.90      0.91      0.90        61
weighted avg       0.91      0.90      0.90        61

```

## 2. Imaging Model — Chest X-Ray
> ⚠ Kaggle dataset not yet downloaded. To train:
> 1. Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
> 2. Extract to: `data/chest_xray/` (folders: train/val/test with NORMAL/PNEUMONIA subfolders)
> 3. Re-run: `python scripts/train_models.py --imaging`
