### Cow Mastitis Detection

a machine learning project for early detection of mastitis(a disease) in dairy cows using sensor-based health data.

- 800 rows, 9 features (Milk Temperature, Somatic Cell Count, Milk Conductivity, etc.)
- binary classification: Healthy (0) vs Mastitis (1)
- source: Kaggle

Three models were trained and compared:

| Model | F1 Score |
|-------|----------|
| Random Forest | 1.00 |
| Logistic Regression | 1.00 |
| XGBoost | 1.00 |

#> Note: Perfect scores are expected as the dataset is synthetically generated.

## Key Findings

we found a data leakage 'clotting' and removed it from the dataset.
we looked at the feature importance and found that 'Somatic_Cell_Count' and 'Milk_Temperature' are the most important features but 'day' was almost not important.
all scores were perfecto also
means that the dataset is synthethic.

## Tools

Python, Pandas, Scikit-learn, XGBoost, Seaborn, Matplotlib
