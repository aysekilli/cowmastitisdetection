import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier

df = pd.read_csv("cow_milk_mastitis_dataset.csv")

X = df.drop(columns=["Cow_ID", "class1", "Clotting"])
y = df["class1"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

# 1. Model karşılaştırması
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.3f}")

plt.bar(results.keys(), results.values())
plt.title("Model Karşılaştırması (F1 Score)")
plt.ylim(0, 1.1)
plt.savefig("model_comparison.png")
plt.clf()

# 2. Confusion Matrix (Random Forest)
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Sağlıklı", "Mastitis"],
            yticklabels=["Sağlıklı", "Mastitis"])
plt.title("Confusion Matrix (Random Forest)")
plt.ylabel("Gerçek")
plt.xlabel("Tahmin")
plt.savefig("confusion_matrix.png")
plt.clf()

# 3. ROC Curve
plt.figure()
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.clf()

print("Tüm grafikler kaydedildi!")