import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score
)

# ==============================
# НАСТРОЙКИ
# ==============================

DATA_DIR = "database/csv"
OUT_DIR = "plots"

os.makedirs(OUT_DIR, exist_ok=True)

sns.set(style="whitegrid")

# ==============================
# ЗАГРУЗКА ДАННЫХ И МОДЕЛИ
# ==============================

print("Загрузка данных и модели...")

feature_importance = pd.read_csv(f"{DATA_DIR}/feature_importance.csv")
game = pd.read_csv(f"{DATA_DIR}/latest_team_stats.csv")

with open(f"{DATA_DIR}/nba_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(f"{DATA_DIR}/feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

print("Готово")

# ==============================
# FEATURE IMPORTANCE
# ==============================

top = feature_importance.sort_values("importance", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top,
    x="importance",
    y="feature"
)
plt.title("Top-20 важных признаков модели")
plt.xlabel("Permutation Importance")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance_top20.png", dpi=150)
plt.close()

# ==============================
# ПОДГОТОВКА ДАННЫХ
# ==============================

y = game["home_win"].astype(int)
X = game[feature_cols]

proba = model.predict_proba(X)[:, 1]
pred = (proba >= 0.5).astype(int)

# ==============================
# ROC-КРИВАЯ
# ==============================

fpr, tpr, _ = roc_curve(y, proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая модели")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150)
plt.close()

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y, pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Away Win", "Home Win"],
    yticklabels=["Away Win", "Home Win"]
)
plt.xlabel("Предсказано")
plt.ylabel("Факт")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# ==============================
# РАСПРЕДЕЛЕНИЕ ВЕРОЯТНОСТЕЙ
# ==============================

plt.figure(figsize=(7, 4))
sns.histplot(proba, bins=30, kde=True)
plt.axvline(0.5, color="red", linestyle="--")
plt.title("Распределение вероятностей победы дома")
plt.xlabel("P(Home Win)")
plt.ylabel("Количество матчей")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/probability_distribution.png", dpi=150)
plt.close()

# ==============================
# BASELINE vs MODEL
# ==============================

baseline_acc = y.mean()
model_acc = accuracy_score(y, pred)

plt.figure(figsize=(5, 4))
sns.barplot(
    x=["Baseline (Always Home)", "Model"],
    y=[baseline_acc, model_acc]
)
plt.ylim(0, 1)
plt.title("Baseline vs Model Accuracy")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/baseline_vs_model.png", dpi=150)
plt.close()

print("Все графики сохранены в:", OUT_DIR)
