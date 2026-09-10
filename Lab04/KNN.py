# ---------- Imports ----------
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report)
from data_loader import load_data, prepare_features

# ---------- Load Data ----------
df = load_data()

# ---------- Prepare Data ----------
X = prepare_features(df)
Y = df["class1"]

# ---------- Train / Test Split ----------
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ---------- Feature Scaling ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- KNN Model Evaluation ----------
k_values = range(1, 22, 2)
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, Y_train)
    Y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracy_scores.append(accuracy)
    print(f"K = {k:2d} | Accuracy: {accuracy * 100:.2f}%")

# ---------- Best K ----------
best_index = accuracy_scores.index(max(accuracy_scores))
best_k = list(k_values)[best_index]
best_accuracy = accuracy_scores[best_index]
print("\n" + "=" * 60)
print("                 KNN CLASSIFICATION")
print("=" * 60)
print(f"\nBest K: {best_k}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")

# ---------- Final KNN Model ----------
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, Y_train)
Y_pred = model.predict(X_test_scaled)

# ---------- Confusion Matrix ----------
cm = confusion_matrix(Y_test, Y_pred)
print("\nConfusion Matrix:")
print(cm)

# ---------- Classification Report ----------
print("\nClassification Report:")
print(classification_report(Y_test,Y_pred,target_names=["No Mastitis", "Mastitis"]))

# ---------- Prediction Results ----------
print("\n" + "=" * 60)
print("                PREDICTION RESULTS")
print("=" * 60)
for actual, predicted in zip(Y_test, Y_pred):
    actual_label = "Mastitis" if actual else "No Mastitis"
    predicted_label = "Mastitis" if predicted else "No Mastitis"
    result = "CORRECT" if actual == predicted else "WRONG"
    print(f"Actual: {actual_label:<12} | "f"Predicted: {predicted_label:<12} | "f"{result}")

# ---------- Accuracy vs K Visualization ----------
plt.figure(figsize=(8, 6))
sns.lineplot(x=list(k_values),y=accuracy_scores,marker="o")
plt.title("KNN Accuracy for Different K Values")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.xticks(list(k_values))
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# ---------- Confusion Matrix Visualization ----------
plt.figure(figsize=(7, 5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["No Mastitis", "Mastitis"],yticklabels=["No Mastitis", "Mastitis"])
plt.title(f"KNN Confusion Matrix (K = {best_k})")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# ---------- Prediction Class Distribution ----------
prediction_df = pd.DataFrame({"Actual": Y_test.values,"Predicted": Y_pred})
plt.figure(figsize=(8, 6))
sns.countplot(data=prediction_df,x="Predicted",hue="Predicted",palette={False: "steelblue", True: "crimson"},legend=False)
plt.title(f"KNN Predicted Class Distribution (K = {best_k})")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.xticks([0, 1],["No Mastitis", "Mastitis"])
plt.tight_layout()
plt.show()