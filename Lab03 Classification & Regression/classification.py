import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report)

from data_loader import load_data, prepare_features


# ---------- Load Data ----------
df = load_data()

# ---------- Prepare Data ----------
X = prepare_features(df)
Y = df["class1"]

# ---------- Train / Test Split ----------
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)


# ---------- Scaling ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- PCA ----------
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
print(f"Total Explained Variance: "f"{pca.explained_variance_ratio_.sum() * 100:.2f}%")

# ---------- Logistic Regression ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, Y_train)
Y_pred = model.predict(X_test_pca)

# ---------- Evaluation ----------
accuracy = accuracy_score(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
print("\n" + "=" * 60)
print("             MASTITIS CLASSIFICATION")
print("=" * 60)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(Y_test,Y_pred,target_names=["No Mastitis", "Mastitis"]))


# ---------- Individual Predictions ----------
print("\n" + "=" * 60)
print("             PREDICTION RESULTS")
print("=" * 60)
for actual, predicted in zip(Y_test, Y_pred):
    actual_label = ("Mastitis" if actual else "No Mastitis")
    predicted_label = ("Mastitis" if predicted else "No Mastitis")
    result = "CORRECT" if actual == predicted else "WRONG"

    print(f"Actual: {actual_label:<12} | "f"Predicted: {predicted_label:<12} | "f"{result}")

# ---------- Confusion Matrix Visualization ----------
plt.figure(figsize=(7, 5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["No Mastitis", "Mastitis"],yticklabels=["No Mastitis", "Mastitis"])
plt.title("Mastitis Classification - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()


# ---------- PCA Visualization ----------
pca_df = pd.DataFrame(X_test_pca,columns=["PC1", "PC2"])
pca_df["Class"] = Y_test.values
pca_df["Prediction"] = Y_pred


plt.figure(figsize=(9, 6))
sns.scatterplot(data=pca_df,x="PC1", y="PC2", hue="Class", style="Prediction", palette={False: "steelblue", True: "crimson"}, s=80)

plt.title("PCA Feature Space - Mastitis Classification")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()


# ---------- Class Distribution ----------
plt.figure(figsize=(7, 5))
sns.countplot(
    x=Y_pred,
    hue=Y_pred,
    palette={False: "steelblue", True: "crimson"}, legend=False)

plt.title("Predicted Mastitis Class Distribution")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.xticks([0, 1], ["No Mastitis", "Mastitis"])

plt.tight_layout()
plt.show()