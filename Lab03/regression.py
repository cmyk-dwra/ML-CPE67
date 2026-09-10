import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

from data_loader import load_data, prepare_features


# ---------- Load Data ----------
df = load_data()

# ---------- Prepare Data ----------
X = prepare_features(df)
Y = df["Milk_Yield"]

# ---------- Train / Test Split ----------
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

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

# ---------- Ridge Regression ----------
model = Ridge(alpha=1.0)
model.fit(X_train_pca, Y_train)
Y_pred = model.predict(X_test_pca)

# ---------- Evaluation ----------
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)
print("\n" + "=" * 60)
print("             MILK YIELD REGRESSION")
print("=" * 60)
print(f"\nMAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.2f}")


# ---------- Individual Predictions ----------
print("\n" + "=" * 60)
print("             PREDICTION RESULTS")
print("=" * 60)

for actual, predicted in zip(Y_test, Y_pred):
    error = actual - predicted
    print(f"Actual: {actual:.2f} | "f"Predicted: {predicted:.2f} | "f"Error: {error:.2f}")


# ---------- Actual vs Predicted Visualization ----------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=Y_test,y=Y_pred,s=80)
plt.plot([Y_test.min(), Y_test.max()],[Y_test.min(), Y_test.max()],linestyle="--")
plt.title("Milk Yield - Actual vs Predicted")
plt.xlabel("Actual Milk Yield")
plt.ylabel("Predicted Milk Yield")
plt.tight_layout()
plt.show()


# ---------- Residual Visualization ----------
errors = Y_test - Y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=Y_pred,y=errors,s=80)
plt.axhline(y=0,linestyle="--")
plt.title("Milk Yield - Residual Plot")
plt.xlabel("Predicted Milk Yield")
plt.ylabel("Residual Error")
plt.tight_layout()
plt.show()

# ---------- PCA Visualization ----------

pca_df = pd.DataFrame(X_test_pca,columns=["PC1", "PC2"])
pca_df["Milk_Yield"] = Y_test.values
plt.figure(figsize=(9, 6))
sns.scatterplot(data=pca_df,x="PC1",y="PC2",hue="Milk_Yield",palette="viridis",s=80)
plt.title("PCA Feature Space - Milk Yield Regression")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

# ---------- Prediction Error Distribution ----------
plt.figure(figsize=(8, 6))
sns.histplot(errors,bins=15,kde=True)
plt.axvline(x=0,linestyle="--")
plt.title("Distribution of Milk Yield Prediction Errors")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()