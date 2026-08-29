import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import load_data, prepare_features


#---------- Load Data ----------
df = load_data()

# ---------- Prepare Data ----------
X = prepare_features(df)
Y = df["Milk_Yield"]

#---------- Train / Test Split ----------
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

#---------- Scaling ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- PCA ----------
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# ---------- Ridge Regression ----------
model = Ridge(alpha=1.0)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# ---------- Evaluation ----------
for actual, predicted in zip(Y_test, Y_pred):
    accuracy = 1 - abs(predicted - actual) / abs(actual)

    print(
        f"Actual: {actual:.2f}, "
        f"Predicted: {predicted:.2f}, "
        f"Accuracy: {accuracy * 100:.2f}%"
    )

mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)
accuracy = 1 - np.abs(Y_pred - Y_test) / np.abs(Y_test)

print(f"\nMAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.2f}")
print(f"Average Accuracy: {accuracy.mean() * 100:.2f}%")