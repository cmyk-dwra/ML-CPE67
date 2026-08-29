from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from data_loader import load_data, prepare_features


# ---------- Load Data ----------
df = load_data()

# ---------- Prepare Data ----------
X = prepare_features(df)
Y = df["class1"]

# ---------- Train / Test Split ----------
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# ---------- Scaling ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- PCA ----------
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# ---------- Logistic Regression ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# ---------- Evaluation ----------
for actual, predicted in zip(Y_test, Y_pred):
    accuracy = 1 - abs(predicted - actual) / abs(actual)

    print(
        f"Actual: {actual:.2f}, "
        f"Predicted: {predicted:.2f}, "
    )

accuracy = accuracy_score(Y_test, Y_pred)

print("=== Mastitis Classification ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))