import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score

Path = "C:/Users/Ansel/Desktop/Ansel/Python/ML-CPE67"
df = pd.read_csv(f"{Path}/Lab04/CowMilkMastitisDataset.csv")

X = df.drop(columns=["class1", "Cow_ID", "Day", "Milk_Yield"]) #Prediction features
Y = df["Milk_Yield"] #Prediction target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) #Scaling and unifying the feature data

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, Y_train) #Training the KNN model

Y_pred = knn.predict(X_test) #Predicting the target values using the KNN model
result = pd.DataFrame({"Actual": Y_test, "Predicted": Y_pred})
print("Training:", len(X_train))
print("Testing:", len(X_test))
accuracy = 1 - np.abs(Y_pred - Y_test) / np.abs(Y_test)

for actual, predicted, acc in zip(Y_test, Y_pred, accuracy):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}, Accuracy: {acc * 100:.2f}%")
print(f"Accuracy: {accuracy.mean() * 100:.2f}%")