import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

Path = "C:/Users/Ansel/Desktop/Ansel/Python/ML-CPE67"
df = pd.read_csv(f"{Path}/Lab04/CowMilkMastitisDataset.csv")

#-----------Data Preparation for KNN-------------
X = df.drop(columns=["Cow_ID", "Day", "Milk_Yield"]) #Prediction features
Y = df["Milk_Yield"] #Prediction target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#-----------Data Scaling---------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) #Scaling and unifying the feature data

#----------KNN Model Training and Prediction-------------
k_values = [3,5,7,9,11]
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, Y_train) #Training the KNN model
    Y_pred = knn.predict(X_test) #Predicting the target values using the KNN model

#-----------Result Display------------- (removed for brevity)
#result = pd.DataFrame({"Actual": Y_test, "Predicted": Y_pred}) #Preparing the result dataframe for display
#print("Training:", len(X_train))
#print("Testing:", len(X_test))

#-----------Accuracy Calculation-----------
    accuracy = 1 - np.abs(Y_pred - Y_test) / np.abs(Y_test)
    print(f"K={k}, Accuracy: {accuracy.mean() * 100:.2f}%")