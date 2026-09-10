import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "cow_milk_mastitis_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
raw_path = os.path.normpath(DATA_PATH)
df = pd.read_csv(f"{DATA_PATH}")

#-----------Data Preparation for KNN-------------
X = df.drop(columns=["Cow_ID", "Day", "class1"]) #Prediction features
Y = df["class1"] #Prediction target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#-----------Data Scaling---------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) #Scaling and unifying the feature data

print("Training:", len(X_train))
print("Testing:", len(X_test))

#----------KNN Model Training and Prediction-------------
k_values = [k for k in range(1, 22, 2)]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train) #Training the KNN model
    Y_pred = knn.predict(X_test) #Predicting the target values using the KNN model

#/-----------Result Display------------- (removed for brevity)
#result = pd.DataFrame({"Actual": Y_test, "Predicted": Y_pred}) #Preparing the result dataframe for display
#print("Training:", len(X_train))
#print("Testing:", len(X_test))

#-----------Accuracy Calculation-----------
    accuracy = np.mean(Y_pred == Y_test)
    print(f"K={k}, Accuracy: {accuracy.mean() * 100:.2f}%")