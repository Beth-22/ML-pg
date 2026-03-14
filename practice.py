

# ===============================
# 1. IMPORT LIBRARIES
# ===============================

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ===============================
# 2. LOAD DATASET
# ===============================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Dataset shape:", X.shape)
print("Classes:", target_names)


# ===============================
# 3. CONVERT TO DATAFRAME
# ===============================

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("\nFirst 5 rows:")
print(df.head())


# ===============================
# 4. TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ===============================
# 5. FEATURE SCALING
# ===============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===============================
# 6. TRAIN MULTIPLE MODELS
# ===============================

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

results = {}

for name, model in models.items():

    print("\n====================")
    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print("Accuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


# ===============================
# 7. BEST MODEL
# ===============================

best_model_name = max(results, key=results.get)

print("\n====================")
print("Best Model:", best_model_name)
print("Best Accuracy:", results[best_model_name])


# ===============================
# 8. MAKE NEW PREDICTION
# ===============================

sample = np.array([[5.1, 3.5, 1.4, 0.2]])

sample = scaler.transform(sample)

best_model = models[best_model_name]

prediction = best_model.predict(sample)

print("\nNew Sample Prediction:", target_names[prediction][0])


# ===============================
# 9. SIMPLE USER INPUT
# ===============================

print("\nTry your own prediction")

sl = float(input("Sepal length: "))
sw = float(input("Sepal width: "))
pl = float(input("Petal length: "))
pw = float(input("Petal width: "))

user_data = np.array([[sl, sw, pl, pw]])
user_data = scaler.transform(user_data)

pred = best_model.predict(user_data)

print("Predicted class:", target_names[pred][0])
