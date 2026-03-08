from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_iris()
X = data.data      # features
y = data.target    # labels

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create model
model = DecisionTreeClassifier()

# 4. Train model
model.fit(X_train, y_train)

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Predictions:", predictions)
print("Actual:", y_test)
print("Accuracy:", accuracy)

# 7. Test with new data
sample = [[5.1, 3.5, 1.4, 0.2]]
result = model.predict(sample)

print("Prediction for sample:", result)
