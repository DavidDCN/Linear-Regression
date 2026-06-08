import numpy as np

class LinearRegressionFromScratch:
    def __init__(self):
        self.slope = 0
        self.intercept = 0

    def fit(self, X, y):
        n = len(X)
        sum_x = np.sum(X)
        sum_y = np.sum(y)
        sum_xy = np.sum(X * y)
        sum_x_squared = np.sum(X ** 2)
        
        # Calculate slope (m) and intercept (b)
        denominator = (n * sum_x_squared) - (sum_x ** 2)
        if denominator == 0:
            raise ValueError("Denominator is zero; cannot compute unique line.")
            
        self.slope = (n * sum_xy - sum_x * sum_y) / denominator
        self.intercept = (sum_y - self.slope * sum_x) / n

    def predict(self, X):
        return self.slope * X + self.intercept

# --- Execution Example ---

# Synthetic data samples: [Size in 1000 sq ft]
X_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([2.0, 4.0, 5.5, 7.0])

# Initialize and train the model
model = LinearRegressionFromScratch()
model.fit(X_train, y_train)

print(f"Trained Model: y = {model.slope:.2f}x + {model.intercept:.2f}")

# Make predictions on new unseen samples
X_new = np.array([2.5, 5.0])
predictions = model.predict(X_new)

for x, pred in zip(X_new, predictions):
    print(f"Prediction for x = {x}: y = {pred:.2f}")