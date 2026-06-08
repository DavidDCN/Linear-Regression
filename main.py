import pandas as pd
import numpy as np

# --- 1. The Algorithm ---
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

# --- 2. Load and Prepare the Data ---
print("Loading dataset...")
# Load the CSV file
df = pd.read_csv('house_price_regression_dataset.csv')

# Split the data into Train and Test sets using the 'Dataset_Split' column
train_df = df[df['Dataset_Split'] == 'Train']
test_df = df[df['Dataset_Split'] == 'Test']

# Extract features (X) and targets (y) as numpy arrays
X_train = train_df['Size_1000_sqft'].values
y_train = train_df['Actual_Price_100k'].values

X_test = test_df['Size_1000_sqft'].values
y_test = test_df['Actual_Price_100k'].values

# --- 3. Train the Model ---
print("Training the model...")
model = LinearRegressionFromScratch()
model.fit(X_train, y_train)

print(f"Model trained successfully!")
print(f"Line of Best Fit: y = {model.slope:.2f}x + {model.intercept:.2f}\n")

# --- 4. Test the Model ---
print("Making predictions on the Test data...")
predictions = model.predict(X_test)

# --- 5. Evaluate the Results ---
# Create a new DataFrame just to display the results nicely
results_df = pd.DataFrame({
    'Size (1000 sqft)': X_test,
    'Actual Price ($100k)': y_test,
    'Predicted Price ($100k)': np.round(predictions, 2),
    'Absolute Error': np.round(abs(y_test - predictions), 2)
})

print("Test Results:")
print("-" * 60)
print(results_df.to_string(index=False))
print("-" * 60)

# Calculate Mean Absolute Error (MAE)
mae = np.mean(abs(y_test - predictions))
print(f"Mean Absolute Error (MAE): {mae:.2f} (This means our predictions are off by an average of ${mae*100000:,.0f})")