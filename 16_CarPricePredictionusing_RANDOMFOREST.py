import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load Dataset
dataset = pd.read_csv("dataset.csv")
dataset = dataset.drop("car_ID", axis=1)

# Feature Selection
X_data = dataset.drop("price", axis="columns")
numerical_cols = X_data.select_dtypes(exclude=["object"]).columns
X = X_data[numerical_cols]

# Target Variable
y = dataset["price"]

# Feature Scaling
X = pd.DataFrame(scale(X), columns=numerical_cols)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# Train Model
model = RandomForestRegressor(random_state=0)
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2 * 100:.2f}%")