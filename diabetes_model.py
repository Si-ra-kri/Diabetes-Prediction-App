import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the iris dataset
df = pd.read_csv("diabetes.csv")

x = df.drop("Outcome", axis=1)
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a random forest classifier
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Save the trained model
joblib.dump(model, 'model.joblib')
