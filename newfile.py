# Medical Insurance Cost Prediction

# ==============================
# 1. Import Libraries
# ==============================

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================
# 2. Load Dataset
# ==============================

df = pd.read_csv("insurance.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ==============================
# 3. Basic Statistics
# ==============================

print("\nStatistical Summary:")
print(df.describe())


# ==============================
# 4. Data Preprocessing
# ==============================

# Convert categorical columns into numbers

le = LabelEncoder()

df["sex"] = le.fit_transform(df["sex"])
df["smoker"] = le.fit_transform(df["smoker"])
df["region"] = le.fit_transform(df["region"])

print("\nAfter Encoding:")
print(df.head())


# ==============================
# 5. Bar Chart
# Average Insurance Charges by Smoker
# ==============================

smoker_charges = df.groupby("smoker")["charges"].mean()

plt.figure(figsize=(7, 5))
plt.bar(["Non-Smoker", "Smoker"], smoker_charges.values)

plt.title("Average Medical Insurance Charges by Smoker")
plt.xlabel("Smoker")
plt.ylabel("Average Charges")

plt.show()


# ==============================
# 6. Line Chart
# Average Charges by Age
# ==============================

age_charges = df.groupby("age")["charges"].mean()

plt.figure(figsize=(9, 5))
plt.plot(age_charges.index, age_charges.values, marker="o")

plt.title("Average Medical Insurance Charges by Age")
plt.xlabel("Age")
plt.ylabel("Average Charges")
plt.grid(True)

plt.show()


# ==============================
# 7. Pie Chart
# Smoker Distribution
# ==============================

smoker_count = df["smoker"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    smoker_count.values,
    labels=["Non-Smoker", "Smoker"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Smoker Distribution")

plt.show()


# ==============================
# 8. Heatmap
# Correlation Matrix
# ==============================

plt.figure(figsize=(9, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# ==============================
# 9. Prepare Data for ML
# ==============================

X = df.drop("charges", axis=1)
y = df["charges"]


# ==============================
# 10. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# 11. Train Model
# ==============================

model = LinearRegression()

model.fit(X_train, y_train)


# ==============================
# 12. Prediction
# ==============================

y_pred = model.predict(X_test)


# ==============================
# 13. Model Evaluation
# ==============================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


# ==============================
# 14. Actual vs Predicted
# ==============================

result = pd.DataFrame({
    "Actual Charges": y_test.values,
    "Predicted Charges": y_pred
})

print("\nActual vs Predicted:")
print(result.head(10))


# ==============================
# 15. Prediction for New Person
# ==============================

# Example:
# age = 25
# sex = 1
# bmi = 27.5
# children = 1
# smoker = 0
# region = 2

new_person = pd.DataFrame({
    "age": [25],
    "sex": [1],
    "bmi": [27.5],
    "children": [1],
    "smoker": [0],
    "region": [2]
})

prediction = model.predict(new_person)

print("\nPredicted Medical Insurance Cost:")
print(prediction[0])