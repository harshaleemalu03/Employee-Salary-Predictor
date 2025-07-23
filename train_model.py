import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
data = pd.read_csv("adult.csv")

# Drop rows with missing values
data.replace("?", pd.NA, inplace=True)
data.dropna(inplace=True)

# Encode categorical variables
X = data.drop("income", axis=1)
X_encoded = pd.get_dummies(X)

# Align with income column
y = data["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

# Balance the dataset
from sklearn.utils import resample
df = pd.concat([X_encoded, y], axis=1)
majority = df[df['income'] == 0]
minority = df[df['income'] == 1]
majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
balanced_df = pd.concat([majority_downsampled, minority])

X_balanced = balanced_df.drop("income", axis=1)
y_balanced = balanced_df["income"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and column info
joblib.dump(model, "model.joblib")
joblib.dump(X_encoded.columns.tolist(), "columns.joblib")
