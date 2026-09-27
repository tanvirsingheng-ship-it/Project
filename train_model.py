import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "DataCoSupplyChainDataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "late_delivery_model.pkl"
)


os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

print("Dataset shape:", df.shape)

features = [
    "Shipping Mode",
    "Order Item Discount Rate",
    "Sales",
    "Order Item Product Price",
    "Order Item Quantity",
    "Market",
    "Order Region",
    "Department Name",
    "Category Name",
    "Customer Country"
]

target = "Late_delivery_risk"


missing_columns = [
    column
    for column in features + [target]
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: These columns are missing:")
    print(missing_columns)
    raise SystemExit


data = df[features + [target]].copy()

print("Training data shape:", data.shape)

print("\nHandling missing values...")

for column in features:

    if data[column].dtype == "object" or pd.api.types.is_string_dtype(data[column]):

        data[column] = data[column].fillna("Unknown")

    else:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        data[column] = data[column].fillna(
            data[column].median()
        )


print("Missing values handled.")

categorical_columns = [
    column
    for column in features
    if (
        data[column].dtype == "object"
        or pd.api.types.is_string_dtype(data[column])
    )
]

print("\nCategorical columns:")
print(categorical_columns)


data = pd.get_dummies(
    data,
    columns=categorical_columns,
    drop_first=True
)

X = data.drop(
    columns=[target]
)

y = data[target]


print("\nFinal feature shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print("Accuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)

model_package = {
    "model": model,
    "features": list(X.columns)
}

joblib.dump(
    model_package,
    MODEL_PATH
)


print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print("Model location:")
print(MODEL_PATH)

print("\nModel file exists:")

print(
    os.path.exists(MODEL_PATH)
)