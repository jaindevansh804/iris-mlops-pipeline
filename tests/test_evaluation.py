import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = joblib.load("artifacts/model.joblib")
    df = pd.read_csv("ga_resources/data/raw/iris.csv")

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_true = df["species"]

    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)

    print(f"✅ Model Accuracy: {acc:.3f}")
    assert acc > 0.90, "Model accuracy too low!"
