from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ================= LOAD & TRAIN MODELS =================

def train_gait_model():
    df = pd.read_csv(r"E:\datasets for the hackatho\gait_fall_dataset.csv")
    X = df.drop(columns=['label'])
    y = LabelEncoder().fit_transform(df['label'])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

def train_har_model():
    df = pd.read_csv(r"E:\datasets for the hackatho\test.csv")
    X = df.drop(columns=['Activity'])
    y = LabelEncoder().fit_transform(df['Activity'])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

def train_motion_model():
    df = pd.read_csv(r"E:\datasets for the hackatho\data_subjects_info.csv")
    X = df.drop(columns=['gender'])
    y = LabelEncoder().fit_transform(df['gender'])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

print("🔄 Training models from CSV...")

gait_model, gait_features = train_gait_model()
har_model, har_features = train_har_model()
motion_model, motion_features = train_motion_model()

print("✅ Models trained successfully")

# ================= FLASK ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    dataset = request.form["dataset"]
    features = request.form["features"]

    input_features = np.array(features.split(","), dtype=float).reshape(1, -1)

    if dataset == "gait":
        prediction = gait_model.predict(input_features)[0]
    elif dataset == "har":
        prediction = har_model.predict(input_features)[0]
    else:
        prediction = motion_model.predict(input_features)[0]

    return render_template("results.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
