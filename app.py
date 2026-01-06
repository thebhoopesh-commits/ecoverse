from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)



def train_gait_model():
    df = pd.read_csv(r"E:\datasets for the hackatho\gait_fall_dataset.csv")

    
    X = df.drop(columns=['label'])
    y_raw = df['label']

    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    print("✅ Model trained successfully")
    print("📌 Encoded labels (index → activity):")
    for i, label in enumerate(le.classes_):
        print(f"   {i} → {label}")

    return model, le, X.columns.tolist()



gait_model, gait_le, gait_features = train_gait_model()



LABEL_MEANING = {
    0: "Standing",
    1: "Walking",
    2: "Sitting",
    3: "Fall"
}


@app.route("/")
def home():
    return render_template("index.html", features=gait_features)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        features_input = request.form.get("features")

        
        values = np.array(
            features_input.split(","), dtype=float
        ).reshape(1, -1)

        
        pred_num = gait_model.predict(values)[0]

        
        activity = LABEL_MEANING.get(pred_num, "Unknown Activity")

        # Prediction confidence
        probabilities = gait_model.predict_proba(values)[0]
        confidence = round(max(probabilities) * 100, 2)

        
        explanation = explain_activity(activity)

        return render_template(
            "results.html",
            activity=activity,
            confidence=confidence,
            explanation=explanation
        )

    except Exception as e:
        return f" Error: {str(e)}"




def explain_activity(activity):
    explanations = {
        "Walking": "Moderate acceleration with smooth and periodic motion patterns.",
        "Sitting": "Very low acceleration values indicating minimal body movement.",
        "Standing": "Stable posture with negligible motion over time.",
        "Fall": "Sudden spike in acceleration followed by rapid inactivity."
    }

    return explanations.get(
        activity,
        "The activity was detected based on motion sensor patterns."
    )



if __name__ == "__main__":
    app.run(debug=True)
