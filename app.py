from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)  # DO NOT include template_folder here unless absolutely needed

# Load model and scaler
model = joblib.load("sleep_disorder_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    "Age", "Gender", "Occupation", "Sleep Duration", "Quality of Sleep",
    "Physical Activity Level", "Stress Level", "BMI Category",
    "Heart Rate", "Daily Steps", "Systolic BP", "Diastolic BP"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([{
            "Age": int(data["Age"]),
            "Gender": int(data["Gender"]),
            "Occupation": int(data["Occupation"]),
            "Sleep Duration": float(data["Sleep Duration"]),
            "Quality of Sleep": int(data["Quality of Sleep"]),
            "Physical Activity Level": int(data["Physical Activity Level"]),
            "Stress Level": int(data["Stress Level"]),
            "BMI Category": int(data["BMI Category"]),
            "Heart Rate": int(data["Heart Rate"]),
            "Daily Steps": int(data["Daily Steps"]),
            "Systolic BP": int(data["Systolic BP"]),
            "Diastolic BP": int(data["Diastolic BP"])
        }])

        scaled = scaler.transform(input_data)
        pred = model.predict(scaled)[0]
        label_map = {0: "Insomnia", 1: "Sleep Apnea", 2: "No Disorder"}

        return jsonify({"result": label_map.get(pred, "Unknown")})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
