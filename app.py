from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("spam_model.pkl")

THRESHOLD = 0.4

@app.route("/")
def home():
    return "Spam Detection API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    message = data.get("message", "")

    prob = model.predict_proba([message])[0][1]
    label = "SPAM" if prob >= THRESHOLD else "HAM"

    return jsonify({
        "message": message,
        "spam_probability": round(float(prob), 3),
        "prediction": label
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
