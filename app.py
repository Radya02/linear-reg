from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)

bundle      = joblib.load("model.pkl")
model       = bundle["model"]
le1         = bundle["le1"]
accuracy    = bundle["accuracy"]
mae         = bundle["mae"]
rmse        = bundle["rmse"]
r2          = bundle["r2"]
plot_base64 = bundle["plot_base64"]

df      = pd.read_csv("train_energy_data.csv").drop("Day of Week", axis=1)
samples = df.head(10).to_dict(orient="records")

@app.route("/")
def index():
    return render_template("index.html",
        accuracy=f"{accuracy:.2f}",
        mae=f"{mae:.2f}",
        rmse=f"{rmse:.2f}",
        r2=f"{r2:.4f}",
        samples=samples,
        building_types=list(le1.classes_),
        plot_base64=plot_base64
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.json
        building = data["building_type"].capitalize()
        if building not in le1.classes_:
            return jsonify({"error": "Неверный тип здания"}), 400
        b      = le1.transform([building])[0]
        sqft   = float(data["sqft"])
        occ    = float(data["occupants"])
        appl   = float(data["appliances"])
        temp   = float(data["temperature"])
        X      = np.array([[b, sqft, occ, appl, temp]])
        result = model.predict(X)[0]
        return jsonify({"result": f"{result:.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)