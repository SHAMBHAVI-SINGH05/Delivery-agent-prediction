from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
STATIC_DIR = os.path.join(BASE_DIR, "static")
FEATURE_COLUMNS = [
    "Date",
    "Day",
    "Hour",
    "Temperature (°C)",
    "Rain",
    "Holiday",
    "Offer Active",
    "Competitor",
    "Orders",
    "Promised Time (min)",
    "Actual Time (min)",
    "Delivered On Time",
    "Early By (min)",
    "Delay By (min)",
    "Timing Deviation Cost",
    "Company_JioMart",
    "Company_Swiggy",
    "Company_Zepto",
    "Traffic Level_Low",
    "Traffic Level_Medium",
    "Order Type_Food",
    "Order Type_Grocery",
    "Area Type_Urban"
]

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static')

MODEL_PATH = os.path.join(BASE_DIR, "lasso_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    graph_generated = False  # Track graph creation

    if request.method == "POST":
        try:
            sample = pd.DataFrame([
                [
                    int(request.form.get("Date")),
                    int(request.form.get("Day")),
                    int(request.form.get("Hour")),
                    float(request.form.get("Temperature")),
                    int(request.form.get("Rain")),
                    int(request.form.get("Holiday")),
                    int(request.form.get("Offer")),
                    int(request.form.get("Competitor")),
                    int(request.form.get("Orders")),
                    int(request.form.get("Promised")),
                    int(request.form.get("Actual")),
                    int(request.form.get("Delivered")),
                    int(request.form.get("Early")),
                    int(request.form.get("Delay")),
                    int(request.form.get("Cost")),
                    int(request.form.get("JioMart")),
                    int(request.form.get("Swiggy")),
                    int(request.form.get("Zepto")),
                    int(request.form.get("TrafficLow")),
                    int(request.form.get("TrafficMedium")),
                    int(request.form.get("Food")),
                    int(request.form.get("Grocery")),
                    int(request.form.get("Urban"))
                ]
            ], columns=FEATURE_COLUMNS)

            # Predict
            pred = model.predict(sample)[0]
            prediction = round(pred)

            # Generate a dummy graph (replace this with real y_test and y_pred if needed)
            y_test = list(range(50))
            y_pred = [i + np.random.randint(-3, 3) for i in y_test]  # simulated output

            plt.figure(figsize=(8, 5))
            plt.plot(y_test, label="Actual", marker='o')
            plt.plot(y_pred, label="Predicted", marker='x')
            plt.title("Actual vs Predicted Delivery Agents")
            plt.xlabel("Sample Index")
            plt.ylabel("Delivery Agents")
            plt.legend()
            plt.tight_layout()

            # Save plot to static folder
            os.makedirs(STATIC_DIR, exist_ok=True)
            graph_path = os.path.join(STATIC_DIR, "graph_actual_vs_predicted.png")
            plt.savefig(graph_path)
            plt.close()
            graph_generated = True

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("form.html", prediction=prediction, graph=graph_generated)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)