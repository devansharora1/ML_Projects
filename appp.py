from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
model_path = "big_mart_sale\model.pkl"
with open('model.pkl', "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get input values from the form
            Item_Identifier = float(request.form["Item_Identifier"])
            Item_Weight = float(request.form["Item_Weight"])
            Item_Visibility = float(request.form["Item_Visibility"])
            Item_MRP = float(request.form["Item_MRP"])
            Outlet_Identifier = float(request.form["Outlet_Identifier"])
            Item_Type = float(request.form["Item_Type"])
            Item_Fat_Content = float(request.form["Item_Fat_Content"])
            Outlet_Establishment_Year = float(request.form["Outlet_Establishment_Year"])







            # Make a prediction using the loaded model
            input_data = np.array([[Item_Identifier, Item_Weight .Item_Visibility ,Item_MRP ,Outlet_Identifier ,Item_Type ,Item_Fat_Content ,Outlet_Establishment_Year]])
            prediction = model.predict(input_data)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
