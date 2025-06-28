import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scale = pickle.load(open("scale.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    try:
        names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year',
                 'month', 'day', 'hours', 'minutes', 'seconds']
        
        input_feature = [float(request.form[name]) for name in names]
        features_values = np.array([input_feature])
        
        data = pd.DataFrame(features_values, columns=names)
        data_scaled = scale.transform(data)
        
        prediction = model.predict(data_scaled)
        text = "Estimated Traffic Volume is: "
        
        return render_template("index.html", prediction_text=text + str(prediction[0]))
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
