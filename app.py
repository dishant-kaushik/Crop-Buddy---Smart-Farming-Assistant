from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sklearn
import pickle
import os

app = Flask(__name__)

# Import model with error handling
try:
    model = pickle.load(open('rf_model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
except (FileNotFoundError, EOFError) as e:
    print(f"Error loading model files: {e}")
    print("Please ensure model files exist and are not corrupted")

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Convert inputs to float
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
        
        # Input validation
        if not all(0 <= val <= 1000 for val in [N, P, K, temp, humidity, ph, rainfall]):
            return render_template('index.html', result="Please enter valid values within reasonable ranges")

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated right there"
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        
        return render_template('index.html', result=result)
    
    except ValueError:
        return render_template('index.html', result="Please enter valid numerical values for all fields")
    except Exception as e:
        return render_template('index.html', result=f"An error occurred: {str(e)}")

# Python main
if __name__ == "__main__":
    app.run(debug=True)