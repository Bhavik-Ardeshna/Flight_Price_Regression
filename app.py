from flask import Flask, render_template, request
import requests
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import pickle5 as pickle

app = Flask(__name__)

model = pickle.load(open('./Model/random_forest_regression_model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    prediction = model.predict([[3, 2, 5, 1000, 2, 9, 5]])
    output = round(prediction[0], 2)
    print(output)
    return render_template('booking.html')


@app.route('/predict', methods=['POST'])
def Predict():
    if request.method == 'POST':
        source = int(request.form.get("source"))
        dest = int(request.form.get("dest"))
        airline = int(request.form.get("airline"))
        stop = int(request.form.get("stop"))
        duration = int(request.form.get("duration"))
        day = int(request.form.get("day"))
        month = int(request.form.get("month"))
        prediction = model.predict(
            [[airline, source, dest, duration, stop, day, month]])
        output = round(prediction[0], 2)

        if output < 0:
            return render_template('booking.html', prediction_texts="Sorry we can\'t predict flight price.")
        else:
            return render_template('booking.html', prediction_texts=output)

    return render_template('booking.html')


if __name__ == "__main__":
    app.run(debug=True)
