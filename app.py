import joblib
import pandas as pd
import numpy as np
import sklearn.impute
import pickle
from sklearn import impute
from sklearn.experimental import enable_iterative_imputer
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediksi', methods=['POST'])
def prediksi():
    if request.method == 'POST':
        loaded_model = pickle.load(open('RFR.pkl', 'rb'))

        def predict_pengunjung(year, month_name):
            def month_to_number(month_name):
                month_mapping = {
                    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5,
                    "Juni": 6, "Juli": 7, "Agustus": 8, "September": 9, 
                    "Oktober": 10, "November": 11, "Desember": 12
                }
                return month_mapping.get(month_name)

            month_number = month_to_number(month_name)
            
            if month_number is None:
                raise ValueError("Invalid month name")

            input_data = pd.DataFrame({'Tahun': [year], 'Bulan': [month_number]})
            predicted_pengunjung = loaded_model.predict(input_data)

            return predicted_pengunjung[0]

        bulan = request.form['bulan']
        tahun = int(request.form['tahun'])
        predicted_value = predict_pengunjung(tahun, bulan)
        
        return render_template('index.html', bulan=bulan, tahun=tahun, predicted_value=predicted_value)

if __name__ == '__main__':
    app.run(debug=True)
