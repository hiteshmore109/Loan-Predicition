from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle


def log_transform(x):
    return np.log1p(x)


app = Flask(__name__)


@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET', 'POST'])
def predict_loan():
    if request.method == 'GET':
        return render_template('home.html')
    data = {
        "Gender": request.form.get("gender"),
        "Married": request.form.get("married"),
        "Dependents": request.form.get("dependents"),
        "Education": request.form.get("education"),
        "Self_Employed": request.form.get("self_employed"),
        "ApplicantIncome": float(request.form.get("applicant_income")),
        "CoapplicantIncome": float(request.form.get("coapplicant_income")),
        "LoanAmount": float(request.form.get("loan_amount")),
        "Loan_Amount_Term": float(request.form.get("loan_amount_term")),
        "Credit_History": float(request.form.get("credit_history")),
        "Property_Area": request.form.get("property_area")
    }
    
    data_df = pd.DataFrame(data, index=[0])

    with open('artifacts/data_pipeline.pkl', 'rb') as f:
        data_pipeline = pickle.load(f)

    data_preprocessed = data_pipeline.transform(data_df)

    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(data_preprocessed)[0]

    return render_template('home.html', prediction=prediction)


if __name__ == '__main__':
    print("Running the app at http://localhost:5000")
    app.run(debug=True)