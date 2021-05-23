from flask import Flask,request,jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)


@app.route('/py_server')
def index():
    return 'Running Python Flask Server'


cancer_model = load_model('breast_cancer_model.h5')
cancer_scaler = joblib.load('breast_cancer_scaler.pkl')

@app.route('/api/cancer_prediction',methods=['GET'])
def cancer_prediction_api():
    content = request.json
    results = cancer_prediction(content,cancer_model,cancer_scaler)
    response = jsonify(results)
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


def cancer_prediction(data,model,scaler):
    
    CDH1_TC = 0
    CDH1_TT = 0
    PTEN_CT = 0
    PTEN_TT = 0
    TP53_TC = 0
    TP53_TT = 0
    ATM_GA = 0
    ATM_GG = 0
    ATM_1_TC = 0
    ATM_1_TT = 0
    ATM_2_GT = 0
    ATM_2_TT = 0
    NBN_AG = 0
    NBN_GG = 0
    NBN_1_CG = 0
    NBN_1_GG = 0
    NBN_2_GA = 0
    NBN_2_GG = 0
    AKT1_CA = 0
    AKT1_CC = 0
    BRCA1_GA = 0
    BRCA1_GG = 0
    BRCA2_AG = 0
    BRCA2_GG = 0
    BRCA2_1_CA = 0
    BRCA2_1_CC = 0
    BRCA2_2_CA = 0
    BRCA2_2_CC = 0
    BRCA2_3_AG = 0
    BRCA2_3_GG = 0

    
    if data['CDH1'] == 'TC':
        CDH1_TC = 1
    if data['CDH1'] == 'TT':
        CDH1_TT = 1
    if data['PTEN'] == 'CT':
        PTEN_CT = 1
    if data['PTEN'] == 'TT':
        PTEN_TT = 1
    if data['TP53'] == 'TC':
        TP53_TC = 1
    if data['TP53'] == 'TT':
        TP53_TT = 1
    if data['ATM'] == 'GA':
        ATM_GA = 1
    if data['ATM'] == 'GG':
        ATM_GG = 1
    if data['ATM_1'] == 'TC':
        ATM_1_TC = 1
    if data['ATM_1'] == 'TT':
        ATM_1_TT = 1
    if data['ATM_2'] == 'GT':
        ATM_2_GT = 1
    if data['ATM_2'] == 'TT':
        ATM_2_TT = 1
    if data['NBN'] == 'AG':
        NBN_AG = 1
    if data['NBN'] == 'GG':
        NBN_GG = 1
    if data['NBN_1'] == 'CG':
        NBN_1_CG = 1
    if data['NBN_1'] == 'GG':
        NBN_1_GG = 1
    if data['NBN_2'] == 'GA':
        NBN_2_GA = 1
    if data['NBN_2'] == 'GG':
        NBN_2_GG = 1
    if data['AKT1'] == 'CA':
        AKT1_CA = 1
    if data['AKT1'] == 'CC':
        AKT1_CC = 1
    if data['BRCA1'] == 'GA':
        BRCA1_GA = 1
    if data['BRCA1'] == 'GG':
        BRCA1_GG = 1
    if data['BRCA2'] == 'AG':
        BRCA2_AG = 1
    if data['BRCA2'] == 'GG':
        BRCA2_GG = 1
    if data['BRCA2_1'] == 'CA':
        BRCA2_1_CA = 1
    if data['BRCA2_1'] == 'CC':
        BRCA2_1_CC = 1
    if data['BRCA2_2'] == 'CA':
        BRCA2_2_CA = 1
    if data['BRCA2_2'] == 'CC':
        BRCA2_2_CC = 1
    if data['BRCA2_3'] == 'AG':
        BRCA2_3_AG = 1
    if data['BRCA2_3'] == 'GG':
        BRCA2_3_GG = 1
    
    
    prediction_data = [[CDH1_TC, CDH1_TT, PTEN_CT, PTEN_TT,
        TP53_TC , TP53_TT, ATM_GA , ATM_GG , ATM_1_TC, ATM_1_TT ,
        ATM_2_GT, ATM_2_TT, NBN_AG, NBN_GG, NBN_1_CG, NBN_1_GG,
        NBN_2_GA, NBN_2_GG, AKT1_CA, AKT1_CC, BRCA1_GA, BRCA1_GG,
        BRCA2_AG, BRCA2_GG, BRCA2_1_CA, BRCA2_1_CC, BRCA2_2_CA, 
        BRCA2_2_CC, BRCA2_3_AG, BRCA2_3_GG,
    ]]

    scaled_pred_data = scaler.transform(prediction_data)
    result = model.predict(scaled_pred_data)     

    return str(result[0][0])

    # if result[0] == 0:
    #     return 'Cancer not detected'
    # else:
    #     return 'Cancer Detected'



if __name__ == '__main__':
    app.run()