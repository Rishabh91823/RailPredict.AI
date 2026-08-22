from datetime import datetime
import json
import pandas as pd
import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
MODEL_FILE = 'railway_model_1cr.pkl'

try:
    model = joblib.load(MODEL_FILE)
except Exception:
    model = None
HIGH_DEMAND = {
    'NDLS', 'NZM', 'DLI', 'ANVT', 'DEE',
    'CSMT', 'BCT', 'BDTS', 'LTT', 'DR', 'KYN', 'TNA',
    'HWH', 'SDAH', 'KOAA', 'SRC',
    'MAS', 'MS', 'TBM', 'CGL', 'CBE', 'MDU', 'TPJ',
    'SBC', 'YPR', 'BNC', 'UBL', 'MYS',
    'SC', 'HYB', 'KCG', 'KZJ',
    'PUNE', 'NGP', 'SUR', 'NK', 'BSL',
    'LKO', 'CNB', 'PRYJ', 'ALD', 'AGC', 'GZB', 'MGS', 'DDU', 'BSB',
    'PNBE', 'RJPB', 'GAYA', 'MFP', 'BJU', 'DBG', 'KIR',
    'ERS', 'ERN', 'TVC', 'KTYM', 'TCR', 'CLT', 'CAN', 'QLN', 'SRR',
    'JP', 'AII', 'JU', 'KOTA', 'BKN', 'AWR',
    'ADI', 'ST', 'BRC', 'RJT', 'BCT',
    'BPL', 'ET', 'JBP', 'INDB', 'RTM', 'R', 'BSP',
    'ASR', 'LDH', 'JUC', 'JRC', 'UMB', 'CDG', 'BTI',
    'HWH', 'SDAH', 'KGP', 'ASN', 'NJP', 'MLDT',
    'BBS', 'CTC', 'KUR', 'TATA', 'RNC', 'DHN', 'BKSC',
    'VSKP', 'BZA', 'GNT', 'TPTY', 'RU', 'GDR', 'OGL'
}

STATION_STATES = {
    'NDLS': 'Delhi', 'ANVT': 'Delhi', 'NZM': 'Delhi',
    'CSMT': 'Maharashtra', 'BCT': 'Maharashtra', 'LTT': 'Maharashtra', 'BDTS': 'Maharashtra', 'PUNE': 'Maharashtra', 'NGP': 'Maharashtra', 'NK': 'Maharashtra', 'BSL': 'Maharashtra', 'KYN': 'Maharashtra', 'ST': 'Maharashtra',
    'HWH': 'West_Bengal', 'SDAH': 'West_Bengal', 'KOAA': 'West_Bengal', 'BWN': 'West_Bengal', 'HDB': 'West_Bengal', 'ASN': 'West_Bengal', 'MLDT': 'West_Bengal', 'BGP': 'West_Bengal',
    'MAS': 'Tamil_Nadu', 'MS': 'Tamil_Nadu',
    'SBC': 'Karnataka', 'YPR': 'Karnataka',
    'SC': 'Telangana', 'HYB': 'Telangana',
    'PNBE': 'Bihar', 'DNR': 'Bihar', 'BJU': 'Bihar', 'MFP': 'Bihar', 'GAYA': 'Bihar',
    'LKO': 'Uttar_Pradesh', 'CNB': 'Uttar_Pradesh', 'PRYJ': 'Uttar_Pradesh', 'BSB': 'Uttar_Pradesh', 'GKP': 'Uttar_Pradesh',
    'ERS': 'Kerala', 'TVC': 'Kerala', 'KTYM': 'Kerala', 'TCR': 'Kerala', 'CLT': 'Kerala',
    'ADI': 'Gujarat', 'ST': 'Gujarat'
}

FESTIVALS = {
    'Bihar': (10, 11),    
    'Kerala': (8, 9),            
    'West_Bengal': (9, 10),      
    'Maharashtra': (8, 9),       
    'Uttar_Pradesh': (10, 11),   
    'Tamil_Nadu': (1, 1)         
}
CLASS_PENALTY = {0: 15, 1: 0, 2: -10, 3: -30}
FEATURES = ['route_tier', 'days_left', 'waitlist_num', 'train_type', 'class', 'quota', 'is_festival']

def days_left(date):
    return (datetime.strptime(date, '%Y-%m-%d') - datetime.today()).days


def festival(src, dest, date):
    month = datetime.strptime(date, '%Y-%m-%d').month
    return int(any(
        state in FESTIVALS and FESTIVALS[state][0] <= month <= FESTIVALS[state][1]
        for state in {STATION_STATES.get(src), STATION_STATES.get(dest)} if state
    ))


@app.get('/')
def home():
    return render_template('index.html')


@app.get('/stations.json')
def stations():
    with open('stations.json', encoding='utf-8') as f:
        return jsonify(json.load(f))


@app.post('/predict')
def predict():
    try:
        data = request.get_json() or {}
        src, dest = data.get('source_station', ''), data.get('destination_station', '')
        date = data.get('journey_date', datetime.today().strftime('%Y-%m-%d'))
        wl = int(data.get('waitlist_number', 10))
        cls = int(data.get('coach_type', 1))
        quota = int(data.get('quota', 0))
        days = days_left(date)

        tier = int(src in HIGH_DEMAND or dest in HIGH_DEMAND) + 1
        fest = festival(src, dest, date)
        features = pd.DataFrame([[tier, days, wl, 0, cls, quota, fest]], columns=FEATURES)
    
        base = model.predict_proba(features)[0][1] * 100 if model else 95 

        if wl > 40:
            wl_penalty = 50 + (wl - 40) * 0.1 
        else:
            wl_penalty = wl * 0.2 
        
        time_penalty = 40 if days < 5 else 0
        
        prob = base - wl_penalty - time_penalty + CLASS_PENALTY.get(cls, 0) - (50 * fest)
        prob = round(max(1.5, min(98.5, float(prob))), 2)

        return jsonify(success=True, probability=prob,
                       status='High Chance of Confirmation' if prob > 50 else 'High Risk of Waitlist / Cancellation')
    except Exception as e:
        return jsonify(success=False, error=str(e)), 400


if __name__ == '__main__':
    app.run(debug=True)
